# KV Cache Monitoring for DQN State Space Integration

## 概述

本文档说明如何：
1. 配置 vLLM 的 KV Cache 块大小 (Block Size)
2. 将 KV Cache 监控与 GPU/CPU 能耗监控同步
3. 构建用于 DQN 的状态空间

---

## 1. 修改 KV Cache Block Size 配置

### 方法一：通过命令行参数（推荐）

启动 vLLM 服务器时使用 `--block-size` 参数：

```bash
# 默认值是 16，可以设置为 1, 8, 16, 32, 64, 128, 256
vllm serve /path/to/model --block-size 32
```

### 方法二：通过环境变量

```bash
export VLLM_CACHE_BLOCK_SIZE=32
vllm serve /path/to/model
```

### 示例：启动 Qwen3-8B 服务器并自定义 Block Size

```bash
#!/bin/bash

# 配置
MODEL="/share-data/wzk-1/model/deepseek-v2-lite"
BLOCK_SIZE=32  # 可选: 1, 8, 16, 32, 64, 128, 256
GPU_MEMORY_UTIL=0.9

# 启动服务器
vllm serve $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --block-size $BLOCK_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code
```

### Block Size 选择指南

| Block Size | Tokens/Block | 适用场景 | 内存效率 | 延迟 |
|------------|--------------|----------|----------|------|
| 8          | 8            | 短文本，低延迟要求 | 较低 | 最低 |
| **16**     | **16**       | **默认值，通用场景** | **中等** | **低** |
| 32         | 32           | 中长文本 | 较高 | 中等 |
| 64         | 64           | 长文本，高吞吐量 | 高 | 较高 |
| 128        | 128          | 超长文本（CPU模式）| 很高 | 高 |

---

## 2. KV Cache 监控与 GPU/CPU 协同

### 监控频率说明

**重要**：KV Cache 监控使用**较长的采样间隔**（与 GPU/CPU 不同）：

```bash
# run_benchmark.sh 配置
GPU_MONITOR_INTERVAL=0.05  # GPU/CPU: 50ms 高频采样（20 Hz）
# KV Cache 自动使用 0.5s 间隔（2 Hz）
```

**为什么不同步？**

1. **数据更新频率差异**：
   - GPU/CPU 功耗：硬件传感器实时更新，可以 50ms 采样
   - KV Cache：vLLM 聚合统计，更新较慢，50ms 采样会获得大量重复值

2. **HTTP 请求开销**：
   - KV Cache 需要通过 HTTP 请求 `/metrics` 端点
   - 50ms 间隔意味着每秒 20 次请求，可能超过服务器处理能力

3. **DQN 实际需求**：
   - KV Cache 状态变化相对缓慢（秒级）
   - 0.5s 采样足以捕获有意义的状态变化

**采样间隔设置**：
- GPU/CPU 功耗：0.05s（50ms）- 捕获瞬时功耗波动
- KV Cache：0.5s（500ms）- 捕获缓存使用趋势

### 启用 KV Cache 监控

```bash
cd /home/user/offload/FUEL/benchmarks

python benchmark_serving.py \
    --backend vllm \
    --model /share-data/wzk-1/model/deepseek-v2-lite \
    --host localhost \
    --port 8000 \
    --dataset-name sharegpt \
    --dataset-path /share-data/wzk-1/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 \
    --request-rate 10 \
    --gpu-ids 0 1 2 3 \
    --gpu-monitor-interval 0.05 \  # KV Cache 采样间隔将匹配这个值
    --monitor-cpu \
    --enable-kv-trace \             # 启用 KV Cache 时序监控
    --save-result \
    --result-dir /home/user/offload/FUEL/benchmark_results
```

---

## 3. 输出数据结构

### Benchmark 结果 JSON 文件包含

```json
{
  "gpu_power_stats": {
    "0": {
      "avg_power": 250000,  // mW
      "avg_gpu_util": 85.5,  // %
      "avg_mem_util": 45.2,  // %
      "power_trace": [...]   // 时间序列
    }
  },
  "cpu_power_stats": {
    "avg_power_w": 45.5,
    ...
  },
  "kv_cache_monitoring_stats": {
    "avg_stats": {
      "avg_cache_usage_perc": 42.5,      // 平均缓存使用率 %
      "avg_used_blocks": 1250,           // 平均已用块数
      "avg_free_blocks": 750,            // 平均空闲块数
      "avg_used_tokens": 20000,          // 平均已用 Token 数
      "avg_requests_running": 5.2,       // 平均运行请求数
      "avg_requests_waiting": 1.3        // 平均等待请求数
    },
    "detailed_stats": {
      "cache_usage": {
        "min": 15.2,
        "p25": 35.8,
        "median": 42.5,
        "p75": 52.1,
        "p95": 68.3,
        "max": 75.5,
        "std": 12.3
      },
      "static_config": {
        "total_gpu_blocks": 2000,
        "block_size": 16,
        "tokens_per_block": 16,
        "total_kv_cache_tokens": 32000,
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_size": 128
      }
    },
    "trace": {
      "cache_usage": [12.5, 15.3, ...],       // 时间序列，与 GPU/CPU 对齐
      "used_blocks": [250, 300, ...],
      "free_blocks": [1750, 1700, ...],
      "used_tokens": [4000, 4800, ...],
      "requests_running": [4, 5, 6, ...],
      "requests_waiting": [0, 1, 2, ...]
    },
    "monitoring_interval_seconds": 0.05,
    "samples_collected": 1000                  // 采样点数量
  }
}
```

---

## 4. 构建 DQN 状态空间

### 4.1 状态向量定义

结合 GPU、CPU 和 KV Cache 监控数据，构建统一的状态空间：

```python
import numpy as np

class LLMInferenceEnvironment:
    """
    大模型推理环境的 DQN 状态空间（优化版）
    
    基于实际监控数据优化：
    - Prefix Caching 导致 KV Cache 使用率很低（0.1%-10%）
    - requests_running 变化范围大（0-20），信息量更丰富
    - 增加 prefix cache 命中率等关键指标
    """
    
    def __init__(self, kv_cache_monitor, gpu_monitor, cpu_monitor):
        self.kv_cache_monitor = kv_cache_monitor
        self.gpu_monitor = gpu_monitor
        self.cpu_monitor = cpu_monitor
        
    def get_state(self) -> np.ndarray:
        """
        获取当前状态向量，用于 DQN
        
        Returns:
            state: shape (10,) 的归一化状态向量
        
        状态维度设计（基于实际数据分析）：
        --------------------------------------------------
        维度  | 指标                    | 范围      | 归一化方式
        --------------------------------------------------
        0     | GPU 总功率              | 0-1000W   | /1000
        1     | GPU 平均利用率          | 0-100%    | /100
        2     | GPU 显存利用率          | 0-100%    | /100
        3     | CPU 功率                | 0-200W    | /200
        4     | CPU 利用率              | 0-100%    | /100
        5     | 运行中的请求数          | 0-50      | /50  (实测 0-20)
        6     | 等待中的请求数          | 0-50      | /50
        7     | KV Cache 使用率         | 0-1       | 直接使用
        8     | 请求吞吐率 (req/s)      | 0-20      | /20
        9     | Token 吞吐率 (tok/s)    | 0-2000    | /2000
        --------------------------------------------------
        总维度: 10
        
        注意：
        1. KV Cache 使用率在启用 Prefix Caching 时很低（<10%），
           但仍保留作为状态特征
        2. requests_running 是最关键的负载指标，变化范围 0-20
        3. 移除了信息量较小的指标（如 kv_blocks_used_ratio）
        """
        state_components = []
        
        # 1. GPU 状态 (3 维) - 能耗和资源利用
        gpu_state = self.get_gpu_state()
        state_components.extend([
            gpu_state.get('total_power', 0) / 1000,      # [0-1] 多GPU总功率
            gpu_state.get('avg_utilization', 0) / 100,   # [0-1] 平均GPU利用率
            gpu_state.get('avg_memory_util', 0) / 100,   # [0-1] 平均显存利用率
        ])
        
        # 2. CPU 状态 (2 维) - 能耗和利用率
        cpu_state = self.get_cpu_state()
        state_components.extend([
            cpu_state.get('power', 0) / 200,             # [0-1] CPU功率
            cpu_state.get('utilization', 0) / 100,       # [0-1] CPU利用率
        ])
        
        # 3. vLLM 请求队列状态 (2 维) - 最关键的负载指标
        kv_state = self.kv_cache_monitor.get_current_state()
        state_components.extend([
            min(kv_state.get('num_requests_running', 0) / 50, 1.0),  # [0-1] 运行请求数
            min(kv_state.get('num_requests_waiting', 0) / 50, 1.0),  # [0-1] 等待请求数
        ])
        
        # 4. KV Cache 状态 (1 维) - 内存使用
        state_components.append(
            kv_state.get('cache_usage_perc', 0)          # [0-1] 已归一化
        )
        
        # 5. 吞吐量状态 (2 维) - 性能指标
        state_components.extend([
            min(self.get_request_throughput() / 20, 1.0),  # [0-1] 请求吞吐
            min(self.get_token_throughput() / 2000, 1.0),  # [0-1] Token吞吐
        ])
        
        return np.array(state_components, dtype=np.float32)
    
    def get_state_dim(self) -> int:
        """
        返回状态空间维度
        
        GPU (3) + CPU (2) + Requests (2) + KV Cache (1) + Throughput (2) = 10 维
        """
        return 10
    
    def get_gpu_state(self) -> dict:
        """获取 GPU 状态（多卡汇总）"""
        if hasattr(self, 'gpu_monitor') and self.gpu_monitor:
            # 从 GPU monitor 获取多卡平均值
            return {
                'total_power': sum(self.gpu_monitor.get_power_readings()),
                'avg_utilization': np.mean(self.gpu_monitor.get_utilization_readings()),
                'avg_memory_util': np.mean(self.gpu_monitor.get_memory_util_readings()),
            }
        return {'total_power': 0, 'avg_utilization': 0, 'avg_memory_util': 0}
    
    def get_cpu_state(self) -> dict:
        """获取 CPU 状态"""
        if hasattr(self, 'cpu_monitor') and self.cpu_monitor:
            return {
                'power': self.cpu_monitor.get_current_power(),
                'utilization': self.cpu_monitor.get_current_utilization(),
            }
        return {'power': 0, 'utilization': 0}
    
    def get_request_throughput(self) -> float:
        """获取请求吞吐率 (requests/second)"""
        # 从最近的 benchmark 数据计算
        if hasattr(self, 'recent_request_count'):
            return self.recent_request_count / max(self.window_size, 1)
        return 0.0
    
    def get_token_throughput(self) -> float:
        """获取 Token 吞吐率 (tokens/second)"""
        # 从最近的 benchmark 数据计算
        if hasattr(self, 'recent_token_count'):
            return self.recent_token_count / max(self.window_size, 1)
        return 0.0
```

### 4.2 动作空间定义

基于 vLLM 源码分析和 GreenDLS 的设计，动作空间设计如下：

#### vLLM 参数的运行时可调性分析

根据 vLLM 源码 (`vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/sched/scheduler.py`)：

| 参数 | 运行时可调整 | 说明 |
|------|-------------|------|
| **batch_size** | ✅ **可以** | 调度器动态决定每个 iteration 的批大小，受 `max_num_batched_tokens` 和 `max_num_seqs` 约束 |
| **kv_block_size** | ❌ **不可以** | 在 `KVCacheManager.__init__()` 初始化时固定，修改需要重启服务并重新分配 GPU 内存 |
| **gpu_frequency** | ✅ **可以** | 通过 NVML API 动态调整 GPU 频率 |

#### 推荐的动作空间设计

```python
class ActionSpace:
    """
    DQN 动作空间：运行时可动态调整的配置
    
    参考 GreenDLS 设计：
    - batch_sizes: 11 种选择
    - gpu_frequencies: 18 种选择
    - 总动作数: 11 × 18 = 198
    
    注意：kv_block_size 作为环境初始化参数，不在动作空间中
    """
    
    def __init__(self):
        # 批大小（参考 GreenDLS）
        self.batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36]
        
        # GPU 频率 (MHz)，根据你的 GPU 型号调整
        # 示例：RTX 3080Ti 支持的频率范围
        self.gpu_frequencies = []
        freq = 540
        while freq <= 1500:
            self.gpu_frequencies.append(freq)
            freq += 60  # 每 60 MHz 递增
        
        # 总动作数 = batch_sizes × gpu_frequencies
        self.n_actions = len(self.batch_sizes) * len(self.gpu_frequencies)
    
    def decode_action(self, action_idx: int) -> dict:
        """
        将动作索引解码为具体配置
        
        Args:
            action_idx: 动作索引 [0, n_actions)
            
        Returns:
            配置字典: {'batch_size': int, 'gpu_frequency': int}
        """
        batch_idx = action_idx // len(self.gpu_frequencies)
        freq_idx = action_idx % len(self.gpu_frequencies)
        
        return {
            'batch_size': self.batch_sizes[batch_idx],
            'gpu_frequency': self.gpu_frequencies[freq_idx],
        }
    
    def encode_action(self, batch_size: int, gpu_frequency: int) -> int:
        """
        将配置编码为动作索引
        """
        batch_idx = self.batch_sizes.index(batch_size)
        freq_idx = self.gpu_frequencies.index(gpu_frequency)
        return batch_idx * len(self.gpu_frequencies) + freq_idx
```

#### KV Block Size 的处理策略

由于 `kv_block_size` 不能在推理过程中动态调整，有以下几种处理方式：

**方案 1：固定配置（推荐用于生产环境）**
```python
# 在启动 vLLM 服务时固定选择
FIXED_KV_BLOCK_SIZE = 16  # 推荐值：16 或 32

# 启动命令
vllm serve /path/to/model --block-size 16
```

**方案 2：离线网格搜索（推荐用于研究）**
```python
# 对不同的 kv_block_sizes 分别训练 DQN
kv_block_sizes = [8, 16, 32, 64]
dqn_models = {}

for block_size in kv_block_sizes:
    # 重启 vLLM 服务
    restart_vllm_server(block_size=block_size)
    
    # 训练专门的 DQN
    dqn = train_dqn_for_block_size(block_size)
    dqn_models[block_size] = dqn

# 部署时根据工作负载选择最优的 (block_size, dqn_model) 组合
```

**方案 3：条件 DQN（高级研究）**
```python
# 将 kv_block_size 作为状态的一部分
def get_state_with_block_size(kv_cache_monitor, gpu_monitor, cpu_monitor, block_size):
    state_components = [
        # ... 其他状态 ...
        block_size / 64,  # 归一化的 block_size
    ]
    return np.array(state_components)

# 训练一个能适应不同 block_size 的通用 DQN
```

### 4.3 奖励函数设计

参考 GreenDLS 的奖励函数（源码：`GreenDLS/llm/src/scheduler.py:340-343`），结合 KV Cache 效率：

```python
def compute_reward(delay, energy, kv_cache_usage, slo_target, k=150):
    """
    奖励函数：平衡延迟、能耗和 KV 缓存效率
    
    参考 GreenDLS 设计：
    - 满足 SLO：奖励能效 (reward = k * delay / energy)
    - 违反 SLO：惩罚 (reward = slo_target - delay)
    
    扩展：加入 KV Cache 使用率的考虑
    
    Args:
        delay: 推理延迟 (ms)
        energy: 单个请求能耗 (mJ)
        kv_cache_usage: KV 缓存使用率 [0-1]
        slo_target: SLO 目标延迟 (ms)
        k: 能效权重系数 (默认 150，与 GreenDLS 一致)
    
    Returns:
        reward: 奖励值
    """
    if delay <= slo_target:
        # 满足 SLO：奖励能效
        # GreenDLS 原始公式: reward = k * delay / energy
        base_reward = k * delay / energy
        
        # KV Cache 效率奖励：
        # - 低使用率（<0.5）：奖励（内存利用不足）
        # - 高使用率（>0.9）：小幅惩罚（可能导致调度延迟）
        # - 中等使用率（0.5-0.9）：最优
        if kv_cache_usage < 0.5:
            kv_penalty = -5 * (0.5 - kv_cache_usage)  # 惩罚低利用率
        elif kv_cache_usage > 0.9:
            kv_penalty = -10 * (kv_cache_usage - 0.9)  # 惩罚过高利用率
        else:
            kv_penalty = 0  # 最优区间
        
        return base_reward + kv_penalty
    else:
        # 不满足 SLO：惩罚（与 GreenDLS 一致）
        return slo_target - delay


# 归一化版本（用于稳定 DQN 训练）
def compute_normalized_reward(delay, energy, kv_cache_usage, slo_target, k=150):
    """
    归一化奖励函数，输出范围约为 [-1, 1]
    """
    raw_reward = compute_reward(delay, energy, kv_cache_usage, slo_target, k)
    
    # 归一化：除以典型奖励的尺度
    # 假设典型能效奖励范围：0-100
    normalized_reward = raw_reward / 50.0
    
    # Clip 到合理范围
    return np.clip(normalized_reward, -2.0, 2.0)
```

---

## 5. 实时状态查询示例

```python
from util.kv_cache_monitor import KVCacheMonitor
import time

# 初始化监控器
monitor = KVCacheMonitor(
    base_url="http://localhost:8000",
    interval=0.05
)

# 启动后台监控
monitor.start()

# 实时查询状态（用于 DQN 决策）
while True:
    current_state = monitor.get_current_state()
    print(f"Current KV Cache State:")
    print(f"  Cache Usage: {current_state.get('kv_cache_usage', 0)*100:.1f}%")
    print(f"  Used Blocks Ratio: {current_state.get('kv_blocks_used_ratio', 0)*100:.1f}%")
    print(f"  Running Requests: {current_state.get('num_requests_running', 0)}")
    print(f"  Waiting Requests: {current_state.get('num_requests_waiting', 0)}")
    print()
    
    time.sleep(1)  # 每秒查询一次

# 停止监控
monitor.stop()
```

---

## 6. 完整的 DQN 训练流程示例

```python
import torch
import numpy as np
import time
import pynvml
from util.kv_cache_monitor import KVCacheMonitor
from GreenDLS.llm.src.rlmodel.DuelingDQN import DuelingDQN

# ========== 1. 初始化环境 ==========

# 固定 KV block size（在启动 vLLM 服务时指定）
FIXED_KV_BLOCK_SIZE = 16  # 推荐值

# 初始化 KV Cache 监控器
kv_monitor = KVCacheMonitor("http://localhost:8000", interval=0.05)
kv_monitor.start()

# 初始化 GPU 监控（NVML）
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# SLO 设置
SLO_TARGET_MS = 10000  # 10 秒，与 GreenDLS 一致

# ========== 2. 定义动作空间 ==========

class ActionSpace:
    def __init__(self):
        self.batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36]
        self.gpu_frequencies = list(range(540, 1560, 60))  # [540, 600, ..., 1500]
        self.n_actions = len(self.batch_sizes) * len(self.gpu_frequencies)
    
    def decode_action(self, action_idx):
        batch_idx = action_idx // len(self.gpu_frequencies)
        freq_idx = action_idx % len(self.gpu_frequencies)
        return {
            'batch_size': self.batch_sizes[batch_idx],
            'gpu_frequency': self.gpu_frequencies[freq_idx],
        }

action_space = ActionSpace()

# ========== 3. 状态空间构建 ==========

def build_state_vector(kv_monitor, gpu_monitor, cpu_monitor):
    """
    构建状态向量（优化版，基于实际监控数据）
    
    状态维度（10维）：
    0. GPU 总功率 (0-1000W)
    1. GPU 平均利用率 (0-100%)
    2. GPU 显存利用率 (0-100%)
    3. CPU 功率 (0-200W)
    4. CPU 利用率 (0-100%)
    5. 运行中的请求数 (0-50, 实测0-20)
    6. 等待中的请求数 (0-50)
    7. KV Cache 使用率 (0-1, 实测<10% due to prefix caching)
    8. 请求吞吐率 (0-20 req/s)
    9. Token 吞吐率 (0-2000 tok/s)
    
    Returns:
        np.ndarray: shape (10,) 的归一化状态向量
    """
    # 获取各监控器的当前状态
    kv_state = kv_monitor.get_current_state() if kv_monitor else {}
    gpu_state = gpu_monitor.get_current_state() if gpu_monitor else {}
    cpu_state = cpu_monitor.get_current_state() if cpu_monitor else {}
    
    state = [
        # GPU 状态 (3维)
        gpu_state.get('total_power', 0) / 1000,                           # [0-1]
        gpu_state.get('avg_utilization', 0) / 100,                        # [0-1]
        gpu_state.get('avg_memory_util', 0) / 100,                        # [0-1]
        
        # CPU 状态 (2维)
        cpu_state.get('power', 0) / 200,                                  # [0-1]
        cpu_state.get('utilization', 0) / 100,                            # [0-1]
        
        # 请求队列状态 (2维) - 最关键
        min(kv_state.get('num_requests_running', 0) / 50, 1.0),           # [0-1]
        min(kv_state.get('num_requests_waiting', 0) / 50, 1.0),           # [0-1]
        
        # KV Cache 状态 (1维)
        kv_state.get('cache_usage_perc', 0),                              # [0-1]
        
        # 吞吐量状态 (2维)
        min(kv_state.get('request_throughput', 0) / 20, 1.0),             # [0-1]
        min(kv_state.get('token_throughput', 0) / 2000, 1.0),             # [0-1]
    ]
    return np.array(state, dtype=np.float32)

# ========== 4. 初始化 DQN 模型 ==========

state_dim = 10  # 优化后的状态维度（从5维增加到10维）
action_dim = action_space.n_actions  # 11 × 17 = 187

dqn_agent = DuelingDQN(
    alpha=0.0001,        # 学习率
    state_dim=state_dim,
    action_dim=action_dim,
    fc1_dim=256,         # 隐藏层维度
    fc2_dim=256,
    device='cuda',
    gamma=0.99,          # 折扣因子
    epsilon=1.0,         # 初始探索率
    eps_end=0.1,         # 最小探索率
    eps_dec=1e-5         # 探索率衰减
)

# ========== 5. 训练循环 ==========

NUM_EPISODES = 500
STEPS_PER_EPISODE = 100

# 初始化观测
current_request_rate = 5.0  # 初始请求速率
observation = build_state_vector(
    kv_monitor.get_current_state(),
    gpu_power=100.0,
    gpu_util=0.5,
    request_rate=current_request_rate
)

for episode in range(NUM_EPISODES):
    total_reward = 0
    episode_start = time.time()
    
    print(f"\n========== Episode {episode + 1}/{NUM_EPISODES} ==========")
    
    for step in range(STEPS_PER_EPISODE):
        # 1. 选择动作
        action = dqn_agent.choose_action(observation, is_trained=(episode > 0))
        config = action_space.decode_action(action)
        
        batch_size = config['batch_size']
        gpu_freq = config['gpu_frequency']
        
        print(f"Step {step}: batch_size={batch_size}, gpu_freq={gpu_freq}")
        
        # 2. 执行动作：调整 GPU 频率
        pynvml.nvmlDeviceSetGpuLockedClocks(gpu_handle, 135, gpu_freq)
        
        # 3. 执行一批推理（这里简化，实际应等待 vLLM 处理）
        inference_start = time.time()
        
        # 模拟推理过程中收集指标
        power_samples = []
        for _ in range(10):  # 采样 10 次
            power_mw = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
            power_samples.append(power_mw / 1000.0)  # 转换为 W
            time.sleep(0.01)
        
        inference_time = time.time() - inference_start
        avg_power = np.mean(power_samples)
        energy = avg_power * inference_time  # J
        
        # 4. 获取下一状态
        kv_state = kv_monitor.get_current_state()
        next_observation = build_state_vector(
            kv_state,
            gpu_power=avg_power,
            gpu_util=0.8,
            request_rate=current_request_rate
        )
        
        # 5. 计算奖励
        delay_ms = inference_time * 1000 / batch_size  # 平均每个请求的延迟
        energy_per_request = energy / batch_size * 1000  # mJ
        
        reward = compute_reward(
            delay=delay_ms,
            energy=energy_per_request,
            kv_cache_usage=kv_state.get('kv_cache_usage', 0),
            slo_target=SLO_TARGET_MS,
            k=150
        )
        
        # 归一化奖励
        reward = reward / 50.0
        total_reward += reward
        
        # 6. 存储经验
        dqn_agent.remember(observation, action, reward, next_observation, done=0)
        
        # 7. 学习
        if episode > 0:  # 跳过第一个 episode（收集经验）
            loss = dqn_agent.learn()
            if step % 20 == 0:
                print(f"  Loss: {loss:.4f}, Reward: {reward:.4f}")
        
        # 8. 更新观测
        observation = next_observation
    
    # Episode 统计
    episode_time = time.time() - episode_start
    avg_reward = total_reward / STEPS_PER_EPISODE
    
    print(f"Episode {episode + 1} Summary:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Avg Reward: {avg_reward:.4f}")
    print(f"  Epsilon: {dqn_agent.epsilon:.4f}")
    print(f"  Time: {episode_time:.2f}s")
    
    # 保存模型
    if (episode + 1) % 50 == 0:
        torch.save(dqn_agent, f'dqn_model_episode_{episode + 1}.pth')
        print(f"  Model saved!")

# ========== 6. 清理 ==========
kv_monitor.stop()
pynvml.nvmlShutdown()
print("\nTraining completed!")
```

### 关键差异说明

与原 GreenDLS 设计相比的改进：

| 方面 | GreenDLS (LLM) | 本设计 (vLLM) |
|------|----------------|---------------|
| **Batch Size 调整** | 手动从队列取固定数量 | 通过 vLLM 调度器参数控制 |
| **KV Block Size** | 不涉及 | 启动时固定，不在动作空间 |
| **状态空间** | GPU 功率、等待时间、推理时间、请求速率 | 增加 KV Cache 使用率、块使用比例 |
| **能耗测量** | NVML 直接测量 | NVML 测量 + vLLM 内部统计 |
| **推理执行** | 直接调用 model.generate() | 通过 vLLM API 服务 |

---

## 7. 测试与验证

### 运行完整 Benchmark

```bash
bash /home/user/offload/FUEL/run_benchmark.sh
```

### 验证数据采集

检查 JSON 结果文件中的采样情况：

```python
import json

with open('benchmark_results/qwen3-8b-benchmark-xxx.json', 'r') as f:
    data = json.load(f)

# 检查采样点数量
gpu_samples = len(data['gpu_power_stats']['0']['power_trace'])
kv_samples = data['kv_cache_monitoring_stats']['samples_collected']

print(f"GPU samples: {gpu_samples} (at 0.05s interval)")
print(f"KV Cache samples: {kv_samples} (at 0.5s interval)")
print(f"Expected ratio: ~10:1 (GPU:KV)")
print(f"Actual ratio: {gpu_samples / max(kv_samples, 1):.1f}:1")

# 验证 KV Cache 数据有效性
if kv_samples > 0:
    avg_usage = data['kv_cache_monitoring_stats']['avg_stats']['avg_cache_usage_perc']
    print(f"✓ KV Cache monitoring successful! Avg usage: {avg_usage:.1f}%")
else:
    print("⚠ Warning: No KV Cache samples collected. Check server metrics endpoint.")
```

---

## 总结

### GreenDLS 设计回顾

**是的，GreenDLS 在设计时考虑了 batch_sizes 的动态调整**：

从源码分析 (`GreenDLS/llm/src/scheduler.py`, `GreenDLS/llm/src/glva.py`)：

1. **动作空间设计**：
   - Batch sizes: `[1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36]` (11 种)
   - GPU 频率: `[517, 577, 637, ..., 1537, 1597]` (18 种)
   - 总动作数: 11 × 18 = **198 种动作组合**

2. **调度策略**：
   - **RL 方法** (`predict_rl`): DQN 决策 `(batch_size, gpu_frequency)`
   - **Clipper 方法** (`clipper`): 根据 SLO 反馈动态调整 batch_size
   - **EAIS 方法** (`eais`): 使用预测模型决策最优配置

3. **状态空间**：
   - GPU 功率（归一化）
   - 等待时间（归一化）
   - 推理时间（归一化）
   - 请求速率（归一化）

4. **奖励函数**：
   ```python
   if delay <= SLO:
       reward = k * delay / energy  # k=150
   else:
       reward = SLO - delay
   ```

### 本设计的改进

基于 vLLM 特性的优化：

| 方面 | GreenDLS | 本设计 (vLLM + KV Cache) |
|------|----------|--------------------------|
| **Batch Size 调整** | ✅ 支持（手动队列管理） | ✅ 支持（vLLM 调度器参数） |
| **KV Block Size** | ❌ 不涉及 | ⚠️ 启动时固定，不在动作空间 |
| **状态空间** | 4 维（功率、等待、推理、速率） | 5 维（增加 KV 使用率、块比例） |
| **推理框架** | HuggingFace Transformers | vLLM (PagedAttention) |
| **内存管理** | PyTorch 默认 | PagedAttention + KV Cache 块管理 |
| **并发支持** | 手动批处理 | vLLM 内置调度器 |

### 最终配置建议

1. **KV Block Size 配置**：
   - 生产环境：固定 `--block-size 16`（默认值，平衡性能和内存）
   - 研究场景：离线测试 `[8, 16, 32, 64]`，选择最优配置

2. **监控同步**：
   - GPU、CPU、KV Cache 使用相同的 **0.05s 采样间隔**
   - 确保时间序列对齐

3. **DQN 动作空间**：
   - Batch sizes: `[1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36]`
   - GPU 频率: `[540, 600, ..., 1500]` (根据 GPU 型号调整)
   - 总动作数: **187-198** (取决于 GPU 频率范围)

4. **状态空间（优化版）**：
   - **10 维**（基于实际监控数据优化）：
     1. GPU 总功率 (0-1000W)
     2. GPU 平均利用率 (0-100%)
     3. GPU 显存利用率 (0-100%)
     4. CPU 功率 (0-200W)
     5. CPU 利用率 (0-100%)
     6. **运行中的请求数 (0-50)** ← 最关键指标，实测变化范围 0-20
     7. **等待中的请求数 (0-50)** ← 队列负载指标
     8. KV Cache 使用率 (0-1) ← 保留但信息量较小（prefix caching导致<10%）
     9. 请求吞吐率 (0-20 req/s)
     10. Token 吞吐率 (0-2000 tok/s)
   - 所有维度归一化到 `[0, 1]`
   
   **优化说明**：
   - ✅ **增加** `requests_running` 和 `requests_waiting` 权重（变化范围大，信息量高）
   - ✅ **增加** 吞吐量指标（直接反映性能）
   - ⚠️ **保留** `kv_cache_usage` 但降低权重（prefix caching 导致使用率很低）
   - ❌ **移除** `kv_blocks_used_ratio`（与 `kv_cache_usage` 高度相关，冗余）

5. **奖励函数**：
   - 基础：GreenDLS 的能效奖励 `k * delay / energy`
   - 扩展：KV Cache 使用率惩罚（避免过低/过高利用率）

### 实际应用流程

```bash
# 1. 启动 vLLM 服务（固定 block_size）
vllm serve /path/to/model --block-size 16 --port 8000

# 2. 运行 benchmark（启用 KV 监控）
# 3. 训练 DQN
python train_dqn.py --episodes 500 --steps-per-episode 100

# 4. 部署优化策略
python deploy_dqn.py --model dqn_model_episode_500.pth
```

---

## 7. 状态空间优化总结

### 7.1 优化依据（基于实际监控数据）

根据 `qwen3-8b-benchmark-20251104-171539.json` 的实测数据分析：

| 指标 | 实测范围 | 变化幅度 | 信息量 | 决策 |
|------|---------|---------|--------|------|
| `kv_cache_usage` | 1.5% - 9.6% | 小（<10%） | ⚠️ 低 | 保留但降低权重 |
| `kv_blocks_used` | 0 - 6 blocks | 极小 | ❌ 极低 | **移除** |
| `kv_blocks_free` | 6475 - 6481 | 极小 | ❌ 极低 | **移除** |
| `requests_running` | 0 - 20 | 大 | ✅ 高 | **核心指标** |
| `requests_waiting` | 0 - ? | 中等 | ✅ 中 | **保留并提权** |
| `prefix_cache_hit_rate` | ~71.7% | 中等 | ✅ 高 | **可选增加** |

**关键发现**：

1. **Prefix Caching 效果显著**：
   ```
   总查询: 93,940 tokens
   缓存命中: 67,344 tokens
   命中率: 71.7%
   ```
   导致 KV Cache 实际使用率极低（<10%），大部分时间只用 0-6 个 blocks。

2. **请求队列是最关键指标**：
   - `requests_running` 变化范围 0-20，直接反映系统负载
   - `requests_waiting` 反映队列积压情况
   - 这两个指标的变化比 KV Cache 使用率更能反映系统状态

3. **吞吐量指标很重要**：
   - `request_throughput`: 4.9 req/s
   - `token_throughput`: 1833 tok/s
   - 直接反映性能，有助于 DQN 学习能效权衡

### 7.2 对比：原设计 vs 优化设计

| 方面 | 原设计 (12维) | 优化设计 (10维) |
|------|--------------|----------------|
| **维度数** | 12 | 10（更简洁） |
| **KV Cache** | 5维（usage, blocks_ratio, running, waiting, total） | 1维（仅 usage） |
| **GPU** | 3维 | 3维（保持） |
| **CPU** | 2维 | 2维（保持） |
| **请求队列** | 包含在 KV 中 | **2维独立**（running, waiting） |
| **吞吐量** | 1维 | **2维**（req/s, tok/s） |
| **冗余度** | 高（KV指标高度相关） | 低（移除冗余） |
| **信息密度** | 中等 | 高 |

### 7.3 适用场景

#### 场景 A：启用 Prefix Caching（推荐）
- **使用优化后的 10维状态空间**
- KV Cache 使用率低（<10%），但保留作为特征
- 重点关注请求队列和吞吐量

#### 场景 B：禁用 Prefix Caching
如果禁用 Prefix Caching：
```bash
vllm serve /path/to/model --disable-prefix-caching
```

则 KV Cache 使用率会显著提高（预期 10%-70%），此时可以：
```python
# 恢复更多 KV Cache 维度
state_components.extend([
    kv_state.get('cache_usage_perc', 0),              # 使用率
    kv_state.get('used_blocks', 0) / total_blocks,    # 块使用比例
    kv_state.get('free_blocks', 0) / total_blocks,    # 剩余块比例
])
```

### 7.4 下一步建议

1. **实验验证**：
   - 在实际训练中对比 10维 vs 12维状态空间的收敛速度
   - 监控各维度的重要性（可通过 feature importance analysis）

2. **可选扩展**：
   - 添加 `prefix_cache_hit_rate` 作为第11维
   - 添加最近的 SLO 满足率作为第12维

3. **动态调整**：
   - 根据不同的负载模式（低负载 vs 高负载）调整归一化范围
   - 考虑使用自适应归一化（如 running mean/std）

这套系统可以用于训练强化学习模型，以优化大模型推理的能效比，同时兼顾 vLLM 的 KV Cache 管理特性。
