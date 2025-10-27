# vLLM Benchmark with Power Monitoring

基于 FUEL 框架的 vLLM 性能和功耗测试工具，支持实时 GPU/CPU 监控和详细的性能指标收集。

---
## 快速开始
### 环境准备
**安装 vLLM**
```bash
pip install vllm
```
### 运行测试

#### 第 1 步：启动 vLLM 服务器

```bash
vllm serve /share-data/wzk-1/model/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 128
```

**参数说明**:
- `--tensor-parallel-size 2`: 使用 2 个 GPU 进行张量并行
- `--gpu-memory-utilization 0.9`: GPU 显存利用率 90%
- `--max-model-len 8192`: 最大上下文长度
- `--port 8000`: 服务端口（benchmark 脚本会连接此端口）

#### 第 2 步：运行 Benchmark 测试

在**另一个终端**中运行：

```bash
cd /home/user/offload/FUEL
./run_benchmark.sh
```

测试脚本会：
1. 连接到 `http://localhost:8000` 的 vLLM 服务器
2. 使用 ShareGPT 数据集发送 100 个请求
3. 实时监控 GPU 功耗、利用率
4. 收集每个请求的性能指标（TTFT、TPOT、E2E 延迟等）
5. 将结果保存到 JSON 文件

### 配置参数

编辑 `run_benchmark.sh` 可修改以下参数：

```bash
# 服务器配置
HOST="localhost"
PORT=8000
MODEL="/share-data/wzk-1/model/Qwen3-8B"

# 测试参数
NUM_PROMPTS=100          # 请求数量
REQUEST_RATE=10          # 请求速率 (QPS)，设为 inf 表示无限制
OUTPUT_LEN=256           # 每个请求的最大输出长度

# 数据集
DATASET_NAME="sharegpt"  # 数据集类型: sharegpt 或 random
DATASET_PATH="/share-data/wzk-1/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

# GPU 监控
GPU_IDS="0,1"            # 要监控的 GPU ID（逗号分隔）
SAMPLE_INTERVAL=0.1      # GPU 采样间隔（秒）
WARMUP_RATIO=0.1         # 热身比例（0.1 表示前 10% 的请求为热身）
```

### 热身机制 (Warmup)

为了获得更准确的性能和功耗测试结果,测试脚本使用了**热身机制**:

#### 为什么需要热身？

1. **冷启动开销**: vLLM 服务器在处理最初几个请求时会有额外开销:
   - 模型权重首次加载到 GPU
   - KV Cache 预分配
   - CUDA 内核编译和优化
   - 系统缓存预热

2. **不稳定的性能**: 前几个请求的延迟和功耗通常会偏高,不代表稳定状态下的性能

#### 热身工作流程

```
总请求数: 100 个
热身比例: 0.1 (10%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
├─ 热身阶段 ─┤──────── 正式测试阶段 ────────────┤
 请求 1-10      请求 11-100
 (不计入统计)   (计入统计,启动监控)
```

**具体行为**:
- **热身阶段** (请求 1-10): 
  - 发送请求到服务器
  - **不**启动 GPU/CPU 监控
  - **不**计入性能统计
  
- **正式测试** (请求 11-100):
  - 在第 11 个请求开始时**启动监控器**
  - 所有性能指标从此开始计算
  - 在所有请求完成后**停止监控器**

#### 查看热身过程

运行测试时会看到如下输出:

```bash
Traffic request rate: 10.0
Sending 10 warmup requests...
[进度显示热身请求]
Warmup completed. Starting benchmark...  # <- 热身结束,开始正式测试
GPU monitoring started.                  # <- 监控器启动
CPU monitoring started.
[Client#001|Server:cmpl-xxxxx] Token #1 | 91.23ms | '你好'
...
Benchmark (nearly) completed. 
Monitoring stopped. Processing results...  # <- 监控器停止
```

#### 调整热身比例

在 `run_benchmark.sh` 中修改 `WARMUP_RATIO`:

```bash
WARMUP_RATIO=0.1    # 10% 热身 (推荐)
WARMUP_RATIO=0.2    # 20% 热身 (服务器负载高时)
WARMUP_RATIO=0.0    # 无热身 (测试冷启动性能)
```

**建议**:
- 生产环境测试: `0.1` (10%)
- 冷启动测试: `0.0` (无热身)
- 大规模测试: `0.05` (5%,避免浪费太多请求)

## 测试输出

### 实时输出

测试过程中会实时显示每个 token 的生成信息：

```
[Client#001|Server:cmpl-xxxxx] Token #  1 |   91.23ms | '你好'
[Client#001|Server:cmpl-xxxxx] Token #  2 |  132.45ms (+41.22ms) | '世界'
[Client#002|Server:cmpl-xxxxx] Token #  1 |   88.67ms | 'Hello'
...
```

**字段说明**:
- `Client#001`: 客户端请求 ID（用于跟踪）
- `Server:cmpl-xxxxx`: 服务器端请求 ID
- `Token #1`: 第几个生成的 token
- `91.23ms`: 从请求开始到此 token 的时间（第一个 token 即 TTFT）
- `+41.22ms`: Inter-token latency（与上一个 token 的间隔）

### 结果文件

测试完成后，结果保存在 `benchmark_results/` 目录下：

```
benchmark_results/
└── qwen3-8b-benchmark-20251027-194328.json
```

#### JSON 文件结构

```json
{
  "date": "20251027-194328",
  "backend": "vllm",
  "model_id": "/share-data/wzk-1/model/Qwen3-8B",
  "request_rate": 10,
  "num_prompts": 100,
  
  // 性能指标
  "duration": 18.71,                    // 总测试时间（秒）
  "completed": 100,                     // 成功完成的请求数
  "total_input_tokens": 24490,          // 总输入 tokens
  "total_output_tokens": 25390,         // 总输出 tokens
  "request_throughput": 5.35,           // 请求吞吐量（req/s）
  "output_throughput": 1357.12,         // token 吞吐量（tok/s）
  
  // 每个请求的详细数据（数组）
  "input_lens": [245, 250, ...],        // 每个请求的输入长度
  "output_lens": [256, 256, ...],       // 每个请求的输出长度
  "ttfts": [0.091, 0.088, ...],         // Time to First Token (秒)
  "itls": [[0.041, 0.040, ...], ...],   // Inter-Token Latency (秒)
  "e2els": [10.5, 10.7, ...],           // End-to-End Latency (秒)
  "prompt": ["...", "...", ...],        // 每个请求的 prompt
  "generated_texts": ["...", ...],      // 生成的文本
  
  // GPU 功耗统计（多 GPU 支持）
  "gpu_power_stats": {
    "0": {
      "avg_power": 162550,              // 平均功耗（毫瓦 mW）
      "avg_gpu_util": 85.2,             // 平均 GPU 利用率 (%)
      "avg_mem_util": 67.8,             // 平均显存利用率 (%)
      "power_stats": {
        "min_power": 145000,
        "power_5p": 150000,
        "power_25p": 155000,
        "median_power": 162000,
        "power_75p": 166000,
        "power_95p": 168820,            // 95% 百分位功耗（重要指标）
        "max_power": 172000,
        "power_std": 5234.5
      },
      "power_trace": [158570, 159000, ...],      // 完整功耗时间序列
      "memory_util_trace": [67.5, 67.8, ...]     // 显存利用率时间序列
    },
    "1": { ... }                        // GPU 1 的统计数据
  },
  
  // 能耗和碳排放统计
  "energy_stats": {
    "total_energy": 3045.2,             // 总能耗（焦耳）
    "energy_per_request": 30.45,        // 每请求能耗
    "energy_per_token": 0.12            // 每 token 能耗
  },
  
  "carbon_stats": {
    "operational_carbon": 0.437,        // 运营碳排放（gCO2eq）
    "embodied_carbon": 0.023,           // 隐含碳排放（gCO2eq）
    "total_carbon": 0.460               // 总碳排放（gCO2eq）
  }
}
```

### 关键指标说明

| 指标 | 说明 | 单位 | 重要性 |
|------|------|------|--------|
| **TTFT** (Time to First Token) | 首个 token 生成时间 | 秒 | ⭐⭐⭐ 用户体验 |
| **TPOT** (Time per Output Token) | 每个 token 生成时间 | 秒 | ⭐⭐⭐ 吞吐量 |
| **E2EL** (End-to-End Latency) | 完整请求延迟 | 秒 | ⭐⭐ 整体性能 |
| **power_95p** | 95% 百分位功耗 | mW | ⭐⭐⭐ 功耗评估 |
| **output_throughput** | Token 吞吐量 | tok/s | ⭐⭐⭐ 系统能力 |
| **avg_gpu_util** | 平均 GPU 利用率 | % | ⭐⭐ 资源利用 |

---

## 如何分析结果？

使用 Python 读取 JSON 文件：

```python
import json
import numpy as np

with open('benchmark_results/qwen3-8b-benchmark-20251027-194328.json', 'r') as f:
    data = json.load(f)

# 分析 TTFT
ttfts = np.array(data['ttfts'])
print(f"TTFT 平均值: {np.mean(ttfts):.3f}s")
print(f"TTFT P95: {np.percentile(ttfts, 95):.3f}s")

# 分析功耗
power_95p_gpu0 = data['gpu_power_stats']['0']['power_stats']['power_95p']
print(f"GPU 0 功耗 P95: {power_95p_gpu0 / 1000:.2f}W")
```

---

## 项目结构

```
├── benchmarks/
│   ├── benchmark_serving.py      # 主测试脚本（已修改）
│   ├── backend_request_func.py   # 请求函数（已修改，添加请求 ID 跟踪）
│   ├── util/
│   │   ├── monitor.py            # GPU 监控模块
│   │   └── cpu_monitor.py        # CPU 监控模块
│   └── benchmark_results/        # 测试结果目录
├── run_benchmark.sh              # 一键测试脚本
└── README.md                     # 本文档
```
---

## 基于 FUEL 框架

本项目基于 [FUEL (Functional Unit-based Evaluation for LLMs)](https://github.com/jojacola/FUEL)框架开发，FUEL 是一个用于评估 LLM 服务碳排放影响的创新框架。

### 主要改进

在 FUEL 原始框架基础上，本项目做了以下增强：

1. **简化的测试流程**: 提供一键运行脚本 `run_benchmark.sh`
2. **更好的可观测性**: 添加客户端请求 ID 跟踪和实时 token 生成日志
3. **修复的时间计算**: 解决了原始代码中的 `completion_time` 和时间同步问题
4. **完善的监控集成**: GPU/CPU 监控器正确启动和停止
5. **清晰的文档**: 面向实际使用的中文文档

### 引用

如果您在研究中使用了本工具，请引用 FUEL 原始论文：

```bibtex
@article{wu2025unveiling,
  title={Unveiling environmental impacts of large language model serving: A functional unit view},
  author={Wu, Yanran and Hua, Inez and Ding, Yi},
  journal={arXiv preprint arXiv:2502.11256},
  year={2025}
}
```

---

## License

本项目继承 FUEL 的 Apache-2.0 License。

## 贡献

欢迎提交 Issue 和 Pull Request！
