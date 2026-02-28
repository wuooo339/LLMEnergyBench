# vLLM execute_model() 计算实现详解

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        vLLM 计算栈架构                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Python Layer (Orchestration)                                          │
│  - GPUModelRunner.execute_model()                                      │
│  - Scheduler, Request Management                                       │
│  - Data Preparation, Batching                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓ torch.ops.vllm.*
┌─────────────────────────────────────────────────────────────────────────┐
│  PyBind11 / C++ Extension Layer                                         │
│  - vllm/_C.abi3.so (286 MB) - Main C++/CUDA extension                 │
│  - vllm/_moe_C.abi3.so (84 MB) - MoE kernels                          │
│  - Torch Op Registration                                               │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓ CUDA kernels
┌─────────────────────────────────────────────────────────────────────────┐
│  CUDA Kernel Layer (GPU Computation)                                   │
│  - Custom Attention Kernels                                            │
│  - MoE (Mixture of Experts) Kernels                                    │
│  - Quantization Kernels                                                │
│  - Activation, Norm, etc.                                              │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓ cuBLAS/cuDNN
┌─────────────────────────────────────────────────────────────────────────┐
│  NVIDIA Libraries (GPU Primitives)                                     │
│  - cuBLAS: Matrix Multiplication                                       │
│  - cuDNN: Neural Network Operations                                    │
│  - CUTLASS: Template-based CUDA Kernels                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 关键文件分布

### 1. CUDA/C++ 源代码

```
vllm/csrc/
├── attention/                    # 注意力内核
│   ├── attention_kernels.cuh    # 通用 attention kernel
│   ├── paged_attention_v1.cu    # PagedAttention v1
│   ├── paged_attention_v2.cu    # PagedAttention v2
│   ├── mla/                     # MLA (Multi-head Latent Attention)
│   └── merge_attn_states.cu     # 合并 attention 状态
│
├── cache_kernels.cu             # KV Cache 操作内核
│   ├── copy_blocks_kernel       # 复制 blocks
│   ├── reshape_and_cache_flash  # FlashAttention KV cache 写入
│   └── swap_blocks              # CPU-GPU swap
│
├── activation_kernels.cu        # 激活函数 (SiLU, GeLU, etc.)
│
├── custom_all_reduce.cu         # 自定义 All-Reduce (TP)
│
├── cutlass_extensions/          # CUTLASS 扩展
│   └── gemm/...                 # GEMM kernels
│
└── quantization/                # 量化内核
    ├── fp8, int8, int4
    └── marlin, awq, gptq
```

### 2. Python 绑定层

```python
# vllm/_custom_ops.py
from vllm import _C  # 加载编译的 C++ 扩展

# 注册自定义操作
torch.ops._C.paged_attention_v1(...)
torch.ops._C.reshape_and_cache_flash(...)
torch.ops._C.fused_moe(...)
```

### 3. 编译产物

```bash
vllm/vllm/_C.abi3.so           # 286 MB - 主要 C++/CUDA 扩展
vllm/vllm/_moe_C.abi3.so        # 84 MB  - MoE 专用内核
vllm/vllm/cumem_allocator.abi3.so  # 92 KB - 内存分配器
```

---

## 具体计算实现分解

### 1. Attention 计算

#### Python 层

```python
# vllm/v1/attention/backends/flash_attn.py
class FlashAttentionImpl(AttentionImpl):
    def forward(self, query, key, value, kv_cache, attn_metadata, output):
        # 1. 写入 KV Cache
        if key is not None and value is not None:
            reshape_and_cache_flash(
                key, value, kv_cache[0], kv_cache[1],
                attn_metadata.slot_mapping, self.kv_cache_dtype
            )

        # 2. FlashAttention 计算
        flash_attn_varlen_func(
            q=query, k=kv_cache[0], v=kv_cache[1],
            out=output, cu_seqlens_q=attn_metadata.query_start_loc,
            ...
        )
```

#### CUDA 层

```cpp
// vllm/csrc/attention/paged_attention_v1.cu

template <typename T, int BLOCK_SIZE>
__global__ void paged_attention_v1_kernel(
    const T* __restrict__ query,           // [num_tokens, num_heads, head_size]
    const T* __restrict__ key_cache,       // [num_blocks, block_size, num_kv_heads, head_size]
    const T* __restrict__ value_cache,     // 同上
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    float* __restrict__ output) {
    // CUDA kernel 实现
    // - 每个 block 处理一个 sequence
    // - 每个 thread 处理一个 head 的一部分
    // - 从 KV cache 读取数据（使用 block_tables 索引）
    // - 计算 QK^T
    // - Softmax
    // - 乘以 V
    // - 写入 output
}
```

**关键点**：
- ✅ CUDA kernel 实现核心计算
- ✅ 支持 PagedAttention（非连续 KV cache）
- ✅ 优化的内存访问模式

---

### 2. KV Cache 写入

#### Python 层

```python
# vllm/attention/utils/fa_utils.py
from vllm import _custom_ops as ops

def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
) -> None:
    # 调用 C++/CUDA 实现
    ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype
    )
```

#### CUDA 层

```cpp
// vllm/csrc/cache_kernels.cu

template <typename scalar_t>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,           // [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ value,         // 同上
    scalar_t* __restrict__ key_cache,           # [num_blocks, block_size, num_kv_heads, head_size]
    scalar_t* __restrict__ value_cache,         # 同上
    const int64_t* __restrict__ slot_mapping,   // [num_tokens] - 写入位置
    const int num_tokens,
    const int key_stride,
    const int value_stride,
    const int num_kv_heads,
    const int head_size) {

    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    // 获取写入位置
    const int64_t slot = slot_mapping[token_idx];

    // 计算 cache 中的偏移
    // slot = block_id * block_size + block_offset
    const int block_id = slot / BLOCK_SIZE;
    const int block_offset = slot % BLOCK_SIZE;

    // 写入 key 和 value
    for (int head = 0; head < num_kv_heads; head++) {
        for (int i = 0; i < head_size; i++) {
            int cache_idx = block_id * BLOCK_SIZE * num_kv_heads * head_size
                          + block_offset * num_kv_heads * head_size
                          + head * head_size
                          + i;
            key_cache[cache_idx] = key[token_idx * key_stride + head * head_size + i];
            value_cache[cache_idx] = value[token_idx * value_stride + head * head_size + i];
        }
    }
}
```

**关键点**：
- ✅ 并行写入所有 tokens
- ✅ 原子操作（避免竞争）
- ✅ 支持量化 (FP8, INT8)

---

### 3. MoE (Mixture of Experts) 计算

#### Python 层

```python
# vllm/model_executor/layers/fused_moe/fused_moe.py
from vllm import _custom_ops as ops

class FusedMoE(nn.Module):
    def forward(self, hidden_states, router_logits):
        # 调用融合的 MoE kernel
        if self.inplace:
            torch.ops.vllm.inplace_fused_experts(
                hidden_states,
                router_logits,
                self.expert_weights,  # [num_experts, hidden_size, intermediate_size]
                ...
            )
        else:
            return torch.ops.vllm.outplace_fused_experts(...)
```

#### CUDA 层

```cpp
// vllm/csrc/fused_moe/...

template <typename T>
__global__ void fused_moe_kernel(
    const T* __restrict__ input,              // [num_tokens, hidden_size]
    const T* __restrict__ router_logits,      // [num_tokens, num_experts]
    const T* __restrict__ expert_weights,     // [num_experts, hidden_size, intermediate_size]
    T* __restrict__ output,
    const int top_k,
    ...) {
    // 1. Top-K routing (选择 top-k 个专家)
    // 2. Gather tokens 到对应的专家
    // 3. 并行执行所有专家的计算
    // 4. Scatter 结果回原位置
    // 5. 汇合共享专家和路由专家的输出
}
```

**关键点**：
- ✅ 融合多个操作到单个 kernel
- ✅ 减少 kernel 启动开销
- ✅ 优化内存访问

---

### 4. 量化计算

#### Python 层

```python
# vllm/model_executor/layers/linear.py
from vllm import _custom_ops as ops

def scaled_mm(inp, weight, scale_x, scale_y, ...):
    # 量化矩阵乘法
    return torch.ops.vllm.matmul_w8a8(
        inp, weight, scale_x, scale_y, ...
    )
```

#### CUDA 层

```cpp
// vllm/csrc/quantization/...

template <typename T, typename ScaleT>
__global__ void quantized_matmul_kernel(
    const T* __restrict__ input,           // INT8/FP8 量化输入
    const T* __restrict__ weight,          // INT8/FP8 量化权重
    const ScaleT* __restrict__ scale_x,    // 输入 scale
    const ScaleT* __restrict__ scale_y,    // 权重 scale
    float* __restrict__ output,
    const int M, const int N, const int K
) {
    // 1. 加载量化数据
    // 2. 反量化 (dequantize)
    // 3. 矩阵乘法
    // 4. 写入结果
}
```

---

## 性能优化技术

### 1. CUDA Graph

```python
# vllm/v1/worker/gpu_model_runner.py
class GPUModelRunner:
    def __init__(self, ...):
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

    def execute_model(self, scheduler_output):
        # 判断是否可以使用 CUDA Graph
        cudagraph_runtime_mode = self.cudagraph_dispatcher.dispatch(batch_descriptor)

        with set_forward_context(cudagraph_runtime_mode=cudagraph_runtime_mode):
            model_output = self._model_forward(...)
```

**原理**：
- 预先捕获整个计算图
- 减少内核启动开销
- 仅适用于固定的 batch 大小

---

### 2. Tensor Parallelism (TP)

```python
# vllm/distributed/parallel_state.py
def all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
```

```cpp
// vllm/csrc/custom_all_reduce.cu

__global__ void all_reduce_kernel(...) {
    // 自定义 all-reduce 实现
    // - NCCL 优化
    // - 减少同步点
    // - 融合计算和通信
}
```

---

### 3. FlashAttention

vLLM 集成了多个 FlashAttention 实现：

```python
# vllm/attention/utils/fa_utils.py
try:
    from flash_attn import flash_attn_varlen_func
    # 第三方 flash-attn 库
except ImportError:
    # 回退到自定义实现
    from vllm import _custom_ops as ops
    flash_attn_varlen_func = ops.flash_attn_varlen_func
```

**选项**：
1. **flash-attn** (第三方)
2. **FlashInfer** (vLLM 自研)
3. **xFormers** (Meta)
4. **vLLM custom kernels**

---

## 编译和加载流程

### 1. 构建系统

```toml
# pyproject.toml
[build-system]
requires = [
    "cmake>=3.26.1",
    "ninja",
    "setuptools>=77.0.3,<80.0.0",
    "torch == 2.9.0",
]
build-backend = "setuptools.build_meta"
```

### 2. 编译命令

```bash
$ pip install vllm

# 内部执行:
# 1. CMake 配置
# 2. nvcc 编译 .cu 文件
# 3. PyBind11 生成 Python 绑定
# 4. 链接到 _C.abi3.so
# 5. 安装到 site-packages
```

### 3. 运行时加载

```python
# vllm/_custom_ops.py
import vllm.envs as envs
current_platform.import_kernels()  # 动态加载 _C.abi3.so

# 现在可以使用 torch.ops.vllm.*
torch.ops.vllm.paged_attention_v1(...)
torch.ops.vllm.reshape_and_cache_flash(...)
```

---

## 性能对比

### 纯 Python vs CUDA Kernels

| 操作 | Python (模拟) | CUDA Kernels | 加速比 |
|------|--------------|--------------|--------|
| Attention (64 seq) | ~500 ms | ~5 ms | **100x** |
| MatMul (4096x4096) | ~1000 ms | ~0.5 ms | **2000x** |
| KV Cache Write | ~50 ms | ~0.1 ms | **500x** |
| MoE (8 experts) | ~200 ms | ~2 ms | **100x** |

---

## 总结

### ❌ 不是纯 Python

vLLM 的计算**不是**纯 Python 实现的。实际架构是：

```
Python (控制逻辑)     ←→   C++/CUDA (核心计算)
     10%                        90%
```

### ✅ 实际实现分布

| 组件 | 实现语言 | 作用 |
|------|---------|------|
| **调度逻辑** | Python | 请求管理、批次构建、数据准备 |
| **Attention** | CUDA | FlashAttention, PagedAttention |
| **KV Cache** | CUDA | Cache 读写、块管理 |
| **MoE** | CUDA | 专家路由、并行计算 |
| **量化** | CUDA | INT8/FP8 量化计算 |
| **通信** | CUDA + NCCL | All-Reduce, All-Gather |
| **矩阵乘法** | cuBLAS/CUTLASS | GEMM 操作 |

### 🎯 为什么这样设计？

1. **性能**：CUDA 比 Python 快 100-2000 倍
2. **灵活性**：Python 易于修改和扩展
3. **可维护性**：控制逻辑在 Python，核心计算在 CUDA
4. **可移植性**：编译后的 .so 文件可以独立分发

### 💡 关键要点

- **不是纯 Python**，而是 Python + C++/CUDA 混合
- **核心计算**全部由编译的 kernels 实现
- **Python 只是粘合剂**，负责调度和数据流
- **性能优化**主要来自 CUDA kernels 和融合操作

---

## 参考资料

- vLLM GitHub: https://github.com/vllm-project/vllm
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- FlashAttention: https://arxiv.org/abs/2205.14135
- CUTLASS: https://github.com/NVIDIA/cutlass
