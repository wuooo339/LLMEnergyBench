# 输入压缩预想方案

### 方案对比

| 方案 | 压缩位置 | 实现难度 | 性能影响 | 质量损失 |
|------|---------|---------|---------|---------|
| **Token Pruning** | Tokenizer后 | ⭐ 简单 | 小 | 中等 |
| **KV Cache Eviction** | 推理中 | ⭐⭐⭐ 中等 | 中等 | 可控 |
| **Prompt Compression** | 输入前 | ⭐⭐ 简单 | 小 | 较大 |
| **Quantization** | 存储 | ⭐⭐⭐⭐ 复杂 | 小 | 小 |
| **Hybrid** | 多层级 | ⭐⭐⭐⭐⭐ 很复杂 | 可优化 | 可控 |

---

### 方案 1: 输入 Token 压缩 (基于重要性)
**核心思想**: 在进入模型前，根据重要性评分压缩 prompt
```python
class InputTokenCompressor:
    def __init__(self, model, tokenizer, compress_ratio=0.3):
        self.model = model
        self.tokenizer = tokenizer
        self.compress_ratio = compress_ratio

    def compress_prompt(self, prompt: str) -> str:
        """
        步骤：
        1. 计算 gradient-based importance
        2. 保留重要 token (system, 关键词)
        3. 移除冗余 token
        """

        # Step 1: Tokenize
        tokens = self.tokenizer.encode(prompt)

        # Step 2: 计算重要性分数
        importance_scores = self._compute_importance(tokens)

        # Step 3: 选择要保留的 tokens
        num_keep = int(len(tokens) * (1 - self.compress_ratio))
        keep_indices = torch.topk(importance_scores, num_keep).indices

        # Step 4: 保护重要 token (系统提示词、标点)
        keep_indices = self._protect_important_tokens(keep_indices, tokens)

        # Step 5: 重建 prompt
        compressed_tokens = tokens[keep_indices.sort().indices]
        return self.tokenizer.decode(compressed_tokens)

    def _compute_importance(self, tokens):
        """
        重要性特征：
        - TF-IDF 分数
        - 位置权重 (开头/结尾更重要)
        - 词性 (名词 > 冠词)
        - 特殊标记保护
        """
        scores = torch.zeros(len(tokens))

        # 位置权重
        for i, token in enumerate(tokens):
            position_weight = 1.0 - abs(i - len(tokens)/2) / (len(tokens)/2)
            scores[i] = position_weight

        return scores
```

**实现位置**: `vllm/v1/engine/processor.py` (在 tokenizer 之后)

---

### 方案 2: KV Cache 动态裁剪 (基于注意力分数)

**核心思想**: 在推理过程中，根据 attention 分数动态淘汰不重要的 KV

```python
class DynamicKVCachePruner:
    def __init__(self, prune_ratio=0.2, window_size=2048):
        self.prune_ratio = prune_ratio
        self.window_size = window_size
        self.attention_history = {}  # {req_id: [attention_scores]}

    def prune_kv_cache(
        self,
        req_id: str,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> Tuple[torch.Tensor, FlashAttentionMetadata]:
        """
        步骤：
        1. 获取历史 attention 分数
        2. 计算累积重要性
        3. 选择要保留的 blocks
        4. 更新 block_table
        """

        # Step 1: 记录当前 attention 分数
        current_attn = self._get_attention_scores(attn_metadata)
        if req_id not in self.attention_history:
            self.attention_history[req_id] = []
        self.attention_history[req_id].append(current_attn)

        # Step 2: 计算累积重要性 (指数移动平均)
        cumulative_scores = self._compute_cumulative_importance(req_id)

        # Step 3: 选择保留的 blocks
        num_total_blocks = attn_metadata.block_table.shape[1]
        num_keep = int(num_total_blocks * (1 - self.prune_ratio))

        # 保留策略：
        # - 最近 window_size 的 tokens (全保留)
        # - 历史重要 tokens (top-k)
        keep_indices = self._select_blocks_to_keep(
            cumulative_scores,
            num_keep,
            attn_metadata.seq_lens[0]
        )

        # Step 4: 更新 block_table
        new_block_table = attn_metadata.block_table[:, keep_indices]

        return kv_cache, attn_metadata

    def _compute_cumulative_importance(self, req_id: str):
        """计算累积重要性 (EMA)"""
        history = self.attention_history[req_id]
        cumulative = history[0]

        for scores in history[1:]:
            cumulative = 0.9 * cumulative + 0.1 * scores

        return cumulative
```

**实现位置**:
- `vllm/v1/attention/backends/flash_attn.py::forward()`
- `vllm/v1/worker/block_table.py` (block 管理逻辑)

---

### 方案 3: 分层缓存 (Hybrid Memory)

**核心思想**: 近期 tokens 用完整 KV，远期 tokens 用压缩版本

```python
class HierarchicalKVCache:
    def __init__(self, window_size=1024, compressed_ratio=4):
        self.window_size = window_size
        self.compressed_ratio = compressed_ratio

        # 三层缓存
        self.hot_cache = None    # 最近 window_size 个 tokens (完整精度)
        self.warm_cache = None   # 中间层 (2x 压缩)
        self.cold_cache = None   # 远期层 (4x 压缩)

    def compute_attention(
        self,
        query: torch.Tensor,
        req_id: str,
        seq_len: int,
    ):
        """
        注意力计算：
        1. hot_cache: 完整 FlashAttention
        2. warm_cache: 低精度或采样
        3. cold_cache: 进一步压缩
        """

        # 1. Hot cache (最近 tokens)
        if seq_len <= self.window_size:
            return self._full_attention(query, self.hot_cache)

        # 2. 分层计算
        hot_output = self._full_attention(query, self.hot_cache)
        warm_output = self._compressed_attention(query, self.warm_cache)
        cold_output = self._highly_compressed_attention(query, self.cold_cache)

        # 3. 加权融合
        alpha = self._compute_fusion_weights(seq_len)
        output = alpha * hot_output + (1-alpha) * warm_output

        return output
```

**实现位置**: 新建 `vllm/v1/attention/backends/hierarchical_attn.py`

---

### 方案 4: 量化 + 压缩组合

**核心思想**: 量化 KV cache + 淘汰不重要 block

```python
class QuantizedPrunedKVCache:
    def __init__(
        self,
        base_dtype=torch.float16,
        quant_dtype=torch.float8_e4m3fn,
        prune_ratio=0.2,
    ):
        self.base_dtype = base_dtype
        self.quant_dtype = quant_dtype
        self.prune_ratio = prune_ratio

        # 量化统计
        self.quant_scale = {}  # {layer_id: scale}
        self.quant_zero_point = {}  # {layer_id: zero_point}

    def store_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_id: int,
        attention_scores: torch.Tensor,
    ):
        """
        存储流程：
        1. 计算 attention-based importance
        2. Prune 不重要的 tokens
        3. 量化剩余的 KV
        4. 存入 cache
        """

        # Step 1: Pruning
        importance = attention_scores.mean(dim=-1)  # [seq_len]
        num_keep = int(len(importance) * (1 - self.prune_ratio))
        keep_mask = torch.topk(importance, num_keep).indices

        pruned_key = key[keep_mask]
        pruned_value = value[keep_mask]

        # Step 2: 量化
        quant_key = self._quantize(pruned_key, layer_id, is_key=True)
        quant_value = self._quantize(pruned_value, layer_id, is_key=False)

        # Step 3: 存储
        self.kv_cache[layer_id] = (quant_key, quant_value)

        return keep_mask  # 返回 mask 用于后续恢复

    def _quantize(self, tensor: torch.Tensor, layer_id: int, is_key: bool):
        """动态量化"""
        # Per-channel 量化
        if is_key:
            scale = tensor.abs().max(dim=-1, keepdim=True).values / 127
        else:
            scale = tensor.abs().max(dim=-1, keepdim=True).values / 127

        quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)

        # 保存量化参数
        key = f"{layer_id}_{'k' if is_key else 'v'}"
        self.quant_scale[key] = scale

        return quantized
```

**实现位置**:
- `vllm/v1/worker/gpu_model_runner.py` (store_kv 逻辑)
- `vllm/v1/attention/backends/flash_attn.py` (使用量化 cache)

---

### 实现路线建议

#### Phase 1: 快速验证 (1-2周)
1. **Token-level Pruning**
   - 在 `Processor` 中添加压缩逻辑
   - 简单的启发式规则 (TF-IDF, position)
   - 评估质量损失

2. **Benchmark**
   - 收集 baseline 指标
   - 测试压缩比 vs 质量

#### Phase 2: 工程优化 (2-4周)
1. **KV Cache Eviction**
   - 实现基于 attention 的淘汰
   - 修改 block_table 管理
   - 性能测试

2. **量化支持**
   - 添加 FP8 KV cache
   - 动态量化策略

#### Phase 3: 生产级方案 (4-8周)
1. **分层缓存**
   - 实现三层缓存
   - 自适应融合策略

2. **完整测试**
   - 多模型验证
   - 性能优化

---

### 关键挑战

1. **语义完整性**: 压缩可能导致语义丢失
2. **性能开销**: 计算 importance 本身有成本
3. **实现复杂度**: 需要修改多个模块
4. **评估指标**: 如何平衡显存 vs 质量

---

## 参考资源

### vLLM 核心文件
- `vllm/entrypoints/openai/api_server.py` - API 服务器
- `vllm/v1/engine/async_llm.py` - 异步引擎
- `vllm/v1/engine/core.py` - 核心引擎
- `vllm/v1/worker/gpu_model_runner.py` - 模型执行器
- `vllm/v1/attention/backends/flash_attn.py` - FlashAttention
- `vllm/v1/worker/block_table.py` - Block 管理

### 相关论文
- "PagedAttention: Efficient Attention via Virtual Memory"
- "FlashAttention: Fast and Memory-Efficient Exact Attention"
- "Compressive Transformers: Long-Range Sequence Modelling"
- "StreamingLLM: Efficient LLM with Attention Sink"

---

## 总结

vLLM 的推理流程是一个复杂的系统工程：

1. **请求处理链**: HTTP → API Server → AsyncLLM → EngineCore → ModelRunner → Attention
2. **并发机制**: 异步I/O + 多进程 + ZMQ通信
3. **内存管理**: PagedAttention + Block管理
4. **性能优化**: CUDA Graph + Batch scheduling

对于输入压缩，建议从简单的 Token Pruning 开始，逐步演进到动态 KV Cache 淘汰，最终实现分层缓存系统。
