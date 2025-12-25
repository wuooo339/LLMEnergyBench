# vLLM execute_model() 完整推理流程详解

## 完整调用链路

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    execute_model() 完整执行流程                          │
└─────────────────────────────────────────────────────────────────────────┘

GPUModelRunner.execute_model(scheduler_output)
│
├─ [Preprocess] 准备输入
│  └─ self._update_states(scheduler_output)
│  └─ self._prepare_inputs(scheduler_output)
│      └─ 构建 attn_metadata (FlashAttentionMetadata)
│      └─ 构建 block_table, slot_mapping
│  └─ self._preprocess()
│      └─ 准备 input_ids, positions, inputs_embeds
│
├─ [Forward] 模型前向传播
│  └─ self._model_forward(input_ids, positions, **model_kwargs)
│      │
│      └─ self.model(...)  ← DeepseekV2ForCausalLM.forward()
│          │
│          ├─ Embedding 层
│          │  input_ids → hidden_states
│          │
│          ├─ 循环执行所有 DecoderLayer (for layer in self.layers)
│          │  │
│          │  └─ DeepseekV2DecoderLayer.forward(positions, hidden_states, residual)
│          │      │
│          │      ├─ Attention 层
│          │      │  └─ DeepseekV2MLAAttention.forward(positions, hidden_states)
│          │      │      │
│          │      │      ├─ Q, K 投影
│          │      │      │  q_proj, kv_a_proj_with_mqa, q_b_proj, ...
│          │      │      │
│          │      │      ├─ RoPE 位置编码
│          │      │      │  rotary_emb(positions, q_pe, k_pe)
│          │      │      │
│          │      │      ├─ KV Cache 写入
│          │      │      │  reshape_and_cache_flash(k, v, kv_cache, slot_mapping)
│          │      │      │
│          │      │      ├─ FlashAttention 计算
│          │      │      │  flash_attn_varlen_func(q, k_cache, v_cache, ...)
│          │      │      │
│          │      │      └─ Output 投影
│          │      │         o_proj(attn_output)
│          │      │
│          │      └─ MoE 层 (DeepSeek V2 特有)
│          │          └─ DeepseekV2MoE.forward(hidden_states)
│          │              ├─ Gate (路由)
│          │              │  gate(hidden_states) → router_logits
│          │              │
│          │              ├─ Shared Experts (共享专家)
│          │              │  shared_experts(x)
│          │              │
│          │              ├─ Routed Experts (路由专家)
│          │              │  fused_moe(hidden_states, router_logits)
│          │              │
│          │              └─ 合并输出
│          │
│          └─ LayerNorm
│              norm(hidden_states, residual)
│
├─ [Postprocess] 后处理
│  ├─ self.model.compute_logits(hidden_states)
│  │  └─ self.lm_head(hidden_states) → logits [vocab_size]
│  │
│  ├─ self._sample(logits, spec_decode_metadata)
│  │  └─ sampler(logits) → sampled_token_ids
│  │
│  └─ self._bookkeeping_sync(...)
│      ├─ 计算 logprobs
│      ├─ 更新请求状态
│      └─ 准备返回数据
│
└─ [Return] 返回 ModelRunnerOutput
   sampled_token_ids, logprobs, ...
```

---

## 关键代码详解

### 1. GPUModelRunner.execute_model() - 主控制流程

**文件**: `vllm/v1/worker/gpu_model_runner.py`

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput:
    """执行模型推理的主函数"""

    # ============ Phase 1: 输入准备 ============
    with record_function_or_nullcontext("Preprocess"):
        # 1. 更新批次状态
        self._update_states(scheduler_output)

        # 2. 准备 Attention metadata (关键!)
        attn_metadata, logits_indices, ... = self._prepare_inputs(scheduler_output)
        # attn_metadata 包含:
        # - query_start_loc: 每个 query 的起始位置
        # - seq_lens: 每个序列的长度
        # - block_table: 映射逻辑位置到物理 KV Cache blocks
        # - slot_mapping: 当前 token 要写入 KV Cache 的物理位置

        # 3. 预处理输入
        (num_scheduled_tokens, input_ids, inputs_embeds,
         positions, intermediate_tensors, model_kwargs) = self._preprocess(
            scheduler_output, num_input_tokens, intermediate_tensors
        )

    # ============ Phase 2: 模型前向传播 ============
    # 使用 CUDA Graph 优化
    with set_forward_context(attn_metadata, ...):
        model_output = self._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        # 这里的 model_output 就是 hidden_states!

    # ============ Phase 3: Logits 计算 ============
    with record_function_or_nullcontext("Postprocess"):
        hidden_states = model_output

        # 计算 logits (只有在最后 PP rank 才需要)
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        # logits.shape = [num_tokens, vocab_size]

    # ============ Phase 4: 采样 ============
    with record_function_or_nullcontext("Sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)
        # sampler_output.sampled_token_ids = [token_id_1, token_id_2, ...]

    # ============ Phase 5: 簿记 (Bookkeeping) ============
    with record_function_or_nullcontext("Bookkeep"):
        (num_nans_in_logits, logprobs_lists, valid_sampled_token_ids,
         prompt_logprobs_dict, ...) = self._bookkeeping_sync(
            scheduler_output, sampler_output, logits, hidden_states,
            num_scheduled_tokens, spec_decode_metadata,
        )

    # ============ Phase 6: 返回结果 ============
    output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        logprobs=logprobs_lists,
    )
    return output
```

---

### 2. self.model.forward() - 模型层循环

**文件**: `vllm/model_executor/models/deepseek_v2.py`

```python
class DeepseekV2ForCausalLM(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """
        这里调用内部的 model (DeepseekV2Model)
        返回 hidden_states，而不是 logits!
        """
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states
```

```python
class DeepseekV2Model(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # Step 1: Embedding
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None

        # Step 2: 循环执行所有层
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        # Step 3: LayerNorm
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(...)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

---

### 3. DeepseekV2DecoderLayer - 单层计算

**文件**: `vllm/model_executor/models/deepseek_v2.py`

```python
class DeepseekV2DecoderLayer(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单个 Transformer Decoder Layer 的完整计算

        Args:
            positions: [num_tokens] - 位置信息
            hidden_states: [num_tokens, hidden_size] - 当前层输入
            residual: [num_tokens, hidden_size] - 残差连接

        Returns:
            (hidden_states, residual) - 输出和残差
        """

        # 1. 残差连接
        if residual is not None:
            hidden_states = hidden_states + residual

        # 2. Attention 层
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        # hidden_states: [num_tokens, hidden_size]

        # 3. 残差
        residual = hidden_states

        # 4. MoE 层 (DeepSeek V2 特有)
        hidden_states = self.mlp(hidden_states)

        # 5. 返回
        return hidden_states, residual
```

---

### 4. Attention Layer - 注意力计算

**文件**: `vllm/model_executor/models/deepseek_v2.py`

```python
class DeepseekV2MLAAttention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        DeepSeek V2 的 MLA (Multi-head Latent Attention)
        """

        # Step 1: 计算 Q
        q = self.q_proj(hidden_states)[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Step 2: 计算 KV (低秩压缩)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]

        # Step 3: RoPE 旋转位置编码
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        # Step 4: 重组 Q, K
        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe

        # Step 5: 注意力计算 (调用 Attention backend)
        attn_output = self.attn(q, k, v)

        # Step 6: Output 投影
        output, _ = self.o_proj(attn_output)
        return output
```

---

### 5. Attention Backend - FlashAttention + KV Cache

**文件**: `vllm/v1/attention/backends/flash_attn.py`

```python
class FlashAttentionImpl(AttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,   # [num_tokens, num_heads, head_size]
        key: torch.Tensor,     # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,   # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor, # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        FlashAttention 前向传播 + KV Cache 管理
        """

        num_actual_tokens = attn_metadata.num_actual_tokens

        # ============ Step 1: 写入 KV Cache ============
        if key is not None and value is not None:
            reshape_and_cache_flash(
                key,
                value,
                kv_cache[0],  # key_cache
                kv_cache[1],  # value_cache
                attn_metadata.slot_mapping,  # 写入位置
                self.kv_cache_dtype,
            )
        # reshape_and_cache_flash 是一个 CUDA kernel，原子操作地写入 KV cache

        # ============ Step 2: FlashAttention 计算 ============
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=kv_cache[0],  # key_cache - 从 cache 读取!
            v=kv_cache[1],  # value_cache - 从 cache 读取!
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,  # 累积序列长度
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,  # 每个序列的实际长度
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,  # 因果掩码
            block_table=attn_metadata.block_table,  # PagedAttention
        )
        # flash_attn_varlen_func 是优化的 FlashAttention kernel
        # 支持变长序列和 PagedAttention

        return output
```

---

### 6. MoE Layer - DeepSeek V2 特有

**文件**: `vllm/model_executor/models/deepseek_v2.py`

```python
class DeepseekV2MoE(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        DeepSeek V2 的 MoE (Mixture of Experts) 层
        """

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Step 1: 路由 (Gate)
        if not self.experts.is_internal_router:
            router_logits, _ = self.gate(hidden_states)
            # router_logits: [num_tokens, n_routed_experts]
            # 每个 token 对每个专家的分数

        # Step 2: Shared Experts (共享专家)
        # 所有 tokens 都经过共享专家
        fused_moe_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        shared_output, final_hidden_states = fused_moe_out

        # Step 3: Routed Experts (路由专家)
        # 根据 gate 的结果，token 被分配到不同的专家
        if self.shared_experts is not None:
            final_hidden_states += shared_output

        # Step 4: AllReduce (TP)
        if not self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)
```

---

## 数据流示例

假设有 2 个请求，每个生成 1 个 token：

### 输入数据

```python
scheduler_output = SchedulerOutput(
    scheduled_requests=[
        Request(id="req1", token_ids=[101, 102, 103, ...], ...),
        Request(id="req2", token_ids=[201, 202, 203, ...], ...),
    ],
    scheduled_blocks={
        "req1": [0, 1, 5, 3],  # block_table
        "req2": [2, 4, 6, 7],
    },
    ...
)

# 合并后的输入
input_ids = [101, 102, 103, 201, 202, 203]  # [6]
positions = [0, 1, 2, 0, 1, 2]  # [6]
seq_lens = [3, 3]  # [2]
```

### 执行流程

```python
# 1. Embedding
hidden_states = embedding(input_ids)
# hidden_states: [6, 5120]  (5120 = hidden_size)

# 2. Layer 0
hidden_states = layer_0(positions, hidden_states)
#   - Attention: flash_attn(q, k_cache, v_cache, ...)
#   - MoE: gate + experts
#   hidden_states: [6, 5120]

# 3. Layer 1
hidden_states = layer_1(positions, hidden_states)
# ... (重复 60 层)

# 4. Layer N-1
hidden_states = layer_59(positions, hidden_states)

# 5. LayerNorm
hidden_states = norm(hidden_states)
# hidden_states: [6, 5120]

# 6. Logits
logits = lm_head(hidden_states)
# logits: [6, 102400]  (102400 = vocab_size)

# 7. Sample
sampled_tokens = sampler(logits)
# sampled_tokens: [104, 204]

# 8. 返回
ModelRunnerOutput(
    req_ids=["req1", "req2"],
    sampled_token_ids=[104, 204],
    ...
)
```

---

## 时间线分析

```
T0: execute_model() 开始
T1: _prepare_inputs() - 构建 attn_metadata
T2: _preprocess() - 准备 input_ids, positions
T3: _model_forward() 开始
T4:    └─ Embedding: input_ids → hidden_states
T5:    └─ Layer 0:
T6:       ├─ Attention (QKV + FlashAttention + KV Cache 写入)
T7:       └─ MoE (Gate + Experts)
T8:    └─ Layer 1:
T9:       └─ ...
T10:   └─ Layer 59 (最后一层)
T11:   └─ LayerNorm
T12: compute_logits() - hidden_states → logits
T13: _sample() - logits → token_ids
T14: _bookkeeping_sync() - 准备返回数据
T15: 返回 ModelRunnerOutput
```

---

## 关键优化技术

### 1. PagedAttention (分页注意力)

```python
# 逻辑序列
tokens: [t0, t1, t2, t3, t4, t5, t6, t7, ...]
blocks: [B0, B0, B0, B0, B1, B1, B1, B1, ...]

# 物理存储
KV Cache:
┌─────────┬─────────┬─────────┬─────┐
│ Block 0 │ Block 1 │ Block 2 │ ... │
│ [16 tok]│ [16 tok]│ [16 tok]│     │
└─────────┴─────────┴─────────┴─────┘

# block_table 映射
block_table = [0, 0, 0, 0, 1, 1, 1, 1, ...]
```

### 2. CUDA Graph 优化

```python
# 预先捕获的 CUDA Graph
with set_forward_context(...):
    model_output = self._model_forward(...)

# 如果 batch 大小符合预先捕获的图，直接运行
# 避免内核启动开销
```

### 3. MLA (Multi-head Latent Attention)

```python
# DeepSeek V2 的 KV 压缩
# 标准 Attention: K, V ∈ [seq_len, num_heads, head_dim]
# MLA: KV 先通过低秩投影: latent ∈ [seq_len, kv_lora_rank]
#     然后 K, V 从 latent 恢复

latent_cache = kv_a_proj_with_mqa(hidden_states)  # 压缩
kv = kv_b_proj(latent_cache)  # 恢复
```

### 4. MoE 优化

```python
# 共享专家 + 路由专家
shared_experts: 所有 tokens 都经过 (减少计算)
routed_experts: 根据 gate 动态路由 (增加容量)
```

---

## 总结

`execute_model()` **确实完成了整个模型的推理和计算**，包括：

### ✅ 完整的流程

1. **输入准备**: 构建 input_ids, positions, attn_metadata, block_table
2. **模型前向传播**:
   - Embedding 层
   - 所有 Transformer Decoder Layer (60 层)
   - 每层包含: Attention + MoE
3. **Logits 计算**: lm_head(hidden_states) → logits
4. **采样**: sampler(logits) → token_ids
5. **后处理**: 计算 logprobs, 准备返回数据

### 🔧 它**不包含**的部分

- **请求调度**: 由 `Scheduler` 完成
- **KV Cache 分配**: 由 `CacheManager` 完成
- **结果返回**: 由 `OutputProcessor` 处理

### 🎯 设计理念

`execute_model()` 专注于**模型执行**，其他功能由其他组件负责，实现了良好的**关注点分离** (Separation of Concerns)。

这种设计使得：
- 模型执行逻辑可以独立优化
- 不同模型架构可以共享相同的执行框架
- 易于添加新的优化技术 (CUDA Graph, quantization, etc.)
