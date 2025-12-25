# vLLM 完整执行流程详解

## 目录

1. [初始化阶段](#初始化阶段)
2. [请求进入阶段](#请求进入阶段)
3. [调度阶段](#调度阶段)
4. [模型执行阶段](#模型执行阶段)
5. [输出处理阶段](#输出处理阶段)
6. [生成循环阶段](#生成循环阶段)
7. [完整时间线](#完整时间线)
8. [KV 管理详解](#kv-管理详解)

---

## 初始化阶段

### 1.1 启动 vLLM 服务器

```bash
# 命令行
vllm serve deepseek-v2-lite \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

### 1.2 初始化流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│  vLLM 服务器启动流程                                                     │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: API Server 进程启动
┌─────────────────────────────────────────────────────────────────────────┐
│  vllm/entrypoints/openai/api_server.py                                  │
│                                                                         │
│  app = FastAPI()                                                        │
│  llm_engine = AsyncLLM.from_engine_args(...)                            │
│                                                                         │
│  内部流程:                                                               │
│  1. 加载配置                                                             │
│  2. 初始化 tokenizer                                                    │
│  3. 启动 EngineCore 后台进程                                             │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓ multiprocessing
┌─────────────────────────────────────────────────────────────────────────┐
│  EngineCore 进程启动 (后台)                                              │
│  vllm/v1/engine/core.py                                                 │
│                                                                         │
│  engine_core = EngineCore(vllm_config, executor_class)                  │
│                                                                         │
│  初始化步骤:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ 1. ModelExecutor 初始化                                             ││
│  │    ├─ 加载模型权重到 GPU                                             ││
│  │    ├─ 初始化 KV Cache 张量                                           ││
│  │    └─ Profiling: 测试最大 blocks 数量                                ││
│  │                                                                     ││
│  │ 2. KVCacheManager 初始化                                            ││
│  │    ├─ block_pool = BlockPool(num_gpu_blocks)                       ││
│  │    ├─ coordinator = KVCacheCoordinator(...)                        ││
│  │    └─ enable_prefix_caching = True                                 ││
│  │                                                                    ││
│  │ 3. Scheduler 初始化                                                 ││
│  │    ├─ waiting_queue = []                                           ││
│  │    ├─ running_queue = []                                           ││
│  │    └─ block_manager = BlockManager(...)                            ││
│  │                                                                    ││
│  │ 4. ZMQ 通信启动                                                     ││
│  │    ├─ input_thread: 接收来自 API Server 的请求                      ││
│  │    └─ output_thread: 发送结果到 API Server                          ││
│  │                                                                    ││
│  │ 5. 启动主循环                                                       ││
│  │    └─ run_busy_loop()                                              ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 关键数据结构初始化

```python
# KV Cache 张量 (GPU 显存)
kv_cache = torch.zeros(
    (2,                    # K + V
     num_gpu_blocks,      # 例如: 1000 个 blocks
     block_size,          # 16 tokens/block
     num_kv_heads,        # 例如: 8 heads
     head_size),          # 例如: 128
    dtype=torch.float16,
    device='cuda'
)
# 形状: [2, 1000, 16, 8, 128]

# Block Pool (GPU 显存中的空闲块)
block_pool = BlockPool(
    free_blocks=[0, 1, 2, ..., 999],  # 初始全部空闲
    block_size=16
)

# Scheduler 队列
waiting_queue = []   # 等待的请求
running_queue = []   # 运行中的请求
```

---

## 请求进入阶段

### 2.1 run_benchmark.sh 发起请求

```bash
# run_benchmark.sh 内部
python benchmark_serving.py \
    --backend vllm \
    --model deepseek-v2-lite \
    --dataset-name sharegpt \
    --num-prompts 100

# benchmark_serving.py 发送 HTTP 请求
async def single_request(
    model: str,
    prompt: str,
    api_url: str,
    request_rate: float
):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.0,
                "stream": True
            }
        ) as response:
            async for chunk in response.content:
                # 处理流式输出
                ...
```

### 2.2 API Server 接收请求

```
┌─────────────────────────────────────────────────────────────────────────┐
│  API Server 进程                                                        │
└─────────────────────────────────────────────────────────────────────────┘

HTTP POST /v1/completions
    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  FastAPI handler: create_completion()                                   │
│  vllm/entrypoints/openai/serving_completion.py                          │
│                                                                         │
│  @app.post("/v1/completions")                                           │
│  async def create_completion(request: CompletionRequest):               │
│      # 1. 参数校验                                                       │
│      # 2. 渲染 prompt (如果有 chat template)                             │
│      engine_prompt = await renderer.render_prompt(...)                  │
│                                                                         │
│      # 3. 构建 SamplingParams                                           │
│      sampling_params = request.to_sampling_params(...)                  │
│                                                                         │
│      # 4. 调用 AsyncLLM.generate()                                      │
│      generator = self.engine_client.generate(                           │
│          prompt=engine_prompt,                                          │
│          sampling_params=sampling_params,                               │
│          request_id=request_id                                          │
│      )                                                                  │
│                                                                         │
│      # 5. 流式返回结果                                                   │
│      async for request_output in generator:                             │
│          yield format_to_openai_response(request_output)                │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  AsyncLLM.generate()                                                    │
│  vllm/v1/engine/async_llm.py                                            │
│                                                                         │
│  async def generate(self, prompt, sampling_params, request_id):         │
│      # Step 1: Tokenize prompt (⭐ Tokenizer 触发!)                     │
│      engine_req, prompt_text = await self._process_input(               │
│          prompt, sampling_params, request_id                            │
│      )                                                                  │
│      # 内部调用:                                                         │
│      # tokenizer.encode(prompt) → token_ids                             │
│      # 例如: "Hello world" → [9609, 11, 4395]                           │
│                                                                         │
│      # Step 2: 创建 Request 对象                                        │
│      request = EngineCoreRequest(                                      │
│          request_id=request_id,                                        │
          prompt_token_ids=token_ids,                                    │
│          sampling_params=sampling_params,                              │
│          ...                                                           │
│      )                                                                 │
│                                                                        │
│      # Step 3: 添加到 OutputProcessor                                   │
│      self.output_processor.add_request(                                │
│          engine_req, prompt_text, None, 0, queue                       │
│      )                                                                 │
│                                                                        │
│      # Step 4: 发送到 EngineCore (ZMQ)                                  │
│      await self.engine_core.add_request_async(request)                 │
│                                                                        │
│      # Step 5: 异步生成结果                                             │
│      async for output in queue:                                        │
│          yield output                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓ ZMQ send
┌─────────────────────────────────────────────────────────────────────────┐
│  EngineCore 进程 - input_thread (后台)                                   │
│                                                                         │
│  def process_input_sockets():                                           │
│      while True:                                                        │
│          # 1. 接收 ZMQ 消息                                              │
│          message = input_socket.recv()                                  │
│                                                                         │
│          # 2. 反序列化 (msgpack)                                         │
│          req_type, req_data = msgspec.msgpack.decode(message)           │
│                                                                         │
│          # 3. 放入 input_queue                                          │
│          input_queue.put((req_type, req_data))                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 调度阶段

### 3.1 主循环处理请求

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EngineCore 主循环                                                       │
│  vllm/v1/engine/core.py::run_busy_loop()                               │
└─────────────────────────────────────────────────────────────────────────┘

while True:
    # Step 1: 处理输入队列
    self._process_input_queue()
    # 从 input_queue.get() → _handle_client_request()
    # → scheduler.add_request(request)

    # Step 2: 执行一步调度 + 模型
    executed = self._process_engine_step()

    # Step 3: 发布统计
    self._maybe_publish_request_counts()
```

### 3.2 添加请求到 Scheduler

```python
# _handle_client_request() 内部
if request_type == EngineCoreRequestType.ADD_REQUEST:
    # ⭐ Scheduler 触发点 1: 添加请求
    self.scheduler.add_request(request)
    # 内部逻辑:
    # - 创建 Request 对象
    # - 计算 hash (用于 prefix caching)
    # - 放入 waiting_queue
```

### 3.3 调度决策

```python
# _process_engine_step() 内部
def _process_engine_step():
    # ⭐ 核心调度点!

    # 1. 检查是否有请求
    if not self.scheduler.has_requests():
        return False

    # 2. 调度 (⭐ Scheduler 触发点 2!)
    scheduler_output = self.scheduler.schedule()

    # 3. 执行模型 (⭐ ModelExecutor 触发!)
    model_output = self.model_executor.execute_model(scheduler_output)

    # 4. 更新状态 (⭐ Scheduler 触发点 3!)
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    # 5. 发送结果
    for req_id, output in engine_core_outputs.items():
        self.output_queue.put_nowait((req_id, output))

    return True
```

### 3.4 Scheduler.schedule() 详解

```python
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        """
        完整调度流程:

        输入状态:
        - waiting_queue: [req1, req2, req3, ...]
        - running_queue: [req4, req5, ...]
        - block_pool.free_blocks: [100, 101, 102, ...]

        输出:
        - SchedulerOutput(scheduled_requests, block_tables, attn_metadata)
        """

        # ========== Phase 1: 选择请求 ==========
        scheduled_from_waiting = self._select_requests_from_waiting()
        scheduled_from_running = self._continue_running_requests()
        scheduled_requests = scheduled_from_waiting + scheduled_from_running

        # ========== Phase 2: 分配 KV Cache (⭐ KVCacheManager!) ==========
        block_tables = {}
        for req in scheduled_requests:
            if req.status == RequestStatus.WAITING:
                # 新请求：分配初始 blocks
                num_tokens = req.num_tokens
                num_blocks = (num_tokens + block_size - 1) // block_size

                # ⭐ KVCacheManager.allocate_slots() 触发!
                allocated_blocks = self.kv_cache_manager.allocate_slots(
                    request=req,
                    num_new_tokens=num_tokens,
                    num_new_computed_tokens=0  # 新请求没有已计算的 tokens
                )

                if allocated_blocks is None:
                    # 内存不足！需要抢占 (Preemption)
                    self._preempt_some_requests()
                    # 再次尝试分配
                    allocated_blocks = self.kv_cache_manager.allocate_slots(...)

                # 更新 block_table
                req.block_table = [b.block_id for b in allocated_blocks.blocks[0]]

            elif req.status == RequestStatus.RUNNING:
                # 继续的请求：可能需要更多 blocks
                current_blocks = len(req.block_table)
                current_capacity = current_blocks * block_size

                if req.num_tokens + 1 > current_capacity:
                    # 需要分配新 block
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request=req,
                        num_new_tokens=1
                    )
                    req.block_table.extend([b.block_id for b in new_blocks.blocks[0]])

            block_tables[req.request_id] = req.block_table

        # ========== Phase 3: 构建 SchedulerOutput ==========
        scheduler_output = SchedulerOutput(
            scheduled_requests=scheduled_requests,
            scheduled_blocks=block_tables,
            req_id_to_index={req.request_id: i for i, req in enumerate(scheduled_requests)},
            ...
        )

        return scheduler_output
```

### 3.5 KVCacheManager.allocate_slots() 详解

```python
class KVCacheManager:
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0
    ) -> KVCacheBlocks | None:
        """
        分配 KV Cache blocks

        输入:
        - request: 请求对象
        - num_new_tokens: 新增的 tokens 数量

        输出:
        - KVCacheBlocks: 分配的 blocks
        - None: 无法分配（内存不足）
        """

        # ========== Step 1: 检查 Prefix Caching ==========
        if self.enable_caching:
            # 计算请求的 hash
            request_hash = compute_hash(request.prompt_token_ids)

            # 查找 cache
            computed_blocks, num_cached = self.coordinator.find_longest_cache_hit(
                request.block_hashes,
                max_cache_hit_length=request.num_tokens - 1
            )

            if num_cached > 0:
                # Cache 命中！复用已有 blocks
                # 减少 KV cache 写入
                pass

        # ========== Step 2: 计算需要的 blocks ==========
        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens,
            self.max_model_len
        )
        num_blocks_to_allocate = (
            num_tokens_need_slot + block_size - 1
        ) // block_size

        # ========== Step 3: 检查是否有足够空闲 blocks ==========
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None  # 内存不足

        # ========== Step 4: 分配 blocks ==========
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot
        )
        # 内部逻辑:
        # - 从 block_pool.pop_free_block() 获取空闲块
        # - 更新 block.ref_count += 1
        # - 更新 block.req_id = request.request_id

        # ========== Step 5: Cache blocks (Prefix Caching) ==========
        if self.enable_caching:
            num_tokens_to_cache = min(
                num_computed_tokens + num_new_tokens,
                request.num_tokens
            )
            self.coordinator.cache_blocks(request, num_tokens_to_cache)
            # 内部逻辑:
            # - 计算 block hashes
            # - 存入 prefix_cache: {hash: block_id}

        return KVCacheBlocks(new_blocks)
```

---

## 模型执行阶段

### 4.1 ModelExecutor.execute_model()

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GPUModelRunner.execute_model(scheduler_output)                        │
│  vllm/v1/worker/gpu_model_runner.py                                    │
└─────────────────────────────────────────────────────────────────────────┘

# ========== Step 1: 更新状态 ==========
self._update_states(scheduler_output)
# - 更新 input_batch
# - 更新 block_table
# - 更新 seq_lens

# ========== Step 2: 准备输入 ==========
attn_metadata, logits_indices, ... = self._prepare_inputs(scheduler_output)
# 内部逻辑:
# - 合并所有请求的 token_ids
# - 计算 positions
# - 构建 attn_metadata (包含 block_table)

# 示例:
# input_ids = [101, 102, 103, 201, 202, 203]  # 2 个请求
# positions = [0, 1, 2, 0, 1, 2]
# seq_lens = [3, 3]
# block_table = {
#     "req1": [0, 1],  # req1 使用 block 0, 1
#     "req2": [2, 3]   # req2 使用 block 2, 3
# }

# ========== Step 3: 预处理 ============
num_scheduled_tokens, input_ids, inputs_embeds, positions, ... = self._preprocess(
    scheduler_output, num_input_tokens, intermediate_tensors
)
# - 准备张量
# - 拷贝到 GPU (如果需要)

# ========== Step 4: 模型前向传播 (⭐ 核心计算!) ==========
with set_forward_context(attn_metadata, ...):
    model_output = self._model_forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
    )

    # 内部调用链:
    # self.model(...)
    #   ├─ Embedding: input_ids → hidden_states
    #   ├─ for layer in self.layers:  (60 层)
    #   │   ├─ Attention:
    #   │   │   ├─ QKV 投影
    #   │   │   ├─ RoPE
    #   │   │   ├─ ⭐ reshape_and_cache_flash: 写入 KV cache
    #   │   │   └─ ⭐ flash_attn_varlen_func: 读取 KV cache + 计算
    #   │   └─ MoE: 专家路由 + 计算
    #   └─ LayerNorm

# ========== Step 5: 计算 logits ==========
hidden_states = model_output
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
# logits.shape = [num_tokens, vocab_size]

# ========== Step 6: 采样 (⭐ Sampler 触发!) ==========
sampler_output = self._sample(logits, spec_decode_metadata)
# 内部调用:
# - sampler(logits) → sampled_token_ids
# - 例如: [104, 204] (2 个请求各生成 1 个 token)

# ========== Step 7: 后处理 ============
(num_nans_in_logits, logprobs_lists, valid_sampled_token_ids,
 prompt_logprobs_dict, req_ids_output_copy, req_id_to_index_output_copy,
 invalid_req_indices) = self._bookkeeping_sync(
    scheduler_output, sampler_output, logits, hidden_states,
    num_scheduled_tokens, spec_decode_metadata
)

# ========== Step 8: 返回结果 ==========
output = ModelRunnerOutput(
    req_ids=req_ids_output_copy,           # ["req1", "req2"]
    req_id_to_index=req_id_to_index_output_copy,  # {"req1": 0, "req2": 1}
    sampled_token_ids=valid_sampled_token_ids,  # [104, 204]
    logprobs=logprobs_lists,
    ...
)
return output
```

### 4.2 单层 Transformer 计算

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DeepseekV2DecoderLayer.forward(positions, hidden_states, residual)   │
│  vllm/model_executor/models/deepseek_v2.py                              │
└─────────────────────────────────────────────────────────────────────────┘

# ========== Attention Layer ==========
hidden_states = self.self_attn(
    positions=positions,
    hidden_states=hidden_states
)

# DeepseekV2MLAAttention.forward():
#   ├─ 1. 计算 Q
#   │   q = self.q_proj(hidden_states)  # [num_tokens, num_heads, head_size]
#   │
#   ├─ 2. 计算 KV (低秩)
#   │   latent_cache = self.kv_a_proj_with_mqa(hidden_states)
#   │   kv = self.kv_b_proj(latent_cache)
#   │   k_nope, v = kv.split(...)
#   │
#   ├─ 3. RoPE 旋转位置编码
#   │   q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
#   │
#   ├─ 4. ⭐ 写入 KV Cache (CUDA kernel!)
#   │   reshape_and_cache_flash(
#   │       key=k, value=v,
#   │       key_cache=kv_cache[0], value_cache=kv_cache[1],
#   │       slot_mapping=attn_metadata.slot_mapping,  # 写入位置
#   │       kv_cache_dtype=self.kv_cache_dtype
#   │   )
#   │   # 这是一个 CUDA kernel，原子操作地写入 KV cache
#   │   # slot_mapping[i] = block_id * block_size + offset
#   │
#   ├─ 5. ⭐ FlashAttention 计算 (CUDA kernel!)
#   │   flash_attn_varlen_func(
#   │       q=query,
#   │       k=kv_cache[0],  # 从 cache 读取!
#   │       v=kv_cache[1],
#   │       out=output,
#   │       cu_seqlens_q=attn_metadata.query_start_loc,
#   │       block_table=attn_metadata.block_table,  # ⭐ 关键!
#   │       max_seqlen_q=attn_metadata.max_query_len,
#   │       ...
#   │   )
#   │   # PagedAttention: 根据 block_table 读取不连续的 blocks
#   │
#   └─ 6. Output 投影
#       output, _ = self.o_proj(attn_output)

# ========== MoE Layer ==========
hidden_states = self.mlp(hidden_states)

# DeepseekV2MoE.forward():
#   ├─ 1. Gate (路由)
#   │   router_logits = self.gate(hidden_states)  # [num_tokens, n_experts]
#   │
#   ├─ 2. Shared Experts
#   │   shared_output = self.shared_experts(hidden_states)
#   │
#   ├─ 3. Routed Experts
#   │   routed_output = self.experts(
#   │       hidden_states=hidden_states,
#   │       router_logits=router_logits
#   │   )
#   │   # FusedMoE kernel: 并行执行多个专家
#   │
#   └─ 4. 合并
#       final_output = shared_output + routed_output

# ========== Residual Connection ==========
residual = hidden_states

return hidden_states, residual
```

---

## 输出处理阶段

### 5.1 Scheduler.update_from_output()

```python
def update_from_output(self, scheduler_output, model_output):
    """
    更新请求状态

    输入:
    - scheduler_output: 调度输出
    - model_output: ModelRunnerOutput
    """

    engine_core_outputs = {}

    # ========== Step 1: 更新每个请求的状态 ==========
    for req_id, sampled_token_id in zip(
        model_output.req_ids,
        model_output.sampled_token_ids
    ):
        request = self.requests[req_id]

        # 1. 添加新生成的 token
        request.add_completed_token(
            token_id=sampled_token_id.item(),
            logprobs=model_output.logprobs[...]
        )
        # 内部逻辑:
        # request.outputs[0].token_ids.append(sampled_token_id)
        # request.num_tokens += 1

        # 2. ⭐ Detokenizer 触发!
        new_text = request.detokenizer.update(
            new_token_ids=[sampled_token_id.item()],
            stop_terminated=False
        )
        # 内部逻辑:
        # - decode_next(sampled_token_id)
        # - 累积 output_text

        # 3. 检查是否完成
        if request.is_finished():
            # ⭐ 释放 KV Cache!
            self.kv_cache_manager.free(request)
            # 内部逻辑:
            # - 释放 request.block_table 中的所有 blocks
            # - block_pool.free(block_id) for each block

            # 移除请求
            self.finish_request(req_id)
        else:
            # 继续生成
            # 可能需要分配新的 KV Cache block
            current_capacity = len(request.block_table) * self.block_size
            if request.num_tokens + 1 > current_capacity:
                # ⭐ 分配新 block!
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request=request,
                    num_new_tokens=1
                )
                request.block_table.extend([b.block_id for b in new_blocks.blocks[0]])

        # 4. 构建输出
        engine_core_outputs[req_id] = EngineCoreOutputs(
            request_id=req_id,
            outputs=request.outputs,
            ...
        )

    return engine_core_outputs
```

### 5.2 Detokenizer.update() 详解

```python
class FastIncrementalDetokenizer:
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """
        增量解码新的 token

        输入:
        - new_token_ids: [104] (单个 token ID)

        输出:
        - 匹配的 stop string，或 None
        """

        # ========== Step 1: 解码新 token ==========
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
            # decode_next 内部:
            # new_token = self.stream.step(self.tokenizer, new_token_id)
            # return new_token or ""

        # ========== Step 2: 检查 stop strings ==========
        stop_string = None
        if self.stop and len(self.output_token_ids) > self.min_tokens:
            stop = check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string
```

---

## 生成循环阶段

### 6.1 单个请求的完整生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│  请求生命周期: 从进入到完成                                              │
└─────────────────────────────────────────────────────────────────────────┘

请求: "What is the capital of France?"
max_tokens: 100

=== Iteration 1: Prefill Phase ===

┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1: Tokenize prompt                                               │
│  Tokenizer: "What is the capital of France?" → [1045, 11, 318, ...]   │
│  prompt_tokens = [1045, 11, 318, 428, 11, 318, 338, 11, 504, 30]     │
│  num_tokens = 10                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 2: Scheduler.add_request()                                       │
│  waiting_queue.append(request)                                          │
│  request.num_computed_tokens = 0                                       │
│  request.block_table = []                                              │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 3: Scheduler.schedule() - 第 1 次调度                             │
│                                                                         │
│  选择请求: [request]                                                    │
│                                                                         │
│  ⭐ KVCacheManager.allocate_slots()                                    │
│    需要的 blocks: (10 + 16 - 1) // 16 = 1                             │
│    检查 Prefix Cache: 无命中                                            │
│    分配 block: [100]                                                    │
│    request.block_table = [100]                                          │
│    block_pool.free_blocks: [0, 1, 2, ..., 99, 101, 102, ...]            │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 4: ModelExecutor.execute_model() - Prefill                       │
│                                                                         │
│  准备输入:                                                              │
│    input_ids = [1045, 11, 318, 428, 11, 318, 338, 11, 504, 30]        │
│    positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]                           │
│    attn_metadata.block_table = [100]                                   │
│    attn_metadata.slot_mapping = [0, 1, 2, ..., 9]  # 写入位置           │
│                                                                         │
│  模型计算 (Prefill):                                                    │
│    ┌─────────────────────────────────────────────────────────────────┐   │
│    │ Layer 0:                                                        │   │
│    │   1. QKV 投影                                                    │   │
│    │   2. ⭐ reshape_and_cache_flash: 写入 KV cache                  │   │
│    │      key_cache[100, 0:10] = K                                   │   │
│    │      value_cache[100, 0:10] = V                                 │   │
│    │   3. ⭐ flash_attn_varlen_func                                  │   │
│    │      读取 KV cache[100, 0:10]                                   │   │
│    │      计算 attention                                             │   │
│    │   4. MoE                                                        │   │
│    └─────────────────────────────────────────────────────────────────┘   │
│    ... (重复 60 层)                                                     │
│                                                                         │
│  采样:                                                                  │
│    logits = lm_head(hidden_states[-1])  # 最后一个 token 的 logits     │
│    sampled_token = sampler(logits)  # 例如: 11 (",")                   │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 5: Scheduler.update_from_output()                                │
│                                                                         │
│  更新状态:                                                              │
│    request.token_ids.append(11)  # [1045, 11, ..., 30, 11]            │
│    request.num_tokens = 11                                              │
│    request.num_computed_tokens = 10  # Prefill 的 tokens                │
│                                                                         │
│  ⭐ Detokenizer.update():                                              │
│    decode_next(11) → ","                                                │
│    request.output_text = ","                                            │
│                                                                         │
│  检查是否需要新 block:                                                 │
│    当前容量: 1 * 16 = 16 tokens                                         │
│    已使用: 11 tokens                                                     │
│    不需要新 block                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 6: 返回结果                                                        │
│  EngineCoreOutputs(                                                    │
│      request_id="req-1",                                                │
│      outputs=[Output(text=",", token_ids=[11])],                        │
│      ...                                                                 │
│  )                                                                      │
└─────────────────────────────────────────────────────────────────────────┘

=== Iteration 2-N: Decode Phase ===

┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1: Scheduler.schedule() - 第 2 次调度                             │
│                                                                         │
│  选择请求: [request] (继续)                                             │
│                                                                         │
│  ⭐ KVCacheManager.allocate_slots()                                    │
│    需要的 tokens: 1 (只生成 1 个新 token)                              │
│    num_tokens_need_slot = 11 + 1 = 12                                  │
│    num_blocks = (12 + 16 - 1) // 16 = 1                                │
│    已有 blocks: 1                                                       │
│    不需要新 block!                                                      │
│                                                                         │
│  构建 SchedulerOutput:                                                 │
│    scheduled_requests = [request]                                       │
│    block_table = [100]  # 保持不变                                      │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 2: ModelExecutor.execute_model() - Decode                         │
│                                                                         │
│  准备输入:                                                              │
│    input_ids = [11]  # 只有新生成的 token!                            │
│    positions = [10]  # 新 token 的位置                                  │
│    attn_metadata.block_table = [100]                                   │
│    attn_metadata.slot_mapping = [10]  # 写入位置 (第 11 个 slot)       │
│    attn_metadata.seq_lens = [11]  # 总序列长度                         │
│                                                                         │
│  模型计算 (Decode):                                                     │
│    ┌─────────────────────────────────────────────────────────────────┐   │
│    │ Layer 0:                                                        │   │
│    │   1. QKV 投影 (只针对新 token!)                                   │   │
│    │      q = q_proj([11])  # [1, num_heads, head_size]               │   │
│    │      k, v = kv_proj([11])                                        │   │
│    │                                                                   │   │
│    │   2. ⭐ reshape_and_cache_flash: 写入新 token 的 KV            │   │
│    │      key_cache[100, 10] = K  # 写入第 10 个位置                 │   │
│    │      value_cache[100, 10] = V                                    │   │
│    │                                                                   │   │
│    │   3. ⭐ flash_attn_varlen_func                                  │   │
│    │      q = [1, num_heads, head_size]  # 只有 1 个新 token        │   │
│    │      k = key_cache[100, 0:11]  # 读取全部 11 个历史 tokens       │   │
│    │      v = value_cache[100, 0:11]                                 │   │
│    │      # 计算 attention:                                          │   │
│    │      # scores = q @ k.T  # [1, num_heads, 11]                    │   │
│    │      # output = scores @ v  # [1, num_heads, head_size]          │   │
│    │                                                                   │   │
│    │   4. MoE (只处理新 token)                                         │   │
│    └─────────────────────────────────────────────────────────────────┘   │
│    ... (重复 60 层)                                                     │
│                                                                         │
│  采样:                                                                  │
│    logits = lm_head(hidden_states[-1])                                  │
│    sampled_token = sampler(logits)  # 例如: 318 (" The")               │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 3: Scheduler.update_from_output()                                │
│                                                                         │
│  更新状态:                                                              │
│    request.token_ids.append(318)  # [..., 11, 318]                    │
│    request.num_tokens = 12                                              │
│    request.num_computed_tokens = 11                                     │
│                                                                         │
│  ⭐ Detokenizer.update():                                              │
│    decode_next(318) → " The"                                           │
│    request.output_text = ", The"                                        │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
... (重复 decode，直到生成 100 个 tokens 或遇到 EOS)

=== Iteration N: Completion ===

┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1: Scheduler.schedule()                                           │
│                                                                         │
│  选择请求: [request]                                                   │
│  已有 tokens: 100                                                       │
│  block_table = [100, 101, 102, 103, 104, 105, 106]  # 7 个 blocks      │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 2: ModelExecutor.execute_model()                                  │
│                                                                         │
│  采样:                                                                  │
│    sampled_token = 2  # EOS token                                      │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 3: Scheduler.update_from_output()                                │
│                                                                         │
│  更新状态:                                                              │
│    request.token_ids.append(2)                                         │
│    request.num_tokens = 101                                             │
│                                                                         │
│  检查是否完成:                                                          │
│    request.is_finished() = True  # 因为遇到 EOS token                   │
│                                                                         │
│  ⭐ KVCacheManager.free()                                              │
│    释放所有 blocks:                                                     │
│    for block_id in request.block_table:                                │
│        block_pool.free(block_id)                                       │
│    # block_pool.free_blocks: [100, 101, 102, 103, 104, 105, 106, ...]│
│                                                                         │
│  移除请求:                                                              │
│    running_queue.remove(request)                                        │
│    del self.requests[request.request_id]                                │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 4: 返回最终结果                                                    │
│  EngineCoreOutputs(                                                    │
│      request_id="req-1",                                                │
│      outputs=[Output(                                                  │
│          text=", The capital of France is Paris.",                      │
│          token_ids=[1045, 11, ..., 2],                                 │
│          finish_reason="stop"                                           │
│      )],                                                                 │
│      finished=True                                                      │
│  )                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 多请求并发

```
┌─────────────────────────────────────────────────────────────────────────┐
│  多请求并发生成                                                          │
└─────────────────────────────────────────────────────────────────────────┘

假设同时有 3 个请求:
- Req1: "What is AI?" (已完成 5 个 tokens, 需要 max_tokens=50)
- Req2: "Hello world" (已完成 2 个 tokens, 需要 max_tokens=30)
- Req3: "Explain quantum" (新请求, 15 个 tokens, 需要 max_tokens=100)

=== Scheduler.schedule() ===

选择请求:
┌─────────────────────────────────────────────────────────────────────────┐
│  策略: FCFS (First Come First Served) + 优先级                         │
│                                                                         │
│  1. waiting_queue: [Req3]  # 新请求                                     │
│  2. running_queue: [Req1, Req2]  # 继续生成                           │
│                                                                         │
│  scheduled_requests = [Req1, Req2, Req3]                                │
└─────────────────────────────────────────────────────────────────────────┘

KV Cache 分配:
┌─────────────────────────────────────────────────────────────────────────┐
│  Req1: 已有 blocks [0, 1] (32 tokens)                                  │
│       需要 33 tokens → 3 blocks                                          │
│       ⭐ 分配新 block: [2]                                              │
│       block_table = [0, 1, 2]                                           │
│                                                                         │
│  Req2: 已有 blocks [3] (16 tokens)                                     │
│       需要 18 tokens → 2 blocks                                          │
│       ⭐ 分配新 block: [4]                                              │
│       block_table = [3, 4]                                              │
│                                                                         │
│  Req3: 新请求, 15 tokens → 1 block                                      │
│       ⭐ 分配 block: [5]                                                │
│       block_table = [5]                                                  │
│                                                                         │
│  总计: 6 blocks                                                         │
│  block_pool.free_blocks: [6, 7, 8, ..., 999]                           │
└─────────────────────────────────────────────────────────────────────────┘

=== ModelExecutor.execute_model() ===

准备输入 (Batching):
┌─────────────────────────────────────────────────────────────────────────┐
│  合并所有 tokens:                                                       │
│    Req1: [11]  (1 个新 token)                                         │
│    Req2: [318]  (1 个新 token)                                        │
│    Req3: [1045, 11, 318, ..., 504]  (15 个 tokens, prefill)          │
│                                                                         │
│  input_ids = [11, 318, 1045, 11, 318, ..., 504]  # [17]               │
│  positions = [5, 2, 0, 1, 2, ..., 14]  # 每个请求独立计数              │
│  seq_lens = [6, 3, 15]  # 每个 request 的总长度                         │
│                                                                         │
│  attn_metadata:                                                         │
│    query_start_loc = [0, 1, 2, 17]  # 累积起始位置                     │
│    block_table = [0,1,2, 3,4, 5]  # 拼平的 block table                │
│    slot_mapping = [32, 16, 0, 1, 2, ..., 14]  # 写入位置                │
└─────────────────────────────────────────────────────────────────────────┘

模型计算 (GPU 并行):
┌─────────────────────────────────────────────────────────────────────────┐
│  for layer in self.layers:  # 60 层                                   │
│      # Attention (PagedAttention)                                      │
│      flash_attn_varlen_func(                                            │
│          q=[...],  # [17, num_heads, head_size]                        │
│          k=key_cache,                                                     │
│          v=value_cache,                                                  │
│          block_table=[0,1,2, 3,4, 5],  # 6 个 blocks                   │
│          cu_seqlens_q=[0, 1, 2, 17],  # [0,1], [1], [2,17]            │
│          seqused_k=[6, 3, 15],  # 每个请求的序列长度                   │
│          ...                                                              │
│      )                                                                   │
│      # MoE (并行执行 8 个 experts)                                       │
│      fused_moe(...)                                                      │
│                                                                         │
│  logits = lm_head(hidden_states)  # [17, vocab_size]                  │
│                                                                         │
│  # 采样 (每个 request 独立)                                             │
│  sampled_tokens = [                                                     │
│      sampler(logits[0:1]),    # Req1: 11                               │
│      sampler(logits[1:2]),    # Req2: 318                              │
│      sampler(logits[2:17])    # Req3: 1045 (Prefill 最后一个)          │
│  ]  # [11, 318, 1045]                                                      │
└─────────────────────────────────────────────────────────────────────────┘

输出:
┌─────────────────────────────────────────────────────────────────────────┐
│  ModelRunnerOutput(                                                     │
│      req_ids=["req1", "req2", "req3"],                                 │
│      sampled_token_ids=[11, 318, 1045]                                 │
│  )                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 完整时间线

### 7.1 从 run_benchmark.sh 到最终输出的时间线

```
┌─────────────────────────────────────────────────────────────────────────┐
│  完整时间线 (单请求完整流程)                                             │
└─────────────────────────────────────────────────────────────────────────┘

T0: run_benchmark.sh 启动
    ↓
    python benchmark_serving.py --backend vllm ...

T1: HTTP 请求发送
    POST /v1/completions
    {"model": "...", "prompt": "Hello world", "stream": True}
    ↓
    [网络传输 ~1ms]

T2: API Server 接收 (vllm/entrypoints/openai/api_server.py)
    create_completion(request)
    ↓
    [参数校验 ~0.1ms]

T3: AsyncLLM.generate() (vllm/v1/engine/async_llm.py)
    _process_input(prompt, ...)
    ↓
    Tokenizer.encode(prompt) → token_ids
    "Hello world" → [9609, 11, 4395]
    ↓ [~1ms]

T4: 创建 Request 对象
    request = EngineCoreRequest(
        request_id="req-1",
        prompt_token_ids=[9609, 11, 4395],
        ...
    )
    ↓ [~0.1ms]

T5: OutputProcessor.add_request()
    output_processor.add_request(request, ...)
    创建 Detokenizer:
        detokenizer = FastIncrementalDetokenizer(tokenizer, request)
    ↓ [~0.5ms]

T6: ZMQ 发送到 EngineCore (后台线程)
    engine_core.add_request_async(request)
    ↓ [IPC ~0.2ms]

T7: EngineCore input_thread 接收 (vllm/v1/engine/core.py)
    process_input_sockets()
    input_queue.put((ADD_REQUEST, request))
    ↓ [~0.1ms]

T8: 主循环处理输入 (run_busy_loop)
    _process_input_queue()
    input_queue.get() → _handle_client_request()
    ↓ [~0.1ms]

T9: Scheduler.add_request()
    waiting_queue.append(request)
    计算 hash: hash([9609, 11, 4395])
    ↓ [~0.2ms]

T10: 等待调度 (主循环继续)
    _process_engine_step()
    scheduler.schedule()
    ↓ [调度间隔 ~10-50ms]
    # 可能等待其他请求，或立即调度

T11: ⭐ Scheduler.schedule() - 第 1 次调度
    _select_requests_from_waiting() → [request]
    ↓

T12: ⭐ KVCacheManager.allocate_slots() - Prefill
    num_tokens = 3
    num_blocks = (3 + 16 - 1) // 16 = 1
    检查 Prefix Cache: 无命中
    block_pool.allocate() → block_id=100
    request.block_table = [100]
    ↓ [~0.1ms]

T13: 构建 SchedulerOutput
    scheduler_output = SchedulerOutput(
        scheduled_requests=[request],
        block_tables={"req-1": [100]},
        attn_metadata=...
    )
    ↓ [~0.1ms]

T14: ⭐ ModelExecutor.execute_model() - Prefill
    _prepare_inputs(scheduler_output)
    input_ids = [9609, 11, 4395]
    positions = [0, 1, 2]
    attn_metadata.slot_mapping = [0, 1, 2]  # 写入位置
    ↓ [~0.5ms]

T15: 模型前向传播 (60 层)
    _model_forward(input_ids, positions, kv_cache, attn_metadata)
    ↓
    for layer in self.layers:  # 重复 60 次
        # Attention
        qkv = qkv_proj(hidden_states)  # [3, num_heads, head_size]
        ↓
        reshape_and_cache_flash(
            k, v,
            key_cache[100, 0:3],  # 写入前 3 个位置
            slot_mapping=[0, 1, 2]
        )
        ↓ [CUDA kernel ~0.1ms]
        flash_attn_varlen_func(
            q, key_cache[100, 0:3], value_cache[100, 0:3],
            block_table=[100]
        )
        ↓ [CUDA kernel ~0.2ms]
        # MoE
        gate(hidden_states) → router_logits
        fused_moe(hidden_states, router_logits)
        ↓ [CUDA kernel ~0.5ms]
    ↓ [总时间 ~30-50ms for 60 layers]

T16: 计算 logits
    hidden_states → lm_head → logits
    logits = lm_head(hidden_states[-1])  # [vocab_size]
    ↓ [~0.1ms]

T17: ⭐ Sampler
    sampler(logits) → token_id
    例如: 11 (",")
    ↓ [~0.1ms]

T18: 后处理 (_bookkeeping_sync)
    计算 logprobs
    ↓ [~0.1ms]

T19: 返回 ModelRunnerOutput
    sampled_token_ids=[11]
    ↓ [~0.1ms]

T20: ⭐ Scheduler.update_from_output()
    request.add_completed_token(11)
    request.num_tokens = 4
    request.num_computed_tokens = 3
    ↓

T21: ⭐ Detokenizer.update()
    decode_next(11) → ","
    request.output_text = ","
    ↓ [~0.2ms]

T22: 检查是否需要新 block
    当前容量: 1 * 16 = 16
    已使用: 4
    不需要新 block
    ↓ [~0.1ms]

T23: 发送到 output_queue
    output_queue.put(("req-1", EngineCoreOutputs(...)))
    ↓ [~0.1ms]

T24: output_thread 发送 ZMQ (后台线程)
    process_output_sockets()
    output_socket.send(msgspec.msgpack.encode(output))
    ↓ [IPC ~0.2ms]

T25: AsyncLLM 接收 (API Server 进程)
    engine_core.get_output_async()
    output = await output_queue.get()
    ↓ [~0.1ms]

T26: OutputProcessor.process_output
    格式化为 OpenAI 响应
    {
        "id": "cmpl-123",
        "choices": [{"index": 0, "text": ","}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1}
    }
    ↓ [~0.1ms]

T27: SSE 流式返回
    yield f"data: {json.dumps(response)}\n\n"
    ↓ [网络传输 ~1ms]

T28: run_benchmark.sh 接收第一个 token
    async for chunk in response.content:
        data = json.loads(chunk)
        print(data["choices"][0]["text"])  # ","
    ↓

T29: ===== 循环回到 T10 (Decode Phase) =====

第 2 次迭代 (Decode):
    T30-T58: 重复 T10-T28，但只处理 1 个新 token
        input_ids = [11]
        positions = [3]
        slot_mapping = [3]  # 写入第 4 个位置
        ...生成下一个 token...

第 3 次迭代 (Decode):
    ...

... (重复 decode，直到 max_tokens 或 EOS)

第 N 次迭代 (完成):
    生成 EOS token (2)
    request.is_finished() = True

    ⭐ KVCacheManager.free()
    释放 block_table = [100, 101, ...]
    block_pool.free(block_id) for each block

    移除请求
    running_queue.remove(request)

T_final: 返回最终响应
    {
        "id": "cmpl-123",
        "choices": [{
            "index": 0,
            "text": ", world!",
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 50}
    }
```

---

## KV 管理详解

### 8.1 KV Cache 数据结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│  KV Cache 物理布局                                                       │
└─────────────────────────────────────────────────────────────────────────┘

GPU 显存中的 KV Cache 张量:
kv_cache.shape = [2, num_blocks, block_size, num_kv_heads, head_size]
                ↑   ↑           ↑            ↑              ↑
                K/V  物理块号    16 tokens    8 heads       128

示例:
num_blocks = 1000
block_size = 16
num_kv_heads = 8
head_size = 128

kv_cache 总大小 = 2 * 1000 * 16 * 8 * 128 * 2 (float16) = 65.5 MB

Block Pool (空闲块管理):
block_pool.free_blocks = [0, 1, 2, ..., 999]  # 初始全部空闲
block_pool.allocated_blocks = {}  # {block_id: ref_count}

Block Table (逻辑 → 物理映射):
request.block_table = [100, 101, 102, ...]
# 逻辑位置: [0-15]  → block 100
# 逻辑位置: [16-31] → block 101
# 逻辑位置: [32-47] → block 102
# ...
```

### 8.2 KV Cache 写入 (Prefill)

```python
# 在 Attention Layer 的 forward() 中

# 假设 prefill 3 个 tokens: [9609, 11, 4395]
input_ids = [9609, 11, 4395]
request.block_table = [100]  # 分配了 1 个 block
request.num_tokens = 3

# 计算 slot_mapping (写入位置)
slot_mapping = [
    100 * 16 + 0,  # block 100, offset 0
    100 * 16 + 1,  # block 100, offset 1
    100 * 16 + 2,  # block 100, offset 2
]  # [1600, 1601, 1602]

# 调用 reshape_and_cache_flash (CUDA kernel)
reshape_and_cache_flash(
    key=key,      # [3, num_kv_heads, head_size]
    value=value,  # [3, num_kv_heads, head_size]
    key_cache=kv_cache[0],      # [1000, 16, 8, 128]
    value_cache=kv_cache[1],    # [1000, 16, 8, 128]
    slot_mapping=slot_mapping,  # [1600, 1601, 1602]
    kv_cache_dtype=torch.float16
)

# CUDA kernel 伪代码:
for i in range(3):  # 并行写入 3 个 tokens
    slot = slot_mapping[i]
    block_id = slot // 16
    offset = slot % 16

    for head in range(8):  # 并行写入 8 个 heads
        for d in range(128):  # 并行写入 128 维
            key_cache[block_id, offset, head, d] = key[i, head, d]
            value_cache[block_id, offset, head, d] = value[i, head, d]

# 结果:
# kv_cache[0, 100, 0:3, :, :] 现在存储了前 3 个 tokens 的 K
# kv_cache[1, 100, 0:3, :, :] 现在存储了前 3 个 tokens 的 V
```

### 8.3 KV Cache 读取 (Decode)

```python
# 在 Attention Layer 的 forward() 中

# 假设 decode 第 4 个 token: [11]
input_ids = [11]
request.block_table = [100]  # 仍然是同一个 block
request.num_tokens = 4

# 计算 slot_mapping (只写入新 token)
slot_mapping = [100 * 16 + 3]  # [1603] (第 4 个位置)

# 调用 flash_attn_varlen_func (CUDA kernel)
flash_attn_varlen_func(
    q=query,                    # [1, num_heads, head_size]  # 只有新 token!
    k=kv_cache[0],              # [1000, 16, 8, 128]
    v=kv_cache[1],              # [1000, 16, 8, 128]
    cu_seqlens_q=[0, 1],        # [0, 1]  # query 的累积长度
    max_seqlen_q=1,             # query 的最大长度
    seqused_k=[4],              # [4]  # key 的实际长度
    max_seqlen_k=4,             # key 的最大长度
    block_table=request.block_table,  # [100]
    ...
)

# CUDA kernel 伪代码:
# 1. 读取 query (新 token)
q = query[0]  # [num_heads, head_size]

# 2. 读取 key cache (全部历史 tokens)
k_cache = kv_cache[0, 100, 0:4, :, :]  # [4, num_heads, head_size]
# [block 100, tokens 0-3, all heads, all dims]

# 3. 计算 attention
scores = q @ k_cache.T  # [num_heads, 4]
# [1, head_size] @ [head_size, 4] = [1, 4]

# 4. Softmax
attn_weights = softmax(scores, dim=-1)  # [1, 4]

# 5. 读取 value cache
v_cache = kv_cache[1, 100, 0:4, :, :]  # [4, num_heads, head_size]

# 6. 加权求和
output = attn_weights @ v_cache  # [1, num_heads, head_size]
# [1, 4] @ [4, head_size] = [1, head_size]

# 7. 写入新 token 的 KV (下一个 token 用)
reshape_and_cache_flash(
    key=new_k,
    value=new_v,
    key_cache=kv_cache[0],
    value_cache=kv_cache[1],
    slot_mapping=[1603],
    ...
)
```

### 8.4 Block 分配和释放

```python
# ========== 分配 Block ==========

class BlockPool:
    def allocate_block(self) -> int:
        """
        分配一个空闲 block

        返回: block_id
        """
        if not self.free_blocks:
            raise OutOfMemoryError("No free blocks")

        block_id = self.free_blocks.pop()
        self.blocks[block_id].ref_count = 1
        self.blocks[block_id].status = BlockStatus.ALLOCATED
        return block_id

# 示例:
# 初始: free_blocks = [0, 1, 2, ..., 999]
# 分配: block_id = block_pool.allocate_block() → 0
# 状态: free_blocks = [1, 2, ..., 999], blocks[0].ref_count = 1

# ========== 释放 Block ==========

    def free_block(self, block_id: int):
        """
        释放一个 block

        注意: 使用引用计数
        """
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            block.status = BlockStatus.FREE
            self.free_blocks.append(block_id)
            # 如果启用 Prefix Caching，保留 block 内容
            # 否则可以清空 block 内容

# 示例:
# 初始: blocks[0].ref_count = 1
# 释放: block_pool.free_block(0)
# 状态: blocks[0].ref_count = 0, blocks[0].status = FREE
#       free_blocks = [..., 0]
```

### 8.5 Prefix Caching 复用

```python
# ========== Prefix Cache 查找 ==========

class KVCacheCoordinator:
    def find_longest_cache_hit(
        self,
        block_hashes: list[int],  # request 的 hash 序列
        max_cache_hit_length: int
    ) -> tuple[list[KVCacheBlock], int]:
        """
        查找最长 cache 命中

        输入:
        block_hashes: [hash_0, hash_1, hash_2, ...]
        max_cache_hit_length: 最大匹配长度

        返回:
        (cached_blocks, num_cached_tokens)
        """

        # 假设 prefix_cache = {
        #     hash("Hello"): [100, 101],  # 2 blocks
        #     hash("Hello world"): [100, 101, 102],
        #     ...
        # }

        num_cached = 0
        cached_blocks = []

        for i in range(max_cache_hit_length):
            prefix_hash = compute_hash(block_hashes[:i+1])

            if prefix_hash in self.prefix_cache:
                # Cache 命中!
                cached_blocks.extend(self.prefix_cache[prefix_hash])
                num_cached += 1
            else:
                # 未命中，停止
                break

        return cached_blocks, num_cached

# 示例:
# Request 1: "Hello world"
# block_hashes = [hash("Hello"), hash(" world")]
# max_cache_hit_length = 1
# 查找: prefix_hash = hash("Hello") in prefix_cache → YES
# 返回: ([block_100], 1)  # 命中了 1 个 block (16 tokens)

# ========== Prefix Cache 存储 ==========

    def cache_blocks(self, request: Request, num_tokens: int):
        """
        存储 blocks 到 prefix cache

        输入:
        request: 请求对象
        num_tokens: 要缓存的 token 数量
        """

        # 计算每个 block 的 hash
        for i in range(0, num_tokens, self.block_size):
            block_start = i
            block_end = min(i + self.block_size, num_tokens)

            # 计算这个 block 的 hash
            block_hash = compute_hash(
                request.prompt_token_ids[block_start:block_end]
            )

            # 存入 cache
            block_ids = request.block_table[
                block_start // self.block_size :
                (block_end + self.block_size - 1) // self.block_size
            ]

            self.prefix_cache[block_hash] = block_ids

# 示例:
# Request 1: "Hello world" (10 tokens)
# num_tokens = 10
# block_size = 16

# 计算 hash:
# block_0_hash = hash([9609, 11, 4395, ...])  # 全部 10 个 tokens
# 存入:
# prefix_cache[block_0_hash] = [100]

# Request 2: "Hello world!" (11 tokens, 有相同前缀)
# 查找:
# block_0_hash = hash([9609, 11, 4395, ...])  # 前 10 个 tokens
# 命中! cached_blocks = [100]
# num_cached = 10
# 只需要计算第 11 个 token
```

---

## 总结

### 完整执行顺序

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Manager 触发顺序总结                                                    │
└─────────────────────────────────────────────────────────────────────────┘

初始化 (服务器启动):
├─ KVCacheManager.__init__()
├─ Scheduler.__init__()
├─ ModelExecutor.__init__()
└─ Detokenizer (每个请求创建时)

每个请求:
├─ Tokenizer (请求开始时，1 次)
└─ Detokenizer.__init__() (请求开始时，1 次)

每次调度 (~10-50ms):
├─ Scheduler.schedule()
│   ├─ KVCacheManager.allocate_slots()  ⭐
│   │   ├─ 检查 Prefix Caching
│   │   ├─ 分配新 blocks
│   │   └─ 更新 block_table
│   │
│   └─ 构建 SchedulerOutput
│
├─ ModelExecutor.execute_model()
│   ├─ _prepare_inputs()
│   ├─ _model_forward()
│   │   ├─ Embedding
│   │   ├─ for layer in self.layers (60 层)
│   │   │   ├─ Attention
│   │   │   │   ├─ QKV 投影
│   │   │   │   ├─ ⭐ reshape_and_cache_flash: 写入 KV
│   │   │   │   └─ ⭐ flash_attn_varlen_func: 读取 KV
│   │   │   └─ MoE
│   │   └─ LayerNorm
│   ├─ compute_logits()
│   └─ ⭐ _sample()
│
└─ Scheduler.update_from_output()
    ├─ Detokenizer.update()  ⭐
    ├─ 检查是否完成
    └─ KVCacheManager.free()  ⭐ (完成时)
```

### 每个 Token 的循环

```python
while not request.is_finished():
    # 1. 调度
    scheduler_output = scheduler.schedule()

    # 2. 分配 KV (如果需要)
    kv_cache_manager.allocate_slots(...)

    # 3. 执行模型
    model_output = model_executor.execute_model(scheduler_output)

    # 4. 采样
    new_token = sampler(logits)

    # 5. 解码
    new_text = detokenizer.update(new_token)

    # 6. 检查完成
    if is_finished:
        kv_cache_manager.free(request)
        break
```

### KV 管理核心

```
每个新 token:
├─ 检查是否需要新 block
│   if num_tokens > len(block_table) * block_size:
│       allocate_new_block()
│
├─ 写入 KV Cache
│   reshape_and_cache_flash(new_k, new_v, kv_cache, slot_mapping)
│
├─ 读取 KV Cache (全部历史)
│   flash_attn_varlen_func(q, kv_cache, block_table, seq_lens)
│
请求完成:
└─ 释放所有 blocks
    for block_id in block_table:
        block_pool.free(block_id)
```
