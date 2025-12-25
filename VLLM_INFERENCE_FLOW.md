# vLLM 推理流程完整分析

本文档详细说明 vLLM 如何从 `run_benchmark.sh` 发送的请求开始，经过服务器处理、模型推理，最终返回生成的 token。

---

## 目录

1. [整体架构概览](#整体架构概览)
2. [请求发起：run_benchmark.sh](#请求发起run_benchmarksh)
3. [API 服务器层](#api-服务器层)
4. [异步引擎层](#异步引擎层)
5. [核心引擎层](#核心引擎层)
6. [模型执行层](#模型执行层)
7. [注意力计算层](#注意力计算层)
8. [响应返回流程](#响应返回流程)
9. [输入压缩预想方案](#输入压缩预想方案)

---

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            vLLM 推理架构                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ run_benchmark.sh  │  ← 用户启动脚本
└──────┬───────┘
       │ HTTP POST /v1/completions
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         API Server Layer                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  vllm/entrypoints/openai/api_server.py                            │ │
│  │  - FastAPI 应用                                                    │ │
│  │  - OpenAI 兼容协议                                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                          ↓                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  vllm/entrypoints/openai/serving_completion.py                    │ │
│  │  - create_completion()                                            │ │
│  │  - 请求解析、参数校验                                              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      AsyncLLM Engine Layer                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  vllm/v1/engine/async_llm.py                                      │ │
│  │  - AsyncLLM.generate()                                            │ │
│  │  - 异步请求管理                                                    │ │
│  │  - 输出处理器                                                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
       │ ZMQ IPC
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       EngineCore Layer                                    │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  vllm/v1/engine/core.py                                            │ │
│  │  - Scheduler (请求调度)                                           │ │
│  │  - 输入队列 / 输出队列                                             │ │
│  │  - 核心事件循环                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       Model Executor Layer                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  vllm/v1/worker/gpu_model_runner.py                               │ │
│  │  - execute_model()                                                │ │
│  │  - 输入准备、批处理                                                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                          ↓                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Model Layers (Transformer/DeepSeek etc.)                         │ │
│  │  每层包含:                                                         │ │
│  │  - Attention Layer                                                │ │
│  │  - FFN Layer                                                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      Attention Layer                                      │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  vllm/v1/attention/backends/flash_attn.py                         │ │
│  │  - forward()                                                       │ │
│  │  1. 计算 Q, K, V                                                   │ │
│  │  2. reshape_and_cache_flash (写入 KV cache)                       │ │
│  │  3. flash_attn_varlen_func (注意力计算)                           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                          ↓                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  KV Cache (PagedAttention)                                        │ │
│  │  [2, num_blocks, block_size, num_kv_heads, head_size]             │ │
│  │       物理块     16 tokens    KV头维度    向量维度                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Sampling & Output                                      │
│  - 采样下一个 token                                                     │
│  - 返回给 EngineCore                                                    │
│  - 通过 AsyncLLM → API Server → run_benchmark.sh                       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 请求发起：run_benchmark.sh

### 文件位置
```
/home/user/offload/FUEL/run_benchmark.sh
```

### 关键代码

```bash
# 脚本配置
MODEL="/share-data/wzk-1/model/deepseek-v2-lite"
HOST="localhost"
PORT="8000"
BACKEND="vllm"

# 调用 benchmark_serving.py
python benchmark_serving.py \
    --backend $BACKEND \
    --model $MODEL \
    --host $HOST \
    --port $PORT \
    --dataset-name sharegpt \
    --dataset-path $DATASET_PATH \
    --num-prompts $NUM_PROMPTS \
    --request-rate $REQUEST_RATE
```

### 数据流

1. **脚本启动** → 读取配置参数
2. **调用 Python 脚本** → `benchmark_serving.py`
3. **构建 HTTP 请求** → 通过 `aiohttp` 发送 POST 请求

---

## API 服务器层

### 文件位置
```
vllm/entrypoints/openai/api_server.py        # FastAPI 应用入口
vllm/entrypoints/openai/serving_completion.py # Completions API 实现
```

### 关键代码

#### api_server.py
```python
app = FastAPI()

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """OpenAI 兼容的 completions 接口"""
    # 调用 serving_completion.py 的处理逻辑
    return await completion_serving.create_completion(request, raw_request)
```

#### serving_completion.py
```python
class OpenAIServingCompletion(OpenAIServing):
    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Request | None = None,
    ):
        # 1. 解析请求参数
        request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"

        # 2. 渲染 prompt
        engine_prompts = await renderer.render_prompt_and_embeds(
            prompt_or_prompts=request.prompt,
            prompt_embeds=request.prompt_embeds,
            config=self._build_render_config(request),
        )

        # 3. 构建 SamplingParams
        sampling_params = request.to_sampling_params(
            max_tokens,
            self.model_config.logits_processor_pattern,
            self.default_sampling_params,
        )

        # 4. 调用引擎生成 (关键步骤!)
        generator = self.engine_client.generate(
            prompt=engine_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        # 5. 流式返回结果
        async for request_output in generator:
            yield format_to_openai_response(request_output)
```

### 数据格式

**输入请求 (JSON):**
```json
{
  "model": "/share-data/wzk-1/model/deepseek-v2-lite",
  "prompt": "You are a helpful assistant.\n\nUser: Hello\n\nAssistant:",
  "max_tokens": 4096,
  "temperature": 0.0,
  "stream": true
}
```

**输出响应 (Streaming SSE):**
```
data: {"id":"cmpl-123","choices":[{"index":0,"text":"Hello"}],"usage":{"prompt_tokens":10,"completion_tokens":1}}
data: {"id":"cmpl-123","choices":[{"index":0,"text":"!"}],"usage":{...}}
data: [DONE]
```

---

## 异步引擎层

### 文件位置
```
vllm/v1/engine/async_llm.py           # AsyncLLM 主类
vllm/v1/engine/core_client.py         # EngineCoreClient (进程间通信)
```

### 关键代码

#### AsyncLLM.generate()
```python
class AsyncLLM(EngineClient):
    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        **kwargs
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        主函数：处理生成请求

        流程：
        1) 创建 AsyncStream 对应 Request
        2) 处理输入 (tokenizer)
        3) 添加 Request 到 Detokenizer
        4) 添加 Request 到 EngineCore (独立进程)
        """

        # Step 1: 处理输入
        engine_req, prompt_text = await self._process_input(
            prompt, sampling_params, request_id, **kwargs
        )

        # Step 2: 创建输出队列
        queue = RequestOutputCollector()

        # Step 3: 添加请求到输出处理器
        self.output_processor.add_request(
            engine_req, prompt_text, None, 0, queue
        )

        # Step 4: 添加请求到 EngineCore (通过 ZMQ)
        await self.engine_core.add_request_async(engine_req)

        # Step 5: 异步生成结果
        async for output in queue:
            yield output
```

### 进程间通信 (ZMQ)

```
┌─────────────────────┐         ZMQ         ┌─────────────────────┐
│   API Server Proc   │ <=================>│  EngineCore Proc    │
│   (AsyncLLM)        │   IPC (msgpack)    │  (后台进程)          │
└─────────────────────┘                    └─────────────────────┘
         ↑                                              ↓
    HTTP Response                              Model Execution
```

**通信协议 (msgpack):**
- 请求类型: `ADD_REQUEST`, `ABORT`, `PROFILE`
- 数据序列化: `MsgpackEncoder` / `MsgpackDecoder`

---

## 核心引擎层

### 文件位置
```
vllm/v1/engine/core.py                      # EngineCore 主类
vllm/v1/core/sched/scheduler.py            # Scheduler 调度器
```

### 关键代码

#### EngineCore 主循环
```python
class EngineCore:
    def __init__(self, vllm_config, executor_class, log_stats):
        # 1. 初始化 Model Executor
        self.model_executor = executor_class(vllm_config)

        # 2. 初始化 KV Cache
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(vllm_config)

        # 3. 初始化 Scheduler
        self.scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
        )

    def run_loop(self):
        """主事件循环"""
        while True:
            # Step 1: 从输入队列获取请求
            req_type, req_data = self.input_queue.get()

            if req_type == EngineCoreRequestType.ADD_REQUEST:
                self.scheduler.add_request(req_data)

            # Step 2: 调度 - 选择要执行的请求
            scheduler_output = self.scheduler.schedule()

            # Step 3: 执行模型
            model_output = self.model_executor.execute_model(
                scheduler_output=scheduler_output
            )

            # Step 4: 更新状态，输出结果
            self._process_model_output(model_output)

            # Step 5: 发送到输出队列
            self.output_queue.put(output_data)
```

#### Scheduler 调度逻辑
```python
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        """
        调度算法：
        1. 从等待队列选择请求 (基于优先级、到达时间)
        2. 分配 KV Cache blocks
        3. 构建批次 (batching)
        4. 返回 SchedulerOutput
        """

        # Phase 1: 选择请求
        scheduled_requests = self._select_requests()

        # Phase 2: 分配 blocks
        for req in scheduled_requests:
            self.block_manager.allocate_blocks(req)

        # Phase 3: 构建 SchedulerOutput
        return SchedulerOutput(
            scheduled_requests=scheduled_requests,
            scheduled_blocks=block_tables,
            ...
        )
```

### 数据结构

**SchedulerOutput:**
```python
@dataclass
class SchedulerOutput:
    # 调度的请求
    scheduled_requests: List[Request]

    # Block 表 (映射逻辑位置到物理 KV Cache)
    scheduled_blocks: Dict[str, torch.Tensor]  # {req_id: block_table}

    # 位置映射
    req_id_to_index: Dict[str, int]

    # 其他元数据
    num_lookahead_slots: int
    ...
```

---

## 模型执行层

### 文件位置
```
vllm/v1/worker/gpu_model_runner.py           # GPUModelRunner
vllm/v1/worker/gpu_input_batch.py           # 输入批次处理
```

### 关键代码

#### execute_model()
```python
class GPUModelRunner:
    def execute_model(
        self,
        scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """
        执行模型前向传播

        流程：
        1. 准备输入 (input_ids, positions, block_table)
        2. 调用 model.forward()
        3. 收集隐藏状态和 logits
        4. 采样下一个 token
        5. 返回 ModelRunnerOutput
        """

        # Step 1: 准备输入张量
        input_batch = self._prepare_input_batch(scheduler_output)
        # input_batch 包含:
        # - input_ids: [num_tokens]
        # - positions: [num_tokens]
        # - seq_lens: [num_reqs]
        # - block_table: [num_reqs, max_blocks_per_seq]

        # Step 2: 执行模型
        hidden_states = self.model(
            input_ids=input_batch.input_ids,
            positions=input_batch.positions,
            kv_cache=self.kv_cache,
            attn_metadata=input_batch.attn_metadata,
        )

        # Step 3: 计算 logits
        logits = self.model.compute_logits(hidden_states)

        # Step 4: 采样
        sampled_tokens = self.sampler(logits)

        # Step 5: 返回输出
        return ModelRunnerOutput(
            sampled_tokens=sampled_tokens,
            ...
        )
```

#### 输入准备细节
```python
def _prepare_input_batch(self, scheduler_output):
    """准备模型输入"""

    # 1. 合并所有请求的 token
    all_input_ids = []
    all_positions = []

    for req in scheduler_output.scheduled_requests:
        all_input_ids.extend(req.token_ids)
        all_positions.extend(req.positions)

    # 2. 转换为张量
    input_ids = torch.tensor(all_input_ids, dtype=torch.int32)
    positions = torch.tensor(all_positions, dtype=torch.int64)

    # 3. 构建 Attention Metadata
    attn_metadata = self.attn_backend.build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )

    return GPUInputBatch(
        input_ids=input_ids,
        positions=positions,
        seq_lens=seq_lens,
        block_table=block_table,
        attn_metadata=attn_metadata,
    )
```

---

## 注意力计算层

### 文件位置
```
vllm/v1/attention/backends/flash_attn.py    # FlashAttention 实现
vllm/v1/attention/ops/paged_attn.py         # PagedAttention 操作
```

### 关键代码

#### Attention Layer forward()
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
        FlashAttention 前向传播

        步骤：
        1. reshape_and_cache_flash: 将 K, V 写入 KV cache
        2. flash_attn_varlen_func: 计算注意力
        3. 返回注意力输出
        """

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Step 1: 将新的 K, V 写入 KV Cache
        if key is not None and value is not None:
            reshape_and_cache_flash(
                key,
                value,
                kv_cache[0],  # key_cache
                kv_cache[1],  # value_cache
                attn_metadata.slot_mapping,  # 写入位置
                self.kv_cache_dtype,
            )

        # Step 2: 计算 FlashAttention
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=kv_cache[0],     # key_cache
            v=kv_cache[1],     # value_cache
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            block_table=attn_metadata.block_table,  # PagedAttention
        )

        return output
```

### KV Cache 结构

```python
# KV Cache 形状
kv_cache.shape = [2, num_blocks, block_size, num_kv_heads, head_size]
#                ↑   ↑           ↑             ↑              ↑
#               K/V  物理block数   每块token数   KV头数      头维度

# Block Table (逻辑位置 → 物理 Block)
block_table[req_id] = [0, 1, 5, 3, ...]
#                          ↑  ↑  ↑  ↑
#                       逻辑token位置对应的物理block号

# Slot Mapping (当前token要写入的位置)
slot_mapping[i] = block_id * block_size + offset
```

### PagedAttention 工作原理

```
逻辑序列:
Token: [t0, t1, t2, t3, t4, t5, t6, t7, ...]
Block: [B0, B0, B0, B0, B1, B1, B1, B1, ...]
              ↓          ↓
物理KV Cache:
┌─────────┬─────────┬─────────┬─────┐
│ Block 0 │ Block 1 │ Block 2 │ ... │
│ [16 tok]│ [16 tok]│ [16 tok]│     │
└─────────┴─────────┴─────────┴─────┘

block_table: [0, 0, 0, 0, 1, 1, 1, 1, ...]
```

---

## 响应返回流程

### 完整数据流

```python
# 1. ModelRunner 采样
sampled_token = sampler(logits)  # → token_id

# 2. EngineCore 处理输出
engine_output = EngineCoreOutputs(
    request_id=request_id,
    outputs=[sampled_token],
    ...
)
output_queue.put(engine_output)  # ZMQ 发送

# 3. AsyncLLM 接收 (后台线程)
async def _output_handler(self):
    while True:
        engine_output = await self.engine_core.get_output_async()

        # 4. OutputProcessor 处理
        request_output = self.output_processor.process_output(engine_output)

        # 5. 格式化为 OpenAI 响应
        formatted = self._format_to_openai(request_output)

        # 6. 发送到 API Server
        yield formatted

# 7. API Server 流式返回
async def stream_results():
    async for chunk in generator:
        yield f"data: {json.dumps(chunk)}\n\n"

# 8. run_benchmark.sh 接收
async for chunk in response.content:
    chunk = chunk.decode("utf-8")
    data = json.loads(chunk.removeprefix("data: "))
    # 处理生成的 token
```

### 时间线示例

```
T0: run_benchmark.sh 发送 HTTP POST
T1: API Server 接收请求
T2: AsyncLLM 处理输入，添加到队列
T3: EngineCore 调度请求
T4: ModelRunner 执行 Prefill (处理所有输入 tokens)
T5: 返回第一个 token (TTFT - Time To First Token)
T6: ModelRunner 执行 Decode (逐个生成)
T7: 每个生成一个 token，立即返回
T8: 达到 max_tokens 或停止条件
T9: 返回最终响应
```

---

## 输入压缩预想方案

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
