import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

import aiohttp
import huggingface_hub.constants
from tqdm.asyncio import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

# Global request counter for tracking requests
_request_counter = 0
_request_counter_lock = None

def _get_next_request_id():
    """Get next request ID in a thread-safe manner"""
    global _request_counter, _request_counter_lock
    import asyncio
    
    if _request_counter_lock is None:
        _request_counter_lock = asyncio.Lock()
    
    # For now, just use a simple counter (will be replaced with proper async lock in actual usage)
    _request_counter += 1
    return _request_counter


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False
    logprobs: Optional[int] = None
    multi_modal_content: Optional[dict] = None
    request_id: Optional[int] = None  # Client-side request ID


@dataclass
class RequestFuncOutput:
    prompt: str = ""
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    arrival_time: Optional[float] = None
    first_scheduled_time: Optional[float] = None
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    time_in_queue: Optional[float] = None
    cumulative_logprob: Optional[float] = None


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    gpu_monitors=None,
    gpu_monitor_flag=None,
    cpu_monitors=None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        params = {
            "best_of": request_func_input.best_of,
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")

                        #NOTE: Sometimes TGI returns a ping response without
                        # any data, we should skip it.
                        if chunk_bytes.startswith(":"):
                            continue
                        chunk = remove_prefix(chunk_bytes, "data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    gpu_monitors=None,
    gpu_monitor_flag=None,
    cpu_monitors=None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        assert request_func_input.best_of == 1
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    gpu_monitors=None,
    gpu_monitor_flag=None,
    cpu_monitors=None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert request_func_input.best_of == 1
        assert not request_func_input.use_beam_search

        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=request_func_input.api_url,
                                    json=payload) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    output.generated_text = parsed_resp["text"][0]
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."
    
    # Assign client-side request ID if not already assigned
    if request_func_input.request_id is None:
        client_req_id = _get_next_request_id()
    else:
        client_req_id = request_func_input.request_id

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "logprobs": 10,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.prompt = request_func_input.prompt  # 记录 prompt

        # The async method does not provide insight into scheduling details, e.g., arrival times, first scheduled time, etc.
        # Only the client-side time is recorded. However, we may still record the request send time, ignoring the transmission delay and treating it as the arrival time. We can also record the completion time, and using completion time - latency, we can approximate the first scheduled time.
        # This is only useful for understanding the scheduling behavior of the engine.
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        token_count = 0
        server_req_id = "unknown"
        
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)
                            
                            # Get server request ID on first chunk
                            if token_count == 0:
                                server_req_id = data.get("id", "unknown")[-8:]

                            # 提取 cumulative_logprob（如果有的话）
                            if "logprobs" in data["choices"][0] and data["choices"][0]["logprobs"]:
                                token_logprobs = data["choices"][0]["logprobs"].get("token_logprobs", [])
                                if token_logprobs and token_logprobs[-1] is not None:
                                    if output.cumulative_logprob is None:
                                        output.cumulative_logprob = 0.0
                                    output.cumulative_logprob += token_logprobs[-1]

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                token_count += 1
                                token_text = data["choices"][0]["text"]
                                
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                    elapsed_ms = ttft * 1000
                                    print(f"[Client#{client_req_id:03d}|Server:{server_req_id}] Token #{token_count:3d} | {elapsed_ms:7.2f}ms | {repr(token_text)}")
                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)
                                    elapsed_ms = (timestamp - st) * 1000
                                    inter_token_ms = (timestamp - most_recent_timestamp) * 1000
                                    print(f"[Client#{client_req_id:03d}|Server:{server_req_id}] Token #{token_count:3d} | {elapsed_ms:7.2f}ms (+{inter_token_ms:5.2f}ms) | {repr(token_text)}")

                                most_recent_timestamp = timestamp
                                generated_text += token_text

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.completion_time = time.time()  # Use time.time() for absolute time
                    output.arrival_time = time.time() - latency  # Approximate arrival time
                    output.first_scheduled_time = time.time() - latency  # Approximate for OpenAI API
                    output.first_token_time = output.arrival_time + ttft if ttft > 0 else None
                    # 计算队列等待时间
                    if output.first_token_time and output.arrival_time:
                        output.time_in_queue = output.first_token_time - output.arrival_time
                    else:
                        output.time_in_queue = None
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    
    return output
async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.append(request_func_input.multi_modal_content)
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                },
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

async def async_request_vllm_demo(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    gpu_monitors: Optional[List] = None,
    gpu_monitor_flag: Optional[int] = None,
    cpu_monitors: Optional[List] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:

        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": 1,
            "stream": True,
        }
        if "im_start" in request_func_input.prompt:
            payload = {
                "prompt": request_func_input.prompt,
                "max_tokens": request_func_input.output_len,
                "temperature": 0.0,
                "top_p": 1.0,
                "logprobs": 1,
                "stop_token_ids": [
                                    151645,
                                    151643
                                ],
                "skip_special_tokens": False,
                "repetition_penalty": 1.05,
                "stop": ["<|im_end|>"],     
                "stream": True,
            }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.prompt = request_func_input.prompt
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output.arrival_time = time.time()
        if gpu_monitor_flag == 1:
            for gpu_monitor in gpu_monitors:
                gpu_monitor.start()
            if cpu_monitors is not None:
                for cpu_monitor in cpu_monitors:
                    cpu_monitor.start()
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        # print(chunk_bytes.decode("utf-8"))
                        data = json.loads(chunk)
                        if data["text"] == "":
                            continue
                        timestamp = time.perf_counter()

                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                            output.first_token_time = time.time()

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp
                        output.generated_text += data["text"]

                    output.latency = time.perf_counter() - st
                    output.success = True
                    output.completion_time = time.time()
                    output.first_scheduled_time = data["first_scheduled_time"]
                    output.time_in_queue = data["time_in_queue"]
                    output.cumulative_logprob = data["cumulative_logprob"]
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        if gpu_monitor_flag == 0:
            for gpu_monitor in gpu_monitors:
                gpu_monitor.stop()
            if cpu_monitors is not None:
                for cpu_monitor in cpu_monitors:
                    cpu_monitor.stop()
        return output



# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv('VLLM_USE_MODELSCOPE', 'False').lower() == 'true':
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str, trust_remote_code: bool
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
            pretrained_model_name_or_path):
        pretrained_model_name_or_path = get_model(
            pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                         trust_remote_code=trust_remote_code)


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "vllm-demo": async_request_vllm_demo,
}
