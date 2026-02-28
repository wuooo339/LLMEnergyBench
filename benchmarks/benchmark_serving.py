import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import time
import warnings
import pandas as pd
from itertools import cycle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

import numpy as np
import aiohttp
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput, RequestFuncOutput)
from datasets import load_dataset
from PIL.Image import Image
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    try:
        # Backward compatibility for older vLLM versions.
        from vllm.transformers_utils.tokenizer import get_tokenizer
    except ImportError:
        from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from util.monitor import GPUMonitor
from util.cpu_monitor import CPUMonitor
from util.kv_cache_monitor import KVCacheMonitor
from util.slurm import get_slurm_cpu_bind

POWER_STAT_WARM_UP_INTERVAL = 5


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


def sample_longbench_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
    max_context_len: Optional[int] = 16000,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    base_prompt = "Summarize the following text:\n"
    dataset = [(data['context'],data['answers'] if isinstance(data['answers'],str) else data['answers'][0]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = base_prompt + data[0]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_token_ids = tokenizer(text).input_ids

        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        output_len = max(output_len, 500)
        if prompt_len + output_len > max_context_len:
            # Prune too long sequences.
            continue
        
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset

def sample_humaneval_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    df = pd.read_parquet(dataset_path)
    dataset = [(row["prompt"], row["canonical_solution"][0]) for _, row in df.iterrows()]

    # Shuffle the dataset.
    random.shuffle(dataset)
    base_prompt = "Write a python program to complete the following code:\n"
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = base_prompt + data[0]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompt_token_ids = tokenizer(text).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len

        output_len = max(output_len, 1000)
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset

def sample_mbpp_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    
    df = pd.read_parquet(dataset_path)
    dataset = [(row["text"], row["code"]) for _, row in df.iterrows()]

    # Shuffle the dataset.
    random.shuffle(dataset)
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break
        # Tokenize the prompts and completions.
        prompt = data[0]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompt_token_ids = tokenizer(text).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue

        output_len = max(output_len, 1500)
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
    min_prompt_len: Optional[int] = None,
    server_max_model_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # Extract prompt and output pairs
    processed_dataset = []
    for data in dataset:
        conversations = data["conversations"]
        for i in range(len(conversations) - 1):  # Iterate until second-to-last element
            if conversations[i]["from"] == "human":
                prompt = conversations[i]["value"]
                output = conversations[i + 1]["value"]
                processed_dataset.append((prompt, output))
                break  # Only take the first human prompt and its response
    dataset = processed_dataset
    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_token_ids = tokenizer(text).input_ids
        real_prompt_token_ids = tokenizer(prompt).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len

        # Clamp output_len to avoid exceeding model context length
        # Use server-provided max_model_len if available, otherwise fall back to tokenizer
        if server_max_model_len is not None:
            max_model_len = server_max_model_len
        else:
            max_model_len = getattr(tokenizer, 'model_max_length', 4096)
            if max_model_len is None or max_model_len > 100000:
                max_model_len = 4096

        # Ensure prompt_len + output_len doesn't exceed max_model_len
        # Leave some headroom (50 tokens) for special tokens
        max_output_len = max_model_len - prompt_len - 50
        if output_len > max_output_len:
            output_len = max(4, max_output_len)

        # output_len = 5000  # Commented out to allow --sharegpt-output-len to work
        if len(real_prompt_token_ids) < 10 or output_len < 4:
            # Prune too short sequences.
            continue
        
        # 筛选最小prompt长度（如果指定）
        if min_prompt_len is not None and prompt_len < min_prompt_len:
            continue
        if prompt_len > 10000:
            # Prune too long sequences.
            continue
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset


def sample_lmsyschat_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    df = pd.read_parquet(dataset_path)
    dataset = [(row["conversation"][0]["content"], row["conversation"][1]["content"]) for _, row in df.iterrows()]
    
    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_token_ids = tokenizer(text).input_ids
        real_prompt_token_ids = tokenizer(prompt).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        
        output_len = 4000
        if len(real_prompt_token_ids) < 10 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 3000:
            # Prune too long sequences.
            continue
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset



def sample_alpaca_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    dataset = [(data["instruction"],
                data["output"]) for data in dataset]
    
    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = data[0]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_token_ids = tokenizer(text).input_ids
        real_prompt_token_ids = tokenizer(prompt).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        
        output_len = 2000
        if len(real_prompt_token_ids) < 10 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 3000:
            # Prune too long sequences.
            continue
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset


def sample_arenahard_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    df = pd.read_parquet(dataset_path)
    dataset = [(row["turns"][0]["content"], row["question_id"]) for _, row in df.iterrows()]
    print(fixed_input_len)
    print(fixed_output_len)
    # Shuffle the dataset.
    random.shuffle(dataset)
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = data[0]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompt_token_ids = tokenizer(text).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
 
        output_len = max(output_len, 3000)
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset

def sample_summeval_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    df = pd.read_parquet(dataset_path)
    dataset = [(row["text"], row["human_summaries"][0]) for _, row in df.iterrows()]
    
    
    # Shuffle the dataset.
    random.shuffle(dataset)
    base_prompt = "Summarize the following text:\n"
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    i = 0
    while len(filtered_dataset) < num_requests:
        current_index = i % len(dataset)
        
        # Tokenize the prompts and completions
        prompt = base_prompt + dataset[current_index][0]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_token_ids = tokenizer(text).input_ids
        
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                i += 1
                continue
                
        completion = dataset[current_index][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        output_len = max(output_len, 200)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences
            i += 1
            continue
            
        filtered_dataset.append((text, prompt_len, output_len, None))
        i += 1


    return filtered_dataset


def sample_newsqa_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
    max_context_len: Optional[int] = 16000,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line.strip()) for line in f]
    base_prompt = "Summarize the following text:\n"
    dataset = [(data['story'], data['summary'])  for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions
        prompt = base_prompt + data[0]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_token_ids = tokenizer(text).input_ids
        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        output_len = max(output_len, 200)
        if prompt_len + output_len > max_context_len:
            # Prune too long sequences.
            continue
        
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset


def sample_gsm8k_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_input_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
    max_context_len: Optional[int] = 16000,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    
    base_prompt = "Quesion:\n"
    df = pd.read_parquet(dataset_path)
    dataset = [(row["question"], row["answer"]) for _, row in df.iterrows()]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for data in cycle(dataset):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = base_prompt + data[0]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_token_ids = tokenizer(text).input_ids

        if fixed_input_len is not None:
            if len(prompt_token_ids) >= fixed_input_len:
                prompt = tokenizer.decode(prompt_token_ids[:fixed_input_len])
                prompt_token_ids = prompt_token_ids[:fixed_input_len]
            else:
                continue
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        output_len = max(output_len, 1000)
        if prompt_len + output_len > max_context_len:
            # Prune too long sequences.
            continue
        
        filtered_dataset.append((text, prompt_len, output_len, None))

    return filtered_dataset

def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int, None]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(''.join(poem_lines)).input_ids
    # print(poem_token_ids)
    # average_poem_len = sum(
    #     len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)
    # FIXME The original implemenation will inflate the average poem length by about 1,
    # which will lead to the result that when you specify --input-len 2048, the actual input length
    # is only ~1900. This is because the tokenizer will add special tokens to the input_ids for each line.
    # But when you concatenate all the lines together, the special tokens will only appear once.
    # Inflated average poem length means less lines will be sampled in each request.
    average_poem_len = len(poem_token_ids) / len(poem_lines)
    # print("Average_poem_len", average_poem_len)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(
            prefix_lines +
            random.sample(poem_lines, num_input_lines - num_prefix_lines))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len, None))

    return sampled_requests


def sample_hf_requests(
    dataset_path: str,
    dataset_subset: str,
    dataset_split: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, str, int, Optional[Dict[str, Collection[str]]]]]:
    dataset = load_dataset(dataset_path,
                           name=dataset_subset,
                           split=dataset_split,
                           streaming=True)
    assert "conversations" in dataset.features, (
        "HF Dataset must have 'conversations' column.")
    filtered_dataset = dataset.shuffle().filter(
        lambda x: len(x["conversations"]) >= 2)
    sampled_requests: List[Tuple[str, int, int, Dict[str,
                                                     Collection[str]]]] = []
    for data in filtered_dataset:
        if len(sampled_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = data["conversations"][0]["value"]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = data["conversations"][1]["value"]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue

        if "image" in data and isinstance(data["image"], Image):
            image: Image = data["image"]
            image = image.convert("RGB")
            image_data = io.BytesIO()
            image.save(image_data, format='JPEG')
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
            mm_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            }
        else:
            mm_content = None

        sampled_requests.append((prompt, prompt_len, output_len, mm_content))

    return sampled_requests


def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    prefix_token_ids = np.random.randint(0,
                                         tokenizer.vocab_size,
                                         size=prefix_len).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(prefix_token_ids +
                                  [(offsets[i] + i + j) % tokenizer.vocab_size
                                   for j in range(input_lens[i])])

        input_requests.append((prompt, int(prefix_len + input_lens[i]),
                               int(output_lens[i]), None))

    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    use_deterministic_rate: bool = False,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        
        if use_deterministic_rate:
            interval = 1.0 / request_rate
        else:
            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # Prefer server-reported completion_tokens when available.
            # This is more accurate for speculative decoding than re-tokenizing
            # generated text on the client.
            if outputs[i].completion_tokens is not None:
                output_len = max(int(outputs[i].completion_tokens), 0)
            else:
                # Fallback for backends not returning usage in stream.
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.median(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.mean(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    logprobs: Optional[int],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    gpu_monitors: Optional[List[GPUMonitor]] = None,
    cpu_monitor: Optional[CPUMonitor] = None,
    kv_cache_monitor: Optional[KVCacheMonitor] = None,
    warmup_ratio: float = 0.1,
    carbon_params: Optional[Dict[str, Any]] = None,
    server_max_model_len: Optional[int] = None,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    print(get_slurm_cpu_bind())
    print("Starting initial single prompt test run...")
    print(input_requests[0])
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0])

    # Clamp output_len to avoid exceeding model context length
    # Use server-provided max_model_len if available, otherwise fall back to tokenizer
    if server_max_model_len is not None:
        max_model_len = server_max_model_len
        print(f"Using server-reported max_model_len: {max_model_len}")
    else:
        max_model_len = getattr(tokenizer, 'model_max_length', 4096)
        if max_model_len is None or max_model_len > 100000:
            max_model_len = 4096
        print(f"Using tokenizer max_model_len: {max_model_len}")

    # Ensure prompt_len + output_len doesn't exceed max_model_len
    # Leave some headroom (50 tokens) for special tokens
    max_output_len = max_model_len - test_prompt_len - 50
    if test_output_len > max_output_len:
        print(f"WARNING: Clamping output_len from {test_output_len} to {max_output_len} "
              f"to avoid exceeding model max length ({max_model_len}) "
              f"with prompt_len ({test_prompt_len})")
        test_output_len = max(4, max_output_len)  # Ensure at least 4 tokens
    print(f"Test will use prompt_len={test_prompt_len}, output_len={test_output_len}")

    if backend != "openai-chat" and test_mm_content is not None:
        # multi-modal benchmark is only available on OpenAI Chat backend.
        raise ValueError(
            "Multi-modal content is only supported on 'openai-chat' backend.")
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        best_of=best_of,
        use_beam_search=use_beam_search,
        multi_modal_content=test_mm_content,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
            use_beam_search=use_beam_search,
            multi_modal_content=test_mm_content,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    print(f"Traffic request rate: {request_rate}")
    print("Start warming up...")
    warmup_length = int(len(input_requests) * warmup_ratio)
    warmup_requests = input_requests[:warmup_length].copy()
    if warmup_length == 0:
        end_up_requests = []
    else:
        end_up_requests = input_requests[-warmup_length:].copy()

    pbar = None if disable_tqdm else tqdm(total=len(input_requests)+warmup_length*2)

    tasks: List[asyncio.Task] = []
    warmup_cnt = 0
    async for request in get_request(warmup_requests + input_requests + end_up_requests, request_rate, use_deterministic_rate=args.use_deterministic_rate):
        prompt, prompt_len, output_len, mm_content = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            best_of=best_of,
            use_beam_search=use_beam_search,
            multi_modal_content=mm_content,
        )
        if warmup_cnt == warmup_length:
            print("Warmup completed. Starting benchmark...")
            benchmark_start_time = time.time()
            # 启动GPU、CPU和KV Cache监控器
            if gpu_monitors:
                for gpu_monitor in gpu_monitors:
                    gpu_monitor.start()
            if cpu_monitor:
                cpu_monitor.start()
            if kv_cache_monitor:
                kv_cache_monitor.start()
                print("KV Cache monitoring started")
            
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                            pbar=pbar)))
        warmup_cnt += 1
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    outputs = outputs[warmup_length:warmup_length + len(input_requests)]
    
    # 停止GPU、CPU和KV Cache监控器
    if gpu_monitors:
        for gpu_monitor in gpu_monitors:
            gpu_monitor.stop()
    if cpu_monitor:
        cpu_monitor.stop()
    if kv_cache_monitor:
        kv_cache_monitor.stop()
        print("KV Cache monitoring stopped")
    print("Monitoring stopped. Processing results...")
    
    # Calculate benchmark duration - handle case where completion_time might be None
    if outputs and outputs[-1].completion_time is not None:
        benchmark_duration = outputs[-1].completion_time - benchmark_start_time
    else:
        benchmark_duration = time.time() - benchmark_start_time

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    

    # for output in outputs:
    #     print(output)

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "start_time": benchmark_start_time,
        "end_time": outputs[-1].completion_time,
        "truncated_time": args.gpu_monitor_truncate,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "cumulative_logprob": [output.cumulative_logprob for output in outputs],
        "prompt_tokens_from_server": [output.prompt_tokens for output in outputs],
        "completion_tokens_from_server": [
            output.completion_tokens for output in outputs
        ],
        "total_tokens_from_server": [output.total_tokens for output in outputs],
        "output_token_count_source": [
            (
                "server_usage"
                if output.completion_tokens is not None
                else "tokenizer_fallback"
            )
            for output in outputs
        ],
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "e2els": [output.latency for output in outputs],
        "arrival_times": [output.arrival_time for output in outputs],
        "first_scheduled_times": [output.first_scheduled_time for output in outputs],
        "first_token_times": [output.first_token_time for output in outputs],
        "completion_times": [output.completion_time for output in outputs],
        "queuing_delays": [output.time_in_queue for output in outputs],
        "prompt": [output.prompt for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "finish_reasons": [output.finish_reason for output in outputs],
        "stop_reasons": [output.stop_reason for output in outputs],
    }
    result["per_request"] = [
        {
            "request_index": i,
            "prompt_len": outputs[i].prompt_len,
            "output_len": actual_output_lens[i] if i < len(actual_output_lens) else 0,
            "ttft_s": outputs[i].ttft,
            "e2e_latency_s": outputs[i].latency,
            "success": bool(outputs[i].success),
            "error": outputs[i].error,
            "finish_reason": outputs[i].finish_reason,
            "stop_reason": outputs[i].stop_reason,
            "prompt_tokens_from_server": outputs[i].prompt_tokens,
            "completion_tokens_from_server": outputs[i].completion_tokens,
            "total_tokens_from_server": outputs[i].total_tokens,
        }
        for i in range(len(outputs))
    ]

    if gpu_monitors is not None:
        result["gpu_power_stats"] = {}
        for gpu_monitor in gpu_monitors:
            avg_power, avg_gpu_util, avg_mem_util = gpu_monitor.results_queue.get()
            power_stats = gpu_monitor.stats_queue.get()
            power_trace = gpu_monitor.hist_queue.get()
            print("{s:{c}^{n}}".format(s=f'GPU Power Consumption (W) for GPU {gpu_monitor.gpu_id}', n=50, c='-'))
            print("{:<40} {:<10.2f}".format("Average Power (W):", avg_power/1000))
            print("{:<40} {:<10.2f}".format("Average GPU Utilization (%):",
                                            avg_gpu_util))
            print("{:<40} {:<10.2f}".format("Average Memory Utilization (%):",
                                            avg_mem_util))
            print("{s:{c}^{n}}".format(s='Power Statistics (W)', n=50, c='-'))
            print("{:<40} {:<10.2f}".format("Min Power (W):", power_stats["min_power"]/1000))
            print("{:<40} {:<10.2f}".format("5th Percentile Power (W):",
                                            power_stats["power_5p"]/1000))
            print("{:<40} {:<10.2f}".format("25th Percentile Power (W):",
                                            power_stats["power_25p"]/1000))
            print("{:<40} {:<10.2f}".format("Median Power (W):",
                                            power_stats["median_power"]/1000))
            print("{:<40} {:<10.2f}".format("75th Percentile Power (W):",
                                            power_stats["power_75p"]/1000))
            print("{:<40} {:<10.2f}".format("95th Percentile Power (W):",
                                            power_stats["power_95p"]/1000))
            print("{:<40} {:<10.2f}".format("Max Power (W):", power_stats["max_power"]/1000))
            print("{:<40} {:<10.2f}".format("Power Standard Deviation (W):",
                                            power_stats["power_std"]))
            memory_used_mb_trace = power_trace[2] if len(power_trace) > 2 else []
            result["gpu_power_stats"][gpu_monitor.gpu_id] = {
                "avg_power": avg_power,
                "avg_gpu_util": avg_gpu_util,
                "avg_mem_util": avg_mem_util,
                "power_stats": power_stats,
                "power_trace": power_trace[0],
                "memory_util_trace": power_trace[1],
                "memory_used_mb_trace": memory_used_mb_trace,
            }
        result["energy_stats"] = {}
        result["energy_stats"]["total_energy"] = result['duration'] * sum([result["gpu_power_stats"][gpu_monitor.gpu_id]["avg_power"] for gpu_monitor in gpu_monitors]) / 1000
        result["energy_stats"]["energy_per_request"] = result["energy_stats"]["total_energy"] / metrics.completed
        result["energy_stats"]["energy_per_token"] = result["energy_stats"]["total_energy"] / (metrics.total_output + metrics.total_input)
        print("{s:{c}^{n}}".format(s='Energy Consumption (J)', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Total Energy (J):", result["energy_stats"]["total_energy"]))
        print("{:<40} {:<10.8f}".format("Total Energy (kWh):", result["energy_stats"]["total_energy"]/3600/1000))
        print("{:<40} {:<10.2f}".format("Energy per Request (J):", result["energy_stats"]["energy_per_request"]))
        print("{:<40} {:<10.2f}".format("Energy per Token processed (J):", result["energy_stats"]["energy_per_token"]))

        if carbon_params is not None:
            result["carbon_stats"] = {}
            result["carbon_stats"]["operational_carbon"] = (result["energy_stats"]["total_energy"]/3600/1000) * carbon_params["carbon_intensity"] 
            result["carbon_stats"]["embodied_carbon"] = (carbon_params["embodied_carbon"] * 1000 / carbon_params["device_lifetime"] / 365 / 24 / 3600) * result["duration"] 
            result["carbon_stats"]["total_carbon"] = result["carbon_stats"]["operational_carbon"] + result["carbon_stats"]["embodied_carbon"]
            result["carbon_stats"]["operational_carbon_per_request"] = result["carbon_stats"]["operational_carbon"] / metrics.completed
            result["carbon_stats"]["operational_carbon_per_token"] = result["carbon_stats"]["operational_carbon"] / (metrics.total_output + metrics.total_input)
            result["carbon_stats"]["embodied_carbon_per_request"] = result["carbon_stats"]["embodied_carbon"] / metrics.completed
            result["carbon_stats"]["embodied_carbon_per_token"] = result["carbon_stats"]["embodied_carbon"] / (metrics.total_output + metrics.total_input)
            result["carbon_stats"]["carbon_per_request"] = result["carbon_stats"]["total_carbon"] / metrics.completed
            result["carbon_stats"]["carbon_per_token"] = result["carbon_stats"]["total_carbon"] / (metrics.total_output + metrics.total_input)
            print("{s:{c}^{n}}".format(s='Carbon Emissions (gCO2e)', n=50, c='-'))
            print("{:<40} {:<10.4f}".format("Operational Carbon (gCO2e):", result["carbon_stats"]["operational_carbon"]))
            print("{:<40} {:<10.4f}".format("Embodied Carbon (gCO2e):", result["carbon_stats"]["embodied_carbon"]))
            print("{:<40} {:<10.4f}".format("Total Carbon (gCO2e):", result["carbon_stats"]["total_carbon"]))
            print("{:<40} {:<10.4f}".format("Operational Carbon per Request (gCO2e):", result["carbon_stats"]["operational_carbon_per_request"]))
            print("{:<40} {:<10.6f}".format("Operational Carbon per Token processed (gCO2e):", result["carbon_stats"]["operational_carbon_per_token"]))
            print("{:<40} {:<10.6f}".format("Embodied Carbon per Request (gCO2e):", result["carbon_stats"]["embodied_carbon_per_request"]))
            print("{:<40} {:<10.6f}".format("Embodied Carbon per Token processed (gCO2e):", result["carbon_stats"]["embodied_carbon_per_token"]))
            print("{:<40} {:<10.4f}".format("Total Carbon per Request (gCO2e):", result["carbon_stats"]["carbon_per_request"]))
            print("{:<40} {:<10.6f}".format("Total Carbon per Token processed (gCO2e):", result["carbon_stats"]["carbon_per_token"]))
    print("=" * 50)

    if cpu_monitor is not None:
        result["cpu_stats"] = cpu_monitor.stats_queue.get()
        result["cpu_trace"] = {}
        result["binded_cpus"] = get_slurm_cpu_bind()
        cpu_utilization_readings, cpu_freq_readings, disk_io_readings, mem_utilization_readings = cpu_monitor.hist_queue.get()
        result["cpu_trace"]["cpu_utils"] = cpu_utilization_readings
        result["cpu_trace"]["cpu_freqs"] = cpu_freq_readings
        result["cpu_trace"]["disk_io"] = disk_io_readings
        result["cpu_trace"]["mem_utils"] = mem_utilization_readings
    
    # Collect KV cache monitoring results (synchronized with GPU/CPU monitoring)
    if kv_cache_monitor is not None:
        avg_stats = kv_cache_monitor.results_queue.get()
        detailed_stats = kv_cache_monitor.stats_queue.get()
        kv_trace = kv_cache_monitor.hist_queue.get()
        
        print("\n" + "=" * 70)
        print(
            "{s:^{n}}".format(
                s='KV Cache / PageEviction / Spec Metrics (Synchronized)',
                n=70,
            )
        )
        print("=" * 70)
        print("{:<50} {:<20.2f}".format("Average Cache Usage (%):", avg_stats['avg_cache_usage_perc']))
        print("{:<50} {:<20,.0f}".format("Average Used Blocks:", avg_stats['avg_used_blocks']))
        print("{:<50} {:<20,.0f}".format("Average Free Blocks:", avg_stats['avg_free_blocks']))
        print("{:<50} {:<20,.0f}".format("Average Used Tokens:", avg_stats['avg_used_tokens']))
        print("{:<50} {:<20.1f}".format("Average Running Requests:", avg_stats['avg_requests_running']))
        print("{:<50} {:<20.1f}".format("Average Waiting Requests:", avg_stats['avg_requests_waiting']))

        if avg_stats.get("total_page_eviction_ops", 0) > 0 or avg_stats.get(
            "total_page_eviction_blocks", 0
        ) > 0:
            print("\n" + "-" * 70)
            print("{s:^{n}}".format(s='PageEviction Statistics', n=70))
            print("-" * 70)
            print(
                "{:<50} {:<20,.0f}".format(
                    "Total Eviction Ops:", avg_stats.get("total_page_eviction_ops", 0)
                )
            )
            print(
                "{:<50} {:<20,.0f}".format(
                    "Total Evicted Blocks:",
                    avg_stats.get("total_page_eviction_blocks", 0),
                )
            )
            print(
                "{:<50} {:<20.3f}".format(
                    "Avg Eviction Ops/Sample:",
                    avg_stats.get("avg_page_eviction_ops_per_sample", 0),
                )
            )
            print(
                "{:<50} {:<20.3f}".format(
                    "Avg Evicted Blocks/Sample:",
                    avg_stats.get("avg_page_eviction_blocks_per_sample", 0),
                )
            )

        if avg_stats.get("total_spec_draft_tokens", 0) > 0:
            print("\n" + "-" * 70)
            print("{s:^{n}}".format(s='Spec Decode Statistics', n=70))
            print("-" * 70)
            print(
                "{:<50} {:<20,.0f}".format(
                    "Total Draft Tokens:",
                    avg_stats.get("total_spec_draft_tokens", 0),
                )
            )
            print(
                "{:<50} {:<20,.0f}".format(
                    "Total Accepted Tokens:",
                    avg_stats.get("total_spec_accepted_tokens", 0),
                )
            )
            print(
                "{:<50} {:<20.4f}".format(
                    "Acceptance Rate:",
                    avg_stats.get("spec_decode_acceptance_rate", 0),
                )
            )
        
        if 'cache_usage' in detailed_stats:
            print("\n" + "-" * 70)
            print("{s:^{n}}".format(s='KV Cache Usage Distribution (%)', n=70))
            print("-" * 70)
            print("{:<50} {:<20.2f}".format("Min:", detailed_stats['cache_usage']['min']))
            print("{:<50} {:<20.2f}".format("25th Percentile:", detailed_stats['cache_usage']['p25']))
            print("{:<50} {:<20.2f}".format("Median:", detailed_stats['cache_usage']['median']))
            print("{:<50} {:<20.2f}".format("75th Percentile:", detailed_stats['cache_usage']['p75']))
            print("{:<50} {:<20.2f}".format("95th Percentile:", detailed_stats['cache_usage']['p95']))
            print("{:<50} {:<20.2f}".format("Max:", detailed_stats['cache_usage']['max']))
        
        if 'used_blocks' in detailed_stats:
            print("\n" + "-" * 70)
            print("{s:^{n}}".format(s='Used KV Blocks Distribution', n=70))
            print("-" * 70)
            print("{:<50} {:<20,}".format("Min:", detailed_stats['used_blocks']['min']))
            print("{:<50} {:<20,}".format("Median:", detailed_stats['used_blocks']['median']))
            print("{:<50} {:<20,}".format("Max:", detailed_stats['used_blocks']['max']))
        
        print("=" * 70 + "\n")
        
        result["kv_cache_monitoring_stats"] = {
            "avg_stats": avg_stats,
            "detailed_stats": detailed_stats,
            "trace": kv_trace,
            "static_config": detailed_stats.get('static_config', {}),
            "monitoring_interval_seconds": kv_cache_monitor.interval,
            "samples_collected": len(kv_trace.get('cache_usage', [])),
        }

        result["page_eviction_monitoring_stats"] = {
            "total_eviction_ops": avg_stats.get("total_page_eviction_ops", 0),
            "total_eviction_ops_prefill": avg_stats.get(
                "total_page_eviction_ops_prefill", 0
            ),
            "total_eviction_ops_decode": avg_stats.get(
                "total_page_eviction_ops_decode", 0
            ),
            "total_evicted_blocks": avg_stats.get("total_page_eviction_blocks", 0),
            "total_prefill_reqs_scheduled": avg_stats.get(
                "total_page_eviction_prefill_reqs_scheduled", 0
            ),
            "total_prefill_reqs_query_len_gt_budget": avg_stats.get(
                "total_page_eviction_prefill_reqs_query_len_gt_budget", 0
            ),
            "prefill_query_len_gt_budget_ratio": avg_stats.get(
                "prefill_query_len_gt_budget_ratio", 0
            ),
            "total_replace_block_req_ids": avg_stats.get(
                "total_page_eviction_replace_block_req_ids", 0
            ),
            "replace_block_req_ids_count_per_sample_p50": avg_stats.get(
                "replace_block_req_ids_count_per_sample_p50", 0
            ),
            "replace_block_req_ids_count_per_sample_p90": avg_stats.get(
                "replace_block_req_ids_count_per_sample_p90", 0
            ),
            "replace_block_req_ids_count_per_sample_p99": avg_stats.get(
                "replace_block_req_ids_count_per_sample_p99", 0
            ),
            "total_score_collect_calls_single": avg_stats.get(
                "total_page_eviction_score_collect_calls_single", 0
            ),
            "total_score_collect_calls_ubatch_list": avg_stats.get(
                "total_page_eviction_score_collect_calls_ubatch_list", 0
            ),
            "total_score_collect_return_none_ubatch_list": avg_stats.get(
                "total_page_eviction_score_collect_return_none_ubatch_list", 0
            ),
            "score_collect_return_none_ubatch_ratio": avg_stats.get(
                "score_collect_return_none_ubatch_ratio", 0
            ),
            "total_prefill_block_scores_returned": avg_stats.get(
                "total_page_eviction_prefill_block_scores_returned", 0
            ),
            "total_decode_token_scores_returned": avg_stats.get(
                "total_page_eviction_decode_token_scores_returned", 0
            ),
            "total_prefill_compress_invocations": avg_stats.get(
                "total_page_eviction_prefill_compress_invocations", 0
            ),
            "prefill_compress_invocations_per_request_mean": avg_stats.get(
                "prefill_compress_invocations_per_request_mean", 0
            ),
            "total_request_success": avg_stats.get("total_request_success", 0),
            "prefill_compress_calls_per_request_mean": avg_stats.get(
                "prefill_compress_calls_per_request_mean", 0
            ),
            "prefill_compress_calls_per_request_p99": avg_stats.get(
                "prefill_compress_calls_per_request_p99", 0
            ),
            "prefill_compress_calls_per_request_max": avg_stats.get(
                "prefill_compress_calls_per_request_max", 0
            ),
            "prefill_compress_time_ms_total": avg_stats.get(
                "prefill_compress_time_ms_total", 0
            ),
            "prefill_compress_time_ms_per_event_p50": avg_stats.get(
                "prefill_compress_time_ms_per_event_p50", 0
            ),
            "prefill_compress_time_ms_per_event_p90": avg_stats.get(
                "prefill_compress_time_ms_per_event_p90", 0
            ),
            "prefill_compress_time_ms_per_event_p99": avg_stats.get(
                "prefill_compress_time_ms_per_event_p99", 0
            ),
            "prefill_keep_len_mean": avg_stats.get("prefill_keep_len_mean", 0),
            "prefill_keep_len_p90": avg_stats.get("prefill_keep_len_p90", 0),
            "prefill_kept_ratio_mean": avg_stats.get("prefill_kept_ratio_mean", 0),
            "prefill_kept_ratio_p90": avg_stats.get("prefill_kept_ratio_p90", 0),
            "decode_eviction_ops_per_request_p50": avg_stats.get(
                "decode_eviction_ops_per_request_p50", 0
            ),
            "decode_eviction_ops_per_request_p90": avg_stats.get(
                "decode_eviction_ops_per_request_p90", 0
            ),
            "decode_eviction_ops_per_request_p99": avg_stats.get(
                "decode_eviction_ops_per_request_p99", 0
            ),
            "decode_evicted_blocks_per_op_p50": avg_stats.get(
                "decode_evicted_blocks_per_op_p50", 0
            ),
            "decode_evicted_blocks_per_op_p90": avg_stats.get(
                "decode_evicted_blocks_per_op_p90", 0
            ),
            "decode_evicted_blocks_per_op_p99": avg_stats.get(
                "decode_evicted_blocks_per_op_p99", 0
            ),
            "decode_eviction_time_ms_per_op_p50": avg_stats.get(
                "decode_eviction_time_ms_per_op_p50", 0
            ),
            "decode_eviction_time_ms_per_op_p90": avg_stats.get(
                "decode_eviction_time_ms_per_op_p90", 0
            ),
            "decode_eviction_time_ms_per_op_p99": avg_stats.get(
                "decode_eviction_time_ms_per_op_p99", 0
            ),
            "decode_pages_scored_per_op_p50": avg_stats.get(
                "decode_pages_scored_per_op_p50", 0
            ),
            "decode_pages_scored_per_op_p90": avg_stats.get(
                "decode_pages_scored_per_op_p90", 0
            ),
            "decode_pages_scored_per_op_p99": avg_stats.get(
                "decode_pages_scored_per_op_p99", 0
            ),
            "active_concurrency_p50": avg_stats.get("active_concurrency_p50", 0),
            "active_concurrency_p90": avg_stats.get("active_concurrency_p90", 0),
            "num_prefill_tokens_scheduled_p50": avg_stats.get(
                "prefill_tokens_scheduled_p50", 0
            ),
            "num_prefill_tokens_scheduled_p90": avg_stats.get(
                "prefill_tokens_scheduled_p90", 0
            ),
            "num_prefill_tokens_scheduled_p99": avg_stats.get(
                "prefill_tokens_scheduled_p99", 0
            ),
            "num_decode_tokens_scheduled_p50": avg_stats.get(
                "decode_tokens_scheduled_p50", 0
            ),
            "num_decode_tokens_scheduled_p90": avg_stats.get(
                "decode_tokens_scheduled_p90", 0
            ),
            "num_decode_tokens_scheduled_p99": avg_stats.get(
                "decode_tokens_scheduled_p99", 0
            ),
            "avg_eviction_ops_per_sample": avg_stats.get(
                "avg_page_eviction_ops_per_sample", 0
            ),
            "avg_evicted_blocks_per_sample": avg_stats.get(
                "avg_page_eviction_blocks_per_sample", 0
            ),
            "trace_eviction_ops_delta": kv_trace.get("page_eviction_ops_delta", []),
            "trace_eviction_ops_prefill_delta": kv_trace.get(
                "page_eviction_ops_prefill_delta", []
            ),
            "trace_eviction_ops_decode_delta": kv_trace.get(
                "page_eviction_ops_decode_delta", []
            ),
            "trace_prefill_reqs_scheduled_delta": kv_trace.get(
                "page_eviction_prefill_reqs_scheduled_delta", []
            ),
            "trace_prefill_reqs_query_len_gt_budget_delta": kv_trace.get(
                "page_eviction_prefill_reqs_query_len_gt_budget_delta", []
            ),
            "trace_replace_block_req_ids_delta": kv_trace.get(
                "page_eviction_replace_block_req_ids_delta", []
            ),
            "trace_score_collect_calls_single_delta": kv_trace.get(
                "page_eviction_score_collect_calls_single_delta", []
            ),
            "trace_score_collect_calls_ubatch_list_delta": kv_trace.get(
                "page_eviction_score_collect_calls_ubatch_list_delta", []
            ),
            "trace_score_collect_return_none_ubatch_list_delta": kv_trace.get(
                "page_eviction_score_collect_return_none_ubatch_list_delta", []
            ),
            "trace_prefill_block_scores_returned_delta": kv_trace.get(
                "page_eviction_prefill_block_scores_returned_delta", []
            ),
            "trace_decode_token_scores_returned_delta": kv_trace.get(
                "page_eviction_decode_token_scores_returned_delta", []
            ),
            "trace_prefill_compress_invocations_delta": kv_trace.get(
                "page_eviction_prefill_compress_invocations_delta", []
            ),
            "trace_evicted_blocks_delta": kv_trace.get(
                "page_eviction_blocks_delta", []
            ),
            "trace_prefill_compress_calls_per_request": kv_trace.get(
                "prefill_compress_calls_per_request", []
            ),
            "trace_active_concurrency": kv_trace.get("active_concurrency", []),
            "trace_prefill_tokens_scheduled_delta": kv_trace.get(
                "prompt_tokens_delta", []
            ),
            "trace_decode_tokens_scheduled_delta": kv_trace.get(
                "generation_tokens_delta", []
            ),
            "trace_decode_evicted_blocks_per_op": kv_trace.get(
                "decode_evicted_blocks_per_op", []
            ),
            "trace_decode_eviction_time_ms_per_op": kv_trace.get(
                "decode_eviction_time_ms_per_op", []
            ),
            "trace_decode_pages_scored_per_op": kv_trace.get(
                "decode_pages_scored_per_op", []
            ),
            "trace_prefill_compress_time_ms_per_event": kv_trace.get(
                "prefill_compress_time_ms_per_event", []
            ),
            "trace_prefill_keep_len": kv_trace.get("prefill_keep_len", []),
            "trace_prefill_kept_ratio": kv_trace.get("prefill_kept_ratio", []),
        }

        result["spec_decode_monitoring_stats"] = {
            "total_draft_tokens": avg_stats.get("total_spec_draft_tokens", 0),
            "total_accepted_tokens": avg_stats.get(
                "total_spec_accepted_tokens", 0
            ),
            "acceptance_rate": avg_stats.get("spec_decode_acceptance_rate", 0),
            "avg_acceptance_rate": avg_stats.get(
                "avg_spec_decode_acceptance_rate", 0
            ),
            "trace_acceptance_rate": kv_trace.get("spec_decode_acceptance_rate", []),
        }

    def _safe_percentile(values: List[float], p: float) -> Optional[float]:
        if not values:
            return None
        return float(np.percentile(values, p))

    def _weighted_tpot_ms(
        output_lens: List[int],
        ttfts: List[float],
        e2els: List[float],
        *,
        min_output_len: int = 2,
        selected_indices: Optional[List[int]] = None,
    ) -> Optional[float]:
        decode_time_sum = 0.0
        decode_token_count = 0
        indices = selected_indices if selected_indices is not None else list(
            range(min(len(output_lens), len(ttfts), len(e2els)))
        )
        for idx in indices:
            if idx >= len(output_lens) or idx >= len(ttfts) or idx >= len(e2els):
                continue
            out_len = int(output_lens[idx])
            if out_len < max(min_output_len, 2):
                continue
            decode_time = max(float(e2els[idx]) - float(ttfts[idx]), 0.0)
            decode_time_sum += decode_time
            decode_token_count += out_len - 1
        if decode_token_count <= 0:
            return None
        return decode_time_sum / decode_token_count * 1000.0

    output_lens_for_stats = [int(x) for x in result.get("output_lens", [])]
    input_lens_for_stats = [int(x) for x in result.get("input_lens", [])]
    ttfts_for_stats = [float(x) for x in result.get("ttfts", [])]
    e2els_for_stats = [float(x) for x in result.get("e2els", [])]
    errors_for_stats = result.get("errors", [])
    finish_reasons_for_stats = result.get("finish_reasons", [])
    stop_reasons_for_stats = result.get("stop_reasons", [])

    result["output_len_distribution"] = {
        "mean": (float(np.mean(output_lens_for_stats)) if output_lens_for_stats else None),
        "p50": _safe_percentile(output_lens_for_stats, 50),
        "p90": _safe_percentile(output_lens_for_stats, 90),
        "p99": _safe_percentile(output_lens_for_stats, 99),
        "ratio_out_len_lt_16": (
            float(sum(1 for x in output_lens_for_stats if x < 16)) / len(output_lens_for_stats)
            if output_lens_for_stats
            else None
        ),
        "ratio_out_len_lt_64": (
            float(sum(1 for x in output_lens_for_stats if x < 64)) / len(output_lens_for_stats)
            if output_lens_for_stats
            else None
        ),
        "ratio_out_len_lt_200": (
            float(sum(1 for x in output_lens_for_stats if x < 200))
            / len(output_lens_for_stats)
            if output_lens_for_stats
            else None
        ),
    }

    finish_reason_counts = {
        "eos": 0,
        "length": 0,
        "stop": 0,
        "error": 0,
        "abort": 0,
        "unknown": 0,
    }
    n_finish = min(
        len(output_lens_for_stats),
        len(errors_for_stats),
        len(finish_reasons_for_stats),
    )
    for i in range(n_finish):
        err = errors_for_stats[i]
        fr = finish_reasons_for_stats[i]
        sr = stop_reasons_for_stats[i] if i < len(stop_reasons_for_stats) else None
        fr_norm = str(fr).lower() if fr is not None else ""
        sr_norm = str(sr).lower() if sr is not None else ""
        if err:
            finish_reason_counts["error"] += 1
        elif fr_norm == "length":
            finish_reason_counts["length"] += 1
        elif fr_norm == "error":
            finish_reason_counts["error"] += 1
        elif fr_norm == "abort":
            finish_reason_counts["abort"] += 1
        elif fr_norm in ("eos",):
            finish_reason_counts["eos"] += 1
        elif "eos" in sr_norm:
            finish_reason_counts["eos"] += 1
        elif fr_norm == "stop":
            finish_reason_counts["stop"] += 1
        else:
            finish_reason_counts["unknown"] += 1
    result["finish_reason_counts"] = finish_reason_counts

    result["weighted_tpot_ms"] = _weighted_tpot_ms(
        output_lens_for_stats,
        ttfts_for_stats,
        e2els_for_stats,
        min_output_len=2,
    )
    result["weighted_tpot_ms_out>=200"] = _weighted_tpot_ms(
        output_lens_for_stats,
        ttfts_for_stats,
        e2els_for_stats,
        min_output_len=200,
    )

    kv_budget = None
    kv_budget_env = os.getenv("KV_CACHE_BUDGET")
    if kv_budget_env is not None and str(kv_budget_env).strip():
        try:
            kv_budget = int(kv_budget_env)
        except ValueError:
            kv_budget = None

    if kv_budget is not None and kv_budget > 0:
        le_budget_indices = [
            i for i, x in enumerate(input_lens_for_stats) if int(x) <= kv_budget
        ]
        gt_budget_indices = [
            i for i, x in enumerate(input_lens_for_stats) if int(x) > kv_budget
        ]

        def _bucket_stats(indices: List[int]) -> Dict[str, Optional[float]]:
            out_vals = [output_lens_for_stats[i] for i in indices if i < len(output_lens_for_stats)]
            ttft_vals = [
                ttfts_for_stats[i] * 1000.0
                for i in indices
                if i < len(ttfts_for_stats)
            ]
            e2e_vals = [
                e2els_for_stats[i] * 1000.0
                for i in indices
                if i < len(e2els_for_stats)
            ]
            tpot_vals = []
            for i in indices:
                if i >= len(output_lens_for_stats) or i >= len(ttfts_for_stats) or i >= len(e2els_for_stats):
                    continue
                out_len = output_lens_for_stats[i]
                if out_len < 2:
                    continue
                tpot_vals.append(
                    max(e2els_for_stats[i] - ttfts_for_stats[i], 0.0) / (out_len - 1) * 1000.0
                )
            return {
                "num_requests": len(indices),
                "output_len_p50": _safe_percentile(out_vals, 50),
                "output_len_p90": _safe_percentile(out_vals, 90),
                "output_len_p99": _safe_percentile(out_vals, 99),
                "ttft_ms_p50": _safe_percentile(ttft_vals, 50),
                "ttft_ms_p90": _safe_percentile(ttft_vals, 90),
                "ttft_ms_p99": _safe_percentile(ttft_vals, 99),
                "e2e_ms_p50": _safe_percentile(e2e_vals, 50),
                "e2e_ms_p90": _safe_percentile(e2e_vals, 90),
                "e2e_ms_p99": _safe_percentile(e2e_vals, 99),
                "tpot_ms_p50": _safe_percentile(tpot_vals, 50),
                "tpot_ms_p90": _safe_percentile(tpot_vals, 90),
                "tpot_ms_p99": _safe_percentile(tpot_vals, 99),
                "weighted_tpot_ms": _weighted_tpot_ms(
                    output_lens_for_stats,
                    ttfts_for_stats,
                    e2els_for_stats,
                    min_output_len=2,
                    selected_indices=indices,
                ),
                "weighted_tpot_ms_out>=200": _weighted_tpot_ms(
                    output_lens_for_stats,
                    ttfts_for_stats,
                    e2els_for_stats,
                    min_output_len=200,
                    selected_indices=indices,
                ),
            }

        result["prompt_len_bucket_stats"] = {
            "budget": kv_budget,
            "prompt_len_le_budget": _bucket_stats(le_budget_indices),
            "prompt_len_gt_budget": _bucket_stats(gt_budget_indices),
        }

    if gpu_monitors is not None and result.get("gpu_power_stats"):
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0
        per_gpu_peak_allocated_mb = {}
        for gpu_id, gpu_stats in result["gpu_power_stats"].items():
            used_trace = gpu_stats.get("memory_used_mb_trace", [])
            peak_gpu_used_mb = float(max(used_trace)) if used_trace else 0.0
            per_gpu_peak_allocated_mb[gpu_id] = peak_gpu_used_mb
            peak_allocated_mb += peak_gpu_used_mb
            # NVML does not expose torch reserved bytes; use allocated approximation.
            peak_reserved_mb += peak_gpu_used_mb
        oom_count = sum(
            1
            for err in errors_for_stats
            if isinstance(err, str) and err and re.search(r"oom|out of memory", err, re.IGNORECASE)
        )
        result["memory_stats"] = {
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "per_gpu_peak_allocated_mb": per_gpu_peak_allocated_mb,
            "oom_count": oom_count,
            "engine_restart_count": 0,
        }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function print and add statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


async def get_server_max_model_len(base_url: str) -> Optional[int]:
    """
    Fetch the actual max_model_len from vLLM server's /v1/models endpoint.

    Args:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000")

    Returns:
        The max_model_len if available, None otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/v1/models"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("data") and len(data["data"]) > 0:
                        max_len = data["data"][0].get("max_model_len")
                        if max_len is not None:
                            return int(max_len)
    except Exception as e:
        print(f"Warning: Failed to fetch max_model_len from server: {e}")

    return None


async def get_kv_cache_stats(base_url: str, enable_trace: bool = False, sample_count: int = 1) -> Dict[str, Any]:
    """
    Fetch KV cache statistics from vLLM server's /metrics endpoint.
    
    Args:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000")
        enable_trace: If True, collect dynamic trace data
        sample_count: Number of samples to collect (for averaging dynamic stats)
    
    Returns:
        Dictionary containing KV cache statistics (both static config and dynamic stats)
    """
    kv_stats = {
        'static_config': {},  # Fixed configuration values
        'dynamic_stats': {},  # Real-time statistics
        'dynamic_samples': [],  # All samples collected
    }
    
    metrics_url = f"{base_url}/metrics"

    def _sum_counter_values(metrics_text: str, metric_name: str) -> float:
        pattern = re.compile(
            rf"^{re.escape(metric_name)}_total(?:\{{[^}}]*\}})?\s+([0-9eE+.\-]+)$",
            re.MULTILINE,
        )
        return float(sum(float(m.group(1)) for m in pattern.finditer(metrics_text)))
    
    try:
        async with aiohttp.ClientSession() as session:
            # Collect multiple samples for better dynamic stats
            for sample_idx in range(sample_count):
                if sample_idx > 0:
                    await asyncio.sleep(0.5)  # Wait 500ms between samples
                
                async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        metrics_text = await response.text()
                        
                        # Parse Prometheus-format metrics for static configuration
                        # Note: All configs are in cache_config_info labels, not separate metrics
                        static_patterns = {
                            'total_gpu_blocks': r'vllm:cache_config_info\{[^}]*num_gpu_blocks="(\d+)"',
                            'block_size': r'vllm:cache_config_info\{[^}]*block_size="(\d+)"',
                            'num_layers': r'vllm:cache_config_info\{[^}]*num_layers="(\d+)"',
                            'num_heads': r'vllm:cache_config_info\{[^}]*num_kv_heads="(\d+)"',
                            'head_size': r'vllm:cache_config_info\{[^}]*head_size="(\d+)"',
                        }
                        
                        # Parse dynamic statistics patterns
                        # Note: vLLM uses 'kv_cache_usage_perc' not 'gpu_cache_usage_perc'
                        dynamic_patterns = {
                            'cache_usage_perc': r'vllm:kv_cache_usage_perc\{[^}]*\}\s+(\d+\.?\d*)',
                            'num_requests_running': r'vllm:num_requests_running\{[^}]*\}\s+(\d+\.?\d*)',
                            'num_requests_waiting': r'vllm:num_requests_waiting\{[^}]*\}\s+(\d+\.?\d*)',
                        }
                        
                        # Extract static configuration (only once)
                        if not kv_stats['static_config']:
                            for key, pattern in static_patterns.items():
                                match = re.search(pattern, metrics_text)
                                if match:
                                    value = match.group(1)
                                    # Convert to int for numeric values
                                    try:
                                        kv_stats['static_config'][key] = int(value)
                                    except ValueError:
                                        kv_stats['static_config'][key] = value
                        
                        # Extract dynamic statistics for this sample
                        sample_stats = {}
                        for key, pattern in dynamic_patterns.items():
                            match = re.search(pattern, metrics_text)
                            if match:
                                sample_stats[key] = float(match.group(1))

                        sample_stats['page_eviction_ops_total'] = _sum_counter_values(
                            metrics_text, 'vllm:page_eviction_num_eviction_ops'
                        )
                        sample_stats['page_eviction_blocks_total'] = _sum_counter_values(
                            metrics_text, 'vllm:page_eviction_num_evicted_blocks'
                        )
                        sample_stats['spec_decode_draft_tokens_total'] = _sum_counter_values(
                            metrics_text, 'vllm:spec_decode_num_draft_tokens'
                        )
                        sample_stats['spec_decode_accepted_tokens_total'] = _sum_counter_values(
                            metrics_text, 'vllm:spec_decode_num_accepted_tokens'
                        )
                        
                        if sample_stats:
                            kv_stats['dynamic_samples'].append(sample_stats)
                    
            # Calculate averaged/aggregated dynamic stats from all samples
            if kv_stats['dynamic_samples']:
                dynamic_stats = kv_stats['dynamic_stats']
                samples = kv_stats['dynamic_samples']
                
                # Average the dynamic metrics
                for key in ['cache_usage_perc', 'num_requests_running', 'num_requests_waiting']:
                    values = [s.get(key, 0) for s in samples]
                    if values:
                        dynamic_stats[f'{key}_avg'] = round(sum(values) / len(values), 2)
                        dynamic_stats[f'{key}_max'] = round(max(values), 2)
                        dynamic_stats[f'{key}_min'] = round(min(values), 2)
                
                static_cfg = kv_stats['static_config']
                
                # Calculate tokens per block based on block_size
                if 'block_size' in static_cfg:
                    static_cfg['tokens_per_block'] = static_cfg['block_size']
                
                # Calculate total KV cache capacity
                if 'total_gpu_blocks' in static_cfg and 'tokens_per_block' in static_cfg:
                    total_tokens = static_cfg['total_gpu_blocks'] * static_cfg['tokens_per_block']
                    static_cfg['total_kv_cache_tokens'] = total_tokens
                
                # Calculate used and free blocks from averaged cache usage percentage
                if 'total_gpu_blocks' in static_cfg and 'cache_usage_perc_avg' in dynamic_stats:
                    total_blocks = static_cfg['total_gpu_blocks']
                    usage_percent = dynamic_stats['cache_usage_perc_avg']
                    used_blocks = int(total_blocks * usage_percent / 100)
                    free_blocks = total_blocks - used_blocks
                    
                    dynamic_stats['used_gpu_blocks_avg'] = used_blocks
                    dynamic_stats['free_gpu_blocks_avg'] = free_blocks
                    dynamic_stats['kv_cache_usage_percentage'] = round(usage_percent, 2)
                    
                    # Calculate used tokens
                    if 'tokens_per_block' in static_cfg:
                        dynamic_stats['used_kv_cache_tokens_avg'] = used_blocks * static_cfg['tokens_per_block']
                
                # Also calculate peak usage
                if 'total_gpu_blocks' in static_cfg and 'cache_usage_perc_max' in dynamic_stats:
                    total_blocks = static_cfg['total_gpu_blocks']
                    peak_usage_percent = dynamic_stats['cache_usage_perc_max']
                    peak_used_blocks = int(total_blocks * peak_usage_percent / 100)
                    
                    dynamic_stats['used_gpu_blocks_peak'] = peak_used_blocks
                    dynamic_stats['free_gpu_blocks_min'] = total_blocks - peak_used_blocks
                    
                    if 'tokens_per_block' in static_cfg:
                        dynamic_stats['used_kv_cache_tokens_peak'] = peak_used_blocks * static_cfg['tokens_per_block']
                
                # Add trace information if enabled
                if enable_trace:
                    kv_stats['trace_info'] = {
                        'timestamp': datetime.now().isoformat(),
                        'collection_method': 'prometheus_metrics',
                        'metrics_endpoint': metrics_url,
                        'samples_collected': len(kv_stats['dynamic_samples']),
                    }

                # Counter deltas from first to last sample.
                if len(samples) >= 2:
                    ops_delta = max(
                        0.0,
                        samples[-1].get('page_eviction_ops_total', 0.0)
                        - samples[0].get('page_eviction_ops_total', 0.0),
                    )
                    blocks_delta = max(
                        0.0,
                        samples[-1].get('page_eviction_blocks_total', 0.0)
                        - samples[0].get('page_eviction_blocks_total', 0.0),
                    )
                    spec_draft_delta = max(
                        0.0,
                        samples[-1].get('spec_decode_draft_tokens_total', 0.0)
                        - samples[0].get('spec_decode_draft_tokens_total', 0.0),
                    )
                    spec_accepted_delta = max(
                        0.0,
                        samples[-1].get('spec_decode_accepted_tokens_total', 0.0)
                        - samples[0].get('spec_decode_accepted_tokens_total', 0.0),
                    )
                else:
                    ops_delta = 0.0
                    blocks_delta = 0.0
                    spec_draft_delta = 0.0
                    spec_accepted_delta = 0.0

                dynamic_stats['page_eviction_ops_total_delta'] = ops_delta
                dynamic_stats['page_eviction_blocks_total_delta'] = blocks_delta
                dynamic_stats['spec_decode_draft_tokens_total_delta'] = spec_draft_delta
                dynamic_stats['spec_decode_accepted_tokens_total_delta'] = spec_accepted_delta
                dynamic_stats['spec_decode_acceptance_rate'] = (
                    (spec_accepted_delta / spec_draft_delta)
                    if spec_draft_delta > 0
                    else 0.0
                )
                    
                print(f"\n✓ Successfully fetched KV cache statistics from server ({len(samples)} samples)")
            else:
                print(f"\n⚠ Warning: No dynamic KV cache metrics found in server response")
                
    except asyncio.TimeoutError:
        print(f"\n⚠ Warning: Timeout while fetching metrics from {metrics_url}")
    except Exception as e:
        print(f"\n⚠ Warning: Error fetching KV cache statistics: {e}")
    
    # Remove raw samples from output to keep JSON clean (optional)
    if not enable_trace:
        kv_stats.pop('dynamic_samples', None)
    
    return kv_stats


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    print(f"The tokenizer id is {tokenizer_id}, the model id is {model_id}")
    if tokenizer_id == "TheBloke/Llama-2-7B-Chat-AWQ":
        tokenizer_id = "meta-llama/Llama-2-7b-chat-hf"
    elif "Llama-2-13B-chat-AWQ" in tokenizer_id:
        tokenizer_id = "meta-llama/Llama-2-13b-chat-hf"
    print(f"The tokenizer id is {tokenizer_id}, the model id is {model_id}")
    args.num_prompts = max(int(args.num_prompts),2)

    if args.gpu_ids is not None:
        gpu_monitors = []
        for gpu_id in args.gpu_ids:
            gpu_monitors.append(GPUMonitor(gpu_id=gpu_id, interval=args.gpu_monitor_interval, truncate=args.gpu_monitor_truncate))
    else:
        gpu_monitors = None

    if args.monitor_cpu:
        cpu_monitor = CPUMonitor(interval=args.gpu_monitor_interval, truncate=args.gpu_monitor_truncate)
    else:
        cpu_monitor = None
    
    # Initialize KV cache monitor with appropriate interval
    # Note: KV cache metrics update slower than GPU/CPU power, so we use a longer interval
    if backend == "vllm" and args.enable_kv_trace:
        base_url = f"http://{args.host}:{args.port}"
        # Use 0.5s interval for KV cache (slower than GPU/CPU but still captures dynamics)
        kv_interval = max(0.5, args.gpu_monitor_interval * 10)  # At least 0.5s, or 10x GPU interval
        kv_cache_monitor = KVCacheMonitor(
            base_url=base_url, 
            interval=kv_interval,
            truncate=args.gpu_monitor_truncate
        )
        print(
            f"✓ KV/PageEviction/Spec Monitor initialized: interval={kv_interval}s "
            f"(GPU/CPU: {args.gpu_monitor_interval}s)"
        )
        print(
            "  Note: cache/eviction/spec metrics use a longer interval than power readings"
        )
    else:
        kv_cache_monitor = None

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"



    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    # Fetch max_model_len from server for dynamic context length management
    server_max_model_len = asyncio.run(get_server_max_model_len(base_url))
    if server_max_model_len is not None:
        print(f"✓ Fetched max_model_len from server: {server_max_model_len}")
    else:
        print("⚠ Could not fetch max_model_len from server, using tokenizer defaults")

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2)
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.sharegpt_input_len,
            fixed_output_len=args.sharegpt_output_len,
            min_prompt_len=args.min_prompt_len,
            server_max_model_len=server_max_model_len,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.sharegpt_input_len,
            fixed_output_len=args.sharegpt_output_len,
            min_prompt_len=args.min_prompt_len,
            server_max_model_len=server_max_model_len,
        )

    elif args.dataset_name == "arenahard":
        input_requests = sample_arenahard_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.arenahard_input_len,
            fixed_output_len=args.arenahard_output_len,
        )


    elif args.dataset_name == "alpaca":
        input_requests = sample_alpaca_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.alpaca_input_len,
            fixed_output_len=args.alpaca_output_len,
        )

    elif args.dataset_name == "lmsyschat":
        input_requests = sample_lmsyschat_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.lmsyschat_input_len,
            fixed_output_len=args.lmsyschat_output_len,
        )

    elif args.dataset_name == "summeval":
        input_requests = sample_summeval_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.summeval_input_len,
            fixed_output_len=args.summeval_output_len,
        )

    elif args.dataset_name == "newsqa":
        input_requests = sample_newsqa_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.newsqa_input_len,
            fixed_output_len=args.newsqa_output_len,
        )

    elif args.dataset_name == "gsm8k":
        input_requests = sample_gsm8k_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.gsm8k_input_len,
            fixed_output_len=args.gsm8k_output_len,
        )

    elif args.dataset_name == "mbpp":
        input_requests = sample_mbpp_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.mbpp_input_len,
            fixed_output_len=args.mbpp_output_len,
        )
    
    
    elif args.dataset_name == "longbench":
        input_requests = sample_longbench_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.longbench_input_len,
            fixed_output_len=args.longbench_output_len,
        )

    elif args.dataset_name == "humaneval":
        input_requests = sample_humaneval_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_input_len=args.humaneval_input_len,
            fixed_output_len=args.humaneval_output_len,
        )

    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt, prompt_len, output_len, None)
                              for prompt, prompt_formatted, prompt_len,
                              output_len, _ in input_requests]
        else:
            assert (
                tokenizer.chat_template or tokenizer.default_chat_template
            ), "Tokenizer/model must have chat template for sonnet dataset."
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt_formatted, prompt_len, output_len, None)
                              for prompt, prompt_formatted, prompt_len,
                              output_len, _ in input_requests]

    elif args.dataset_name == "hf":
        input_requests = sample_hf_requests(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.hf_output_len,
        )

    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    carbon_params = {
        "carbon_intensity": args.carbon_intensity,
        "embodied_carbon": args.embodied_carbon,
        "device_lifetime": args.device_lifetime,
    }

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            gpu_monitors=gpu_monitors,
            cpu_monitor=cpu_monitor,
            kv_cache_monitor=kv_cache_monitor,
            warmup_ratio=args.warmup_ratio,
            carbon_params=carbon_params,
            server_max_model_len=server_max_model_len,
        ))

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Fetch KV cache statistics from server
        if backend == "vllm":
            print("\nFetching KV cache statistics from server...")

            # Check if we already have KV cache monitoring data from benchmark run
            kv_monitoring_stats = result_json.get("kv_cache_monitoring_stats")

            if kv_monitoring_stats and kv_monitoring_stats.get("samples_collected", 0) > 0:
                # Use the real-time monitoring data collected during benchmark
                print(f"  Using real-time monitoring data ({kv_monitoring_stats['samples_collected']} samples)")

                detailed = kv_monitoring_stats.get("detailed_stats", {})
                avg_stats = kv_monitoring_stats.get("avg_stats", {})
                static_cfg = detailed.get("static_config", {})

                # Build kv_cache_stats from monitoring data
                kv_cache_stats = {
                    "static_config": static_cfg,
                    "dynamic_stats": {
                        # Cache usage stats
                        "cache_usage_perc_avg": avg_stats.get("avg_cache_usage_perc", 0),
                        "cache_usage_perc_max": detailed.get("cache_usage", {}).get("max", 0),
                        "cache_usage_perc_min": detailed.get("cache_usage", {}).get("min", 0),
                        # Request stats
                        "num_requests_running_avg": avg_stats.get("avg_requests_running", 0),
                        "num_requests_running_max": detailed.get("requests_running", {}).get("max", 0),
                        "num_requests_running_min": detailed.get("requests_running", {}).get("min", 0),
                        "num_requests_waiting_avg": avg_stats.get("avg_requests_waiting", 0),
                        "num_requests_waiting_max": detailed.get("requests_waiting", {}).get("max", 0),
                        "num_requests_waiting_min": detailed.get("requests_waiting", {}).get("min", 0),
                        # GPU blocks stats
                        "used_gpu_blocks_avg": avg_stats.get("avg_used_blocks", 0),
                        "free_gpu_blocks_avg": avg_stats.get("avg_free_blocks", 0),
                        "used_kv_cache_tokens_avg": avg_stats.get("avg_used_tokens", 0),
                        # Peak values
                        "used_gpu_blocks_peak": detailed.get("used_blocks", {}).get("max", 0),
                        "free_gpu_blocks_min": detailed.get("used_blocks", {}).get("min", 0),
                        "used_kv_cache_tokens_peak": detailed.get("used_tokens", {}).get("max", 0),
                        # Combined usage percentage
                        "kv_cache_usage_percentage": avg_stats.get("avg_cache_usage_perc", 0),
                        # PageEviction counters
                        "page_eviction_ops_total_delta": avg_stats.get(
                            "total_page_eviction_ops", 0
                        ),
                        "page_eviction_blocks_total_delta": avg_stats.get(
                            "total_page_eviction_blocks", 0
                        ),
                        "page_eviction_ops_avg_per_sample": avg_stats.get(
                            "avg_page_eviction_ops_per_sample", 0
                        ),
                        "page_eviction_blocks_avg_per_sample": avg_stats.get(
                            "avg_page_eviction_blocks_per_sample", 0
                        ),
                        # Spec decode counters
                        "spec_decode_draft_tokens_total_delta": avg_stats.get(
                            "total_spec_draft_tokens", 0
                        ),
                        "spec_decode_accepted_tokens_total_delta": avg_stats.get(
                            "total_spec_accepted_tokens", 0
                        ),
                        "spec_decode_acceptance_rate": avg_stats.get(
                            "spec_decode_acceptance_rate", 0
                        ),
                    },
                    "dynamic_samples": [],
                    "trace_info": {
                        "timestamp": datetime.now().isoformat(),
                        "collection_method": "prometheus_metrics_realtime",
                        "samples_collected": kv_monitoring_stats.get("samples_collected", 0),
                    },
                }
                result_json["kv_cache_stats"] = kv_cache_stats

                # Print static configuration
                if static_cfg:
                    print("\n" + "=" * 50)
                    print("KV Cache Static Configuration:")
                    print("=" * 50)
                    if 'total_gpu_blocks' in static_cfg:
                        print(f"  Total GPU Blocks: {static_cfg['total_gpu_blocks']}")
                    if 'block_size' in static_cfg:
                        print(f"  Block Size: {static_cfg['block_size']}")
                    if 'tokens_per_block' in static_cfg:
                        print(f"  Tokens per Block: {static_cfg['tokens_per_block']}")
                    if 'total_kv_cache_tokens' in static_cfg:
                        print(f"  Total KV Cache Tokens: {static_cfg['total_kv_cache_tokens']:,}")

                # Print dynamic statistics from real-time monitoring
                dynamic_stats = kv_cache_stats.get("dynamic_stats", {})
                if dynamic_stats:
                    print("\n" + "=" * 50)
                    print("KV Cache Dynamic Statistics (Real-time Monitoring):")
                    print("=" * 50)
                    if 'kv_cache_usage_percentage' in dynamic_stats:
                        print(f"  Avg Cache Usage: {dynamic_stats['kv_cache_usage_percentage']:.2f}%")
                    if 'used_kv_cache_tokens_avg' in dynamic_stats:
                        print(f"  Avg Used Tokens: {dynamic_stats['used_kv_cache_tokens_avg']:,.0f}")
                    if 'used_kv_cache_tokens_peak' in dynamic_stats:
                        print(f"  Peak Used Tokens: {dynamic_stats['used_kv_cache_tokens_peak']:,}")
                    if 'num_requests_running_avg' in dynamic_stats:
                        print(f"  Avg Running Requests: {dynamic_stats['num_requests_running_avg']:.1f}")
                    if 'num_requests_running_max' in dynamic_stats:
                        print(f"  Max Running Requests: {int(dynamic_stats['num_requests_running_max'])}")
                    if dynamic_stats.get("page_eviction_ops_total_delta", 0) > 0:
                        print(
                            "  PageEviction Ops: "
                            f"{dynamic_stats['page_eviction_ops_total_delta']:,.0f}"
                        )
                        print(
                            "  PageEviction Evicted Blocks: "
                            f"{dynamic_stats['page_eviction_blocks_total_delta']:,.0f}"
                        )
                    if dynamic_stats.get("spec_decode_draft_tokens_total_delta", 0) > 0:
                        print(
                            "  Spec Draft Tokens: "
                            f"{dynamic_stats['spec_decode_draft_tokens_total_delta']:,.0f}"
                        )
                        print(
                            "  Spec Accepted Tokens: "
                            f"{dynamic_stats['spec_decode_accepted_tokens_total_delta']:,.0f}"
                        )
                        print(
                            "  Spec Acceptance Rate: "
                            f"{dynamic_stats['spec_decode_acceptance_rate']:.4f}"
                        )

                print("=" * 50)
            else:
                # Fallback: collect KV cache stats after benchmark (may be empty if benchmark just finished)
                enable_trace = args.enable_kv_trace if hasattr(args, 'enable_kv_trace') else False
                kv_cache_stats = asyncio.run(get_kv_cache_stats(base_url, enable_trace=enable_trace, sample_count=5))
                if kv_cache_stats and (kv_cache_stats.get('static_config') or kv_cache_stats.get('dynamic_stats')):
                    result_json["kv_cache_stats"] = kv_cache_stats
                    # ... (rest of existing code for printing static/dynamic config)

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile, indent=2)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "sonnet", "random", "hf", "longbench", "humaneval", "summeval", "newsqa", "gsm8k", "mbpp", "alpaca", "lmsyschat", "arenahard"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=float,
        default=1000.0,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    sharegpt_group.add_argument(
        "--sharegpt-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    sharegpt_group.add_argument(
        "--min-prompt-len",
        type=int,
        default=None,
        help="最小 prompt 长度（tokens）。只有 prompt 长度大于等于此值的请求才会被发送。"
        "用于筛选长请求进行测试。")
    
    arenahard_group = parser.add_argument_group("arenahard dataset options")
    arenahard_group.add_argument(
        "--arenahard-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the arenahard dataset.")
    arenahard_group.add_argument(
        "--arenahard-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the arenahard dataset.")
    
    alpaca_group = parser.add_argument_group("alpaca dataset options")
    alpaca_group.add_argument(
        "--alpaca-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the alpaca dataset.")
    alpaca_group.add_argument(
        "--alpaca-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the alpaca dataset.")
    
    lmsyschat_group = parser.add_argument_group("lmsyschat dataset options")
    lmsyschat_group.add_argument(
        "--lmsyschat-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the lmsyschat dataset.")
    lmsyschat_group.add_argument(
        "--lmsyschat-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the lmsyschat dataset.")
    
    
    summeval_group = parser.add_argument_group("summeval dataset options")
    summeval_group.add_argument(
        "--summeval-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the summeval dataset.")
    summeval_group.add_argument(
        "--summeval-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the summeval dataset.")
    
    mbpp_group = parser.add_argument_group("mbpp dataset options")
    mbpp_group.add_argument(
        "--mbpp-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the mbpp dataset.")
    mbpp_group.add_argument(
        "--mbpp-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the mbpp dataset.")
    
    newsqa_group = parser.add_argument_group("newsqa dataset options")
    newsqa_group.add_argument(
        "--newsqa-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the newsqa dataset.")
    newsqa_group.add_argument(
        "--newsqa-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the newsqa dataset.")

    gsm8k_group = parser.add_argument_group("gsm8k dataset options")
    gsm8k_group.add_argument(
        "--gsm8k-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the gsm8k dataset.")
    gsm8k_group.add_argument(
        "--gsm8k-input-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the gsm8k dataset.")
    
    longbench_group = parser.add_argument_group("longbench dataset options")
    longbench_group.add_argument(
        "--longbench-input-len",
        type=int,
        default=None,
        help=
        "Number of input tokens per request, overrides the input lengths "
        "from the sampled longbench dataset.",
    )
    longbench_group.add_argument(
        "--longbench-output-len",
        type=int,
        default=None,
        help=
        "Number of output tokens per request, overrides the output lengths "
        "from the sampled longbench dataset.",
    )

    humaneval_group = parser.add_argument_group("humaneval dataset options")
    humaneval_group.add_argument(
        "--humaneval-input-len",
        type=int,
        default=None,
        help=
        "Number of input tokens per request, used only for humaneval dataset.",
    )
    humaneval_group.add_argument(
        "--humaneval-output-len",
        type=int,
        default=None,
        help=
        "Number of output tokens per request, used only for humaneval dataset.",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="GPU ids to monitor power consumption.",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=0.025,
        help="Interval to monitor GPU power consumption.",
    )
    parser.add_argument(
        "--gpu-monitor-truncate",
        type=float,
        default=0,
        help="Truncate the first and last `truncate` seconds of the monitoring data.",
    )
    parser.add_argument(
        "--monitor-cpu",
        action="store_true",
        help="Monitor the CPU util, frequency, disk io, and mem util",
    )

    parser.add_argument(
        "--use-deterministic-rate",
        action="store_true",
        help="Use deterministic rate instead of exponential distribution.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Ratio of the additional warm up requests to the total number of requests.",
    )
    parser.add_argument(
        "--carbon-intensity",
        type=float,
        default=530,
        help="Carbon intensity in gCO2e/kWh. Default value is 530 gCO2e/kWh from MISO 2023 on electricitymap.",
    )
    parser.add_argument(
        "--embodied-carbon",
        type=float,
        default=26.6,
        help="Embodied carbon in kgCO2e. Default value is 26.6 kgCO2e of L40.",
    )
    parser.add_argument(
        "--device-lifetime",
        type=float,
        default=7,
        help="Device lifetime in years.",
    )
    parser.add_argument(
        "--enable-kv-trace",
        action="store_true",
        help="Enable KV cache trace recording to track dynamic block allocation. "
        "This will add additional trace information to the output JSON.",
    )

    args = parser.parse_args()
    main(args)
