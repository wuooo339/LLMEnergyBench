import argparse
from typing import List, Tuple
import math
from tqdm import tqdm
import torch
import gc

from vllm import LLM, SamplingParams
from datasets import load_dataset

def get_sliding_windows(
    tokenizer, text: str, max_length: int, stride: int
) -> List[Tuple[List[int], int, int]]:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    seq_len = input_ids.size(0)

    windows = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        windows.append((
            input_ids[begin_loc:end_loc].tolist(),
            prev_end_loc - begin_loc,
            trg_len
        ))
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return windows

def process_batch(
    llm: LLM,
    batch: List[Tuple[List[int], int, int]],
    sampling_params: SamplingParams
) -> float:
    input_ids = [window[0] for window in batch]
    outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)

    batch_nll = 0.0
    batch_tokens = 0

    for output, (tokens, ctxlen, trg_len) in zip(outputs, batch):
        continuation_logprobs, _ = _parse_logprobs(tokens, output, ctxlen)
        batch_nll -= continuation_logprobs
        batch_tokens += trg_len

    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()

    return batch_nll, batch_tokens

def _parse_logprobs(tokens: List[int], output, ctxlen: int) -> Tuple[float, bool]:
    continuation_logprobs_dicts = output.prompt_logprobs
    def coerce_logprob_to_num(logprob):
        return getattr(logprob, "logprob", logprob)

    continuation_logprobs_dicts = [
        {
            token: coerce_logprob_to_num(logprob)
            for token, logprob in logprob_dict.items()
        }
        if logprob_dict is not None
        else None
        for logprob_dict in continuation_logprobs_dicts
    ]

    ctxlen = max(1, ctxlen)
    continuation_logprobs = sum(
        logprob_dict.get(token, float('-inf'))
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        )
    )

    is_greedy = all(
        token == max(logprob_dict, key=logprob_dict.get)
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        )
        if logprob_dict
    )

    return continuation_logprobs, is_greedy

def calculate_perplexity(llm: LLM, windows: List[Tuple[List[int], int, int]], batch_size: int = 1) -> float:
    total_nll = 0.0
    total_tokens = 0

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    for i in tqdm(range(0, len(windows), batch_size), desc="Processing batches"):
        batch = windows[i:i + batch_size]
        batch_nll, batch_tokens = process_batch(llm, batch, sampling_params)
        
        total_nll += batch_nll
        total_tokens += batch_tokens

    perplexity = math.exp(total_nll / total_tokens)
    return perplexity

def main(args):
    # Load the model
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True, gpu_memory_utilization=0.8)
    tokenizer = llm.get_tokenizer()

    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(dataset["text"])

    # Get the model's maximum sequence length
    max_length = llm.llm_engine.model_config.max_model_len
    max_length = min(4096, max_length)
    # Create sliding windows
    windows = get_sliding_windows(tokenizer, text, max_length, args.stride)

    # Calculate perplexity
    perplexity = calculate_perplexity(llm, windows, args.batch_size)
    print(f"Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate perplexity on WikiText-2 using vLLM")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    main(args)