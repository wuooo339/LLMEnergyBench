"""Benchmark the latency of processing a single batch of requests."""
import argparse
import json
import random
import threading
import time
import queue
import gc
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import DEVICE_OPTIONS, EngineArgs
from vllm.inputs import PromptInputs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser

# nvml method or zeus-ml method
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetUtilizationRates, NVMLError 
from zeus.monitor import ZeusMonitor
import pandas as pd

ALCAPA_DATA_PATH = '/local/scratch/a/shi676/llm_profiling/llm-decarbonization/llm-sys-main/splitwise/datasets/alpaca_data.json'

class GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.025):
        self.gpu_id = gpu_id
        self.done = False
        self.results_queue = queue.Queue()
        self.interval = interval
        self.thread = None
        
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(gpu_id)
        except NVMLError as e:
            print(f"NVML Error: {e}")
            self.gpu_handle = None

    def start(self):
        if self.gpu_handle is None:
            print("GPU handle not initialized. Monitoring cannot start.")
            return
        self.done = False
        self.thread = threading.Thread(target=self._monitor_gpu)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.done = True
            self.thread.join()

    def _monitor_gpu(self):
        gpu_power_readings = []
        gpu_utilization_readings = []
        memory_utilization_readings = []
        
        while not self.done:
            power = nvmlDeviceGetPowerUsage(self.gpu_handle)
            utilization = nvmlDeviceGetUtilizationRates(self.gpu_handle)

            gpu_power_readings.append(power)
            gpu_utilization_readings.append(utilization.gpu)
            memory_utilization_readings.append(utilization.memory)

            time.sleep(self.interval)

        avg_power = sum(gpu_power_readings) / len(gpu_power_readings) if gpu_power_readings else 0
        avg_gpu_util = sum(gpu_utilization_readings) / len(gpu_utilization_readings) if gpu_utilization_readings else 0
        avg_mem_util = sum(memory_utilization_readings) / len(memory_utilization_readings) if memory_utilization_readings else 0

        self.results_queue.put((avg_power, avg_gpu_util, avg_mem_util))

    def __del__(self):
        if self.gpu_handle:
            self.stop()
            nvmlShutdown()

class MyParser:
    # Parse Alpaca file
    def loadJson(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        instructions = [f"{item['instruction']}\n{item['input']}" for item in data]
        return instructions

    # Parse GSM8K files
    def loadJsonl(self, file_path):
        with open(file_path, 'r') as file:
            data_lines = file.readlines()
        questions = [json.loads(line)['question'] for line in data_lines]
        return questions
    

class BatchPrompt:
    def __init__(self, prompts):
        self.prompts = prompts  # List of input prompts
        self.generated_texts = []  # List to store generated outputs
        self.time_first_token_generation = []
        self.time_rest_token_generation = None
        self.prefill_avg_power = None
        self.prefill_avg_gpu_util = None
        self.prefill_avg_mem_util = None
        self.tokenization_avg_power = None
        self.tokenization_avg_gpu_util = None
        self.tokenization_avg_mem_util = None
        
    def add_generated_text(self, text):
        self.generated_texts.append(text)
        
    def update_token_stats(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = [input_tokens[i] + output_tokens[i] for i in range(len(input_tokens))]  
            
    def printGeneratedResult(self):
        print("\nGenerated Texts:")
        for i, text in enumerate(self.generated_texts):
            print(f"Prompt {i+1}: {self.prompts[i]}")
            print(f"Generated Text {i+1}: {text}\n")


class MetricsTracker:
    def __init__(self, model_name):
        # Initialize a DataFrame with specific metrics as columns
        self.model_name = model_name
        self.metrics_df = pd.DataFrame(columns=[
            'Batch Size', 
            'Prompt Token #', 
            'Output Token #', 
            'TTFT', 
            'Prefill GPU Power', 
            'Prefill GPU Util', 
            'Prefill Memory Util', 
            'Total Decode Time (s)', 
            'TPOT', 
            'Decode GPU Power', 
            'Decode GPU Util', 
            'Decode Memory Util', 
            'Total Time'
        ])
        
    def insert_latency_metrics(self, batchPrompt):
        batch_size = len(batchPrompt.prompts)
        total_input_tokens = sum(batchPrompt.input_tokens)
        total_output_tokens = sum(batchPrompt.output_tokens)
        
        ttft = self._find_median(batchPrompt.time_first_token_generation)
        
        
        new_row = {
            'Batch Size': batch_size,
            'Prompt Token #': total_input_tokens, 
            'Output Token #': total_output_tokens, 
        }

        if batchPrompt.time_first_token_generation is not None:
            new_row['TTFT'] = ttft
        
        if batchPrompt.prefill_avg_power is not None:
            new_row['Prefill GPU Power'] = batchPrompt.prefill_avg_power
            
        if batchPrompt.prefill_avg_gpu_util is not None:
            new_row['Prefill GPU Util'] = batchPrompt.prefill_avg_gpu_util
            
        if batchPrompt.prefill_avg_mem_util is not None:
            new_row['Prefill Memory Util'] = batchPrompt.prefill_avg_mem_util

        if batchPrompt.time_rest_token_generation is not None and total_output_tokens > 0:
            trtg = self._find_median(batchPrompt.time_rest_token_generation)
            batchPrompt.time_per_output_token = [batchPrompt.time_rest_token_generation[i] / (batchPrompt.output_tokens[i]-1) for i in range(batch_size)]
            new_row['Total Decode Time (s)'] = trtg
            # new_row['TBT'] = trtg / total_output_tokens
            new_row['TPOT'] = self._find_median(batchPrompt.time_per_output_token)
            
        if batchPrompt.prefill_avg_power is not None:
            new_row['Decode GPU Power'] = batchPrompt.tokenization_avg_power
            
        if batchPrompt.prefill_avg_gpu_util is not None:
            new_row['Decode GPU Util'] = batchPrompt.tokenization_avg_gpu_util
            
        if batchPrompt.prefill_avg_mem_util is not None:
            new_row['Decode Memory Util'] = batchPrompt.tokenization_avg_mem_util

        if batchPrompt.time_first_token_generation is not None and batchPrompt.time_rest_token_generation is not None:
            new_row['Total Time'] = np.median(np.array(batchPrompt.time_first_token_generation).reshape((-1,batch_size)) + np.array(batchPrompt.time_rest_token_generation))
            
        new_row = pd.Series(new_row)
        self.metrics_df = pd.concat([self.metrics_df, new_row.to_frame().T], ignore_index=True)

        
    def export_to_csv(self, batch_size, run_id=None):
        # Finalize the DataFrame, calculate metrics we are interested in, and export to CSV

        self.metrics_df['Prefill GPU Energy'] = self.metrics_df['Prefill GPU Power'] / 1000 * self.metrics_df['TTFT']
        self.metrics_df['Decoding GPU Energy'] = self.metrics_df['Decode GPU Power'] / 1000 * self.metrics_df['Total Decode Time (s)'] 
        self.metrics_df['E2E Energy'] = self.metrics_df['Prefill GPU Energy'] + self.metrics_df['Decoding GPU Energy']

        self.metrics_df['Throughput'] = (self.metrics_df['Prompt Token #']+self.metrics_df['Output Token #'])  / self.metrics_df['Total Time']
        self.metrics_df['Prefill Throughput'] = self.metrics_df['Prompt Token #'] / self.metrics_df['TTFT']
        self.metrics_df['Decoding Throughput'] = self.metrics_df['Output Token #'] / self.metrics_df['Total Decode Time (s)']

        self.metrics_df['Prefill normalized Energy'] = self.metrics_df['Prefill GPU Energy'] / self.metrics_df['Prompt Token #']
        self.metrics_df['Decoding normalized Energy'] = self.metrics_df['Decoding GPU Energy'] / self.metrics_df['Output Token #']


        postfix = self.model_name.split("/")[1]
        if run_id is not None:
            postfix = f"{postfix}_run-{run_id}"
        file_name = f"/local/scratch/a/shi676/llm_profiling/llm-decarbonization/llm-sys-main/splitwise/experiments/{postfix}_batch_{batch_size}.csv"
        self.metrics_df.to_csv(file_name,index=False)
        
    def _find_median(self, arr):
        if len(arr) == 0:
            return 0
        
        sorted_arr = sorted(arr)
        n = len(sorted_arr)
        if n % 2 == 0:
            return (sorted_arr[n//2 - 1] + sorted_arr[n//2]) / 2
        else:
            return sorted_arr[n//2]
        


def main(args: argparse.Namespace):
    print(args)
    gpu_monitor = GPUMonitor(gpu_id=args.gpu_id)

    model_types = {
        "bloom-3b":"bigscience/bloom-3b", 
        "bloom-1b":"bigscience/bloom-1b1", 
        "llama-7b":"openlm-research/open_llama_7b_v2", 
        "llama-3b":"openlm-research/open_llama_3b", 
        "llama-1b":"TinyLlama/TinyLlama-1.1B-Chat-v0.1",
        "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct"
    }

    model_name = model_types[args.model_shortcut] if args.model_shortcut else args.model

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=model_name,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        speculative_draft_tensor_parallel_size=\
            args.speculative_draft_tensor_parallel_size,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        ray_workers_use_nsight=args.ray_workers_use_nsight,
        use_v2_block_manager=args.use_v2_block_manager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        download_dir=args.download_dir,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        load_format=args.load_format,
        distributed_executor_backend=args.distributed_executor_backend,
        otlp_traces_endpoint=args.otlp_traces_endpoint,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)

    # Generate dummy inputs.
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_inputs: List[PromptInputs] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    if args.use_dummy_inputs:
        print("Using dummy inputs for benchmarking.")
        inputs = dummy_inputs
    else:
        print("Using real inputs for benchmarking.")
        # Use the following code to load the prompt from a file
        parser = MyParser()
        test_dataset = parser.loadJson(ALCAPA_DATA_PATH)
        # Set random seed
        random.seed(args.random_seed)
        # shuffle test dataset
        random.shuffle(test_dataset)
        inputs = [test_dataset[i * args.batch_size:(i + 1) * args.batch_size] for i in range(args.num_iters)]
        
    prefill_sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        use_beam_search=False,
        ignore_eos=True,
        max_tokens=1,
    )

    def run_prefill(inputs, batchPrompt, sampling_params = prefill_sampling_params):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # print("Warmup prefill...")
        for _ in range(args.num_iters_warmup):
            llm.generate(inputs,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            
        # print("Prefill...")
        if args.use_online_ttft:
            gpu_monitor.start()
            for _ in range(args.prefill_num_iters):
                outputs = llm.generate(inputs,
                             sampling_params=sampling_params,
                             use_tqdm=False)
                batchPrompt.time_first_token_generation += [output.metrics.first_token_time - output.metrics.arrival_time for output in outputs]
            gpu_monitor.stop()
        else:
            gpu_monitor.start()
            for _ in range(args.prefill_num_iters):
                outputs = llm.generate(inputs,
                            sampling_params=sampling_params,
                            use_tqdm=False)
                batchPrompt.time_first_token_generation += [output.metrics.first_token_time - output.metrics.first_scheduled_time for output in outputs]
            gpu_monitor.stop()

        for _ in range(args.num_iters_warmup):
            llm.generate(inputs,
                         sampling_params=sampling_params,
                         use_tqdm=False)
        
        batchPrompt.prefill_avg_power, batchPrompt.prefill_avg_gpu_util, batchPrompt.prefill_avg_mem_util = gpu_monitor.results_queue.get()

    def run_decoding(inputs, batchPrompt, sampling_params):
        outputs = llm.generate(inputs,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                        gpu_monitor=gpu_monitor)
        batchPrompt.time_rest_token_generation = [output.metrics.finished_time - output.metrics.first_token_time for output in outputs]
        # print(outputs[0].metrics)
        batchPrompt.tokenization_avg_power, batchPrompt.tokenization_avg_gpu_util, batchPrompt.tokenization_avg_mem_util = gpu_monitor.results_queue.get()

        return outputs


    def run_to_completion(inputs, profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                llm.generate(inputs,
                             sampling_params=sampling_params,
                             use_tqdm=False)
            print(p.key_averages())
        else:
            start_time = time.perf_counter()
            llm.generate(inputs,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency
        
    # Benchmark.
    metrics_tracker = MetricsTracker(model_name)
    for input_batch in tqdm(inputs):
        batchPrompt = BatchPrompt(input_batch)
        run_prefill(input_batch, batchPrompt)
        outputs = run_decoding(input_batch, batchPrompt, sampling_params)

        input_token_nums = [len(output.prompt_token_ids) for output in outputs]
        output_token_nums = [len(output.outputs[0].token_ids) for output in outputs]
        batchPrompt.update_token_stats(input_token_nums, output_token_nums)
        metrics_tracker.insert_latency_metrics(batchPrompt)
    
    metrics_tracker.export_to_csv(args.batch_size, args.run_id)
    print("Latency metrics exported to CSV.")

    # print("Warming up...")
    # for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
    #     run_to_completion(dummy_inputs, profile_dir=None)

    # if args.profile:
    #     profile_dir = args.profile_result_dir
    #     if not profile_dir:
    #         profile_dir = Path(
    #             "."
    #         ) / "vllm_benchmark_result" / f"latency_result_{time.time()}"
    #     print(f"Profiling (results will be saved to '{profile_dir}')...")
    #     run_to_completion(profile_dir=profile_dir)
    #     return

    # # Benchmark.
    # latencies = []
    # for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
    #     latencies.append(run_to_completion(profile_dir=None))
    # latencies = np.array(latencies)
    # percentages = [10, 25, 50, 75, 90, 99]
    # percentiles = np.percentile(latencies, percentages)
    # print(f'Avg latency: {np.mean(latencies)} seconds')
    # for percentage, percentile in zip(percentages, percentiles):
    #     print(f'{percentage}% percentile latency: {percentile} seconds')

    # Output JSON results if specified
    # if args.output_json:
    #     results = {
    #         "avg_latency": np.mean(latencies),
    #         "latencies": latencies.tolist(),
    #         "percentiles": dict(zip(percentages, percentiles.tolist())),
    #     }
    #     with open(args.output_json, "w") as f:
    #         json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--model-shortcut', type=str, default=None)
    parser.add_argument('--speculative-model', type=str, default=None)
    parser.add_argument('--num-speculative-tokens', type=int, default=None)
    parser.add_argument('--speculative-draft-tensor-parallel-size',
                        '-spec-draft-tp',
                        type=int,
                        default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--use-online-ttft', action='store_true')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=5,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument('--prefill-num-iters',
                        type=int,
                        default=50,
                        help='Number of iterations to run for prefill.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--use-dummy-inputs',
                        action='store_true',
                        help='use dummy inputs for benchmarking')
    parser.add_argument('--random-seed',
                        type=int,
                        default=42,
                        help='random seed for shuffling the dataset')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument("--device",
                        type=str,
                        default="auto",
                        choices=DEVICE_OPTIONS,
                        help='device type for vLLM execution')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='GPU ID to monitor.')
    parser.add_argument('--run-id',
                        type=str,
                        default=None,
                        help='Run ID for the experiment.')
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='block size of key/value cache')
    parser.add_argument(
        '--enable-chunked-prefill',
        action='store_true',
        help='If True, the prefill requests can be chunked based on the '
        'max_num_batched_tokens')
    parser.add_argument("--enable-prefix-caching",
                        action='store_true',
                        help="Enable automatic prefix caching")
    parser.add_argument('--use-v2-block-manager', action='store_true')
    parser.add_argument(
        "--ray-workers-use-nsight",
        action='store_true',
        help="If specified, use nsight to profile ray workers",
    )
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the latency results in JSON format.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument(
        '--load-format',
        type=str,
        default=EngineArgs.load_format,
        choices=[
            'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
            'bitsandbytes'
        ],
        help='The format of the model weights to load.\n\n'
        '* "auto" will try to load the weights in the safetensors format '
        'and fall back to the pytorch bin format if safetensors format '
        'is not available.\n'
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        'a numpy cache to speed up the loading.\n'
        '* "dummy" will initialize the weights with random values, '
        'which is mainly for profiling.\n'
        '* "tensorizer" will load the weights using tensorizer from '
        'CoreWeave. See the Tensorize vLLM Model script in the Examples'
        'section for more information.\n'
        '* "bitsandbytes" will load the weights using bitsandbytes '
        'quantization.\n')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument(
        '--otlp-traces-endpoint',
        type=str,
        default=None,
        help='Target URL to which OpenTelemetry traces will be sent.')
    args = parser.parse_args()
    main(args)
