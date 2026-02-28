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
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptInputs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser

# from zeus.monitor import ZeusMonitor
import pandas as pd
from util.monitor import GPUMonitor
from util.stats import MetricsTracker
from util.parse import MyParser

ALCAPA_DATA_PATH = './power_monitoring_dataset/alpaca_data.json'
RESULT_DIR = '/local/scratch/a/shi676/llm_profiling/results/motivitional-fig1/l40/'  

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

def main(args: argparse.Namespace):
    print(args)
    gpu_monitors = []
    for i in args.gpu_ids:
        gpu_monitors.append(GPUMonitor(gpu_id=i))

    model_types = {
        "bloom-3b":"bigscience/bloom-3b", 
        "bloom-1b":"bigscience/bloom-1b1", 
        "llama-7b":"openlm-research/open_llama_7b_v2",
        "meta-llama-7b":"meta-llama/Llama-2-7b-hf", 
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
        max_num_seqs=args.batch_size if args.unlock_max_num_seqs else 256,
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
        inputs = [dummy_inputs for _ in range(args.num_iters)]
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
            for gpu_monitor in gpu_monitors:
                gpu_monitor.start()
            for _ in range(args.prefill_num_iters):
                outputs = llm.generate(inputs,
                             sampling_params=sampling_params,
                             use_tqdm=False)
                batchPrompt.time_first_token_generation += [output.metrics.first_token_time - output.metrics.arrival_time for output in outputs]
            for gpu_monitor in gpu_monitors:
                gpu_monitor.stop()
        else:
            for gpu_monitor in gpu_monitors:
                gpu_monitor.start()
            for _ in range(args.prefill_num_iters):
                outputs = llm.generate(inputs,
                            sampling_params=sampling_params,
                            use_tqdm=False)
                batchPrompt.time_first_token_generation += [output.metrics.first_token_time - output.metrics.first_scheduled_time for output in outputs]
            for gpu_monitor in gpu_monitors:
                gpu_monitor.stop()

        for _ in range(args.num_iters_warmup):
            llm.generate(inputs,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            
        if args.inspect_batch_splits:
            print("Checking if the batch is split into multiple parts...")
            print(batchPrompt.time_first_token_generation[-len(outputs):])
        

        batchPrompt.prefill_avg_power, batchPrompt.prefill_avg_gpu_util, batchPrompt.prefill_avg_mem_util = [], [], []
        for gpu_monitor in gpu_monitors:
            a, b, c = gpu_monitor.results_queue.get()
            batchPrompt.prefill_avg_power.append(a)
            batchPrompt.prefill_avg_gpu_util.append(b)
            batchPrompt.prefill_avg_mem_util.append(c)

    def run_decoding(inputs, batchPrompt, sampling_params):
        outputs = llm.generate(inputs,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                        gpu_monitors=gpu_monitors)
        batchPrompt.time_rest_token_generation = [output.metrics.finished_time - output.metrics.first_token_time for output in outputs]
        # print(outputs[0].metrics)
        batchPrompt.tokenization_avg_power, batchPrompt.tokenization_avg_gpu_util, batchPrompt.tokenization_avg_mem_util = [], [], []
        for gpu_monitor in gpu_monitors:
            a, b, c = gpu_monitor.results_queue.get()
            batchPrompt.tokenization_avg_power.append(a)
            batchPrompt.tokenization_avg_gpu_util.append(b)
            batchPrompt.tokenization_avg_mem_util.append(c)

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
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            for gpu_monitor in gpu_monitors:
                gpu_monitor.start()
            outputs = llm.generate(inputs,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            for gpu_monitor in gpu_monitors:
                gpu_monitor.stop()

            if args.use_online_ttft:
                e2e_latency = [output.metrics.finished_time - output.metrics.arrival_time for output in outputs]
            else:
                e2e_latency = [output.metrics.finished_time - output.metrics.first_scheduled_time for output in outputs]
            avg_power, avg_gpu_util, avg_mem_util = [], [], []
            for gpu_monitor in gpu_monitors:
                a, b, c = gpu_monitor.results_queue.get()
                avg_power.append(a)
                avg_gpu_util.append(b)
                avg_mem_util.append(c)
            # in run_batch, they record all the GPU power readings every 100 ms and report the median power, which is confusing
            # Theoretically, using the arithemtic mean of the power readings is identical to calculating the energy consumed in each time interval and summing them up
            # Plus, the figures do not report the median power, but the total energy consumption
            # So that's another flaw in their code
            # The default interval is 0.025s, but for non-splitwise measurement, we may want to go with 0.1s and see if the results are consistent with adding the two phases of energy consumption up

            input_token_nums = [len(output.prompt_token_ids) for output in outputs]
            output_token_nums = [len(output.outputs[0].token_ids) for output in outputs]

            metrics = {
                'Batch Size': args.batch_size,
                'Prompt Token #': input_token_nums,
                'Output Token #': output_token_nums,
                'E2E Latency': e2e_latency,
                'Avg Power': avg_power,
                'Avg GPU Util': avg_gpu_util,
                'Avg Mem Util': avg_mem_util,
            }

            return outputs, metrics
        
    # Benchmark.
    metrics_df = pd.DataFrame()
    metrics_json = []
    metrics_tracker = MetricsTracker(model_name)
    for input_batch in tqdm(inputs):
        batchPrompt = BatchPrompt(input_batch)
        if args.use_splitwise:
            run_prefill(input_batch, batchPrompt)
            outputs = run_decoding(input_batch, batchPrompt, sampling_params)
            input_token_nums = [len(output.prompt_token_ids) for output in outputs]
            output_token_nums = [len(output.outputs[0].token_ids) for output in outputs]
            batchPrompt.update_token_stats(input_token_nums, output_token_nums)
            metrics_tracker.insert_latency_metrics(batchPrompt)
        else:
            outputs, m = run_to_completion(input_batch)
            metrics_json.append(m)
            s = pd.Series([args.batch_size, sum(m['Prompt Token #']), sum(m['Output Token #']), np.median(m['E2E Latency']), np.sum(m['Avg Power']), np.sum(m['Avg GPU Util']), np.sum(m['Avg Mem Util']), np.median(m['E2E Latency'])*m['Avg Power']/1000], index=['Batch Size', 'Prompt Token #', 'Output Token #', 'E2E Latency', 'Avg Power', 'Avg GPU Util', 'Avg Mem Util', 'E2E Energy'])
            metrics_df = pd.concat([metrics_df, s.to_frame().T], axis=0)
    
    if args.use_splitwise:
        metrics_tracker.export_to_csv(args.batch_size, result_dir=RESULT_DIR, run_id=args.run_id+f'_input_{args.input_len}_output_{args.output_len}')
    else:
        metrics_df.to_csv(RESULT_DIR + f"{args.model_shortcut}_run-{args.run_id}_input_{args.input_len}_output_{args.output_len}_batch_{args.batch_size}.csv", index=False)
        with open(RESULT_DIR + f"{args.model_shortcut}_run-{args.run_id}_input_{args.input_len}_output_{args.output_len}_batch_{args.batch_size}.json", "w") as f:
            json.dump(metrics_json, f)
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
    parser.add_argument('--use-splitwise', action='store_true')
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
                        default=30,
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "openvino", "tpu", "xpu"],
        help='device type for vLLM execution, supporting CUDA, OpenVINO and '
        'CPU.')
    parser.add_argument('--gpu-ids',
                        type=int,
                        nargs="+",
                        default=None,
                        help='GPU IDs to monitor.')
    parser.add_argument('--gpu-monitoring-interval',
                        type=float,
                        default=0.025,
                        help='Log Interval for GPU monitoring.')
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
    parser.add_argument('--inspect-batch-splits',
                        action='store_true',
                        help='Inspect if the batch is split into multiple parts.')
    parser.add_argument('--unlock-max-num-seqs',
                        action='store_true',
                        help='Unlock the maximum number of sequences per iteration.')
    args = parser.parse_args()
    main(args)
