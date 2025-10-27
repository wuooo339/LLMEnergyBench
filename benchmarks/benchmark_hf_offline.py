import time
import torch
import gc
import argparse
from util.stats import MetricsTracker
from util.parse import Parser
from util.monitor import GPUMonitor
import numpy as np
import pandas as pd
from torch import cuda, bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
from huggingface_hub import login
import json


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
    
    
    
# Runs model
class ModelRunner:
    # Set up tokenizer and model 
    def __init__(self, model_path, gpu_id=0, gpu_monitor_interval=0.025):
        
        self.device = 'cuda'
        device_name = torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else 'CPU'
        print(f"Using device: {self.device} ({device_name})")
        print(f"Model path: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if model_path == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.gpu_monitor = GPUMonitor(gpu_id=gpu_id, interval=gpu_monitor_interval)
        
    def run_batch(self, batchPrompt):
        
        # Preprocess
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
             
        # Prompt phase
        input_ids = self.prompt_processing_batch(batchPrompt)
        
        
        # Run the prefill phase repetitively to amortize short period of process
        for i in range(50):
            if i == 2:
                self.gpu_monitor.start()
            
            if i >= 2 and i <= 47:
                first_token_ids = self.first_to_second_token_generation_batch(batchPrompt, input_ids, drop=False)
            else:
                first_token_ids = self.first_to_second_token_generation_batch(batchPrompt, input_ids, drop=True)
            
            if i == 47:
                self.gpu_monitor.stop()
                
        batchPrompt.prefill_avg_power, batchPrompt.prefill_avg_gpu_util, batchPrompt.prefill_avg_mem_util = self.gpu_monitor.results_queue.get()

        # Continue with the output token generation
        self.gpu_monitor.start()
        generate_ids = self.second_to_end_token_generation_batch(batchPrompt, first_token_ids)
        self.gpu_monitor.stop()
        batchPrompt.tokenization_avg_power, batchPrompt.tokenization_avg_gpu_util, batchPrompt.tokenization_avg_mem_util = self.gpu_monitor.results_queue.get()
        # print(1)
        input_tokens = [len(self.tokenizer(prompt, add_special_tokens=False, padding=False).input_ids) for prompt in batchPrompt.prompts]
        output_tokens = []
        

        for gen_id in generate_ids:
            text = self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            batchPrompt.add_generated_text(text)
            output_tokens.append(len(self.tokenizer(text, add_special_tokens=False, padding=False).input_ids))

        # print(2)
        batchPrompt.update_token_stats(input_tokens, [output_tokens[i]-input_tokens[i]-1 for i in range(len(input_tokens))])


    def prompt_processing_batch(self, batchPrompt):
        inputs = self.tokenizer(batchPrompt.prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return inputs.input_ids.to(self.device)

    def first_to_second_token_generation_batch(self, batchPrompt, input_ids, drop=True):      
        # Generate
        first_token_time_start = time.time()
        first_token_ids = self.model.generate(input_ids, max_new_tokens=1, pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()        
        first_token_time_end = time.time()

        # Record
        if not drop:
            batchPrompt.time_first_token_generation.append(first_token_time_end - first_token_time_start)
        return first_token_ids
    
    def second_to_end_token_generation_batch(self, batchPrompt, first_token_ids):
        token_generation_start = time.time()
        generate_ids = self.model.generate(first_token_ids, max_new_tokens=150, do_sample=True, repetition_penalty=1.2, pad_token_id=self.tokenizer.eos_token_id)
        token_generation_end = time.time()
        batchPrompt.time_rest_token_generation = (token_generation_end - token_generation_start)
        return generate_ids
    
    def run_batch_e2e(self, batchPrompt, align_processing=False):
        # Preprocess
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
             
        # Prompt phase
        input_ids = self.prompt_processing_batch(batchPrompt)

        torch.cuda.synchronize()
        start_time = time.time()
        self.gpu_monitor.start()
        
        # Generate
        if align_processing:
            generate_ids = self.model.generate(input_ids, max_length=150, min_length=150, do_sample=True, repetition_penalty=1.2)
        else:
            generate_ids = self.model.generate(input_ids, max_new_tokens=150, min_new_tokens=150, do_sample=True, repetition_penalty=1.2)

        torch.cuda.synchronize()
        self.gpu_monitor.stop()
        end_time = time.time()

        input_tokens = [len(self.tokenizer(prompt, add_special_tokens=False, padding=False).input_ids) for prompt in batchPrompt.prompts]
        output_tokens = []
        
        for gen_id in generate_ids:
            text = self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            batchPrompt.add_generated_text(text)
            output_tokens.append(len(self.tokenizer(text, add_special_tokens=False, padding=False).input_ids))
        
        batchPrompt.update_token_stats(input_tokens, [output_tokens[i]-input_tokens[i] for i in range(len(input_tokens))])
        avg_power, avg_gpu_util, avg_mem_util = self.gpu_monitor.results_queue.get()
        e2e_latency = end_time - start_time
        
        s = pd.Series({'Batch Size': len(batchPrompt.prompts), 'Prompt Token #': sum(batchPrompt.input_tokens), 'Output Token #': sum(batchPrompt.output_tokens), 'E2E Latency': e2e_latency, 'Avg GPU Power': avg_power, 'Avg GPU Util': avg_gpu_util, 'Avg Memory Util': avg_mem_util, 'E2E Energy': avg_power / 1000 * e2e_latency})

        return batchPrompt, s


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference.")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to use for generation")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--niter", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-monitor-interval", type=float, default=0.025)
    parser.add_argument("--e2e", action='store_true')
    parser.add_argument("--align_processing", action='store_true')
    args = parser.parse_args()
    
    models = {
        "bloom-3b":"bigscience/bloom-3b", 
        "bloom-1b":"bigscience/bloom-1b1", 
        "llama-7b":"openlm-research/open_llama_7b_v2", 
        "llama-3b":"openlm-research/open_llama_3b", 
        "llama-1b":"TinyLlama/TinyLlama-1.1B-Chat-v0.1",
        "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct"
    }
    
    if args.model_name not in models:
        raise ValueError("Unknown model path!")
    # if args.model_name == "llama3.1-8b":
    #     login(token="put your token here")
    
    model_runner = ModelRunner(models[args.model_name], gpu_id=args.gpu_id, gpu_monitor_interval=args.gpu_monitor_interval)
    metricsTracker = MetricsTracker(models[args.model_name])

    # Generate test dataset
    parser = Parser()
    test_dataset = parser.loadJson('datasets/alpaca_data.json')
    # print(len(test_dataset))

    # Set random seed
    random.seed(args.random_seed)
    # shuffle test dataset
    random.shuffle(test_dataset)
    
    batch_size = int(args.batch_size)
    num_iter = int(args.niter)
    # print(num_iter)
    # recommended num_iter: {
    # 1B: {1:200, 2:200, 4:200, 8:200, 16:200, 32:135, 64:75}, 
    # 3B: {1:200, 2:200, 4:200, 8:150, 16:100, 32:60, 64:35}, 
    # 7B: {1:180, 2:150, 4:120, 8:100, 16:70, 32:35, 64:15},
    # 8B: {1:130, 2:100, 4:100, 8:90, 16:70, 32:30, 64:13}
    # }
    # Each num_iter keeps GPU running for about 10 mins
    
    print(f"Batch Size : {batch_size}")
    if args.e2e:
        metrics_df = pd.DataFrame()
        for i in tqdm(range(0, num_iter * batch_size, batch_size)):
            batch = test_dataset[i:i+batch_size]
            batchPrompt = BatchPrompt(batch)
            batchPrompt, metrics = model_runner.run_batch_e2e(batchPrompt, align_processing=args.align_processing)
            metrics_df = pd.concat([metrics_df, metrics.to_frame().T], axis=0)
        mid_fix = f"e2e-{args.run_id}" if args.run_id is not None else "e2e"
        metrics_df.to_csv(f"./experiments/{models[args.model_name].split('/')[-1]}_{mid_fix}_batch_{args.batch_size}.csv", index=False)
    else:
        for i in tqdm(range(0, (2 + num_iter) * batch_size, batch_size)):
            batch = test_dataset[i:i+batch_size]
            batchPrompt = BatchPrompt(batch)
            model_runner.run_batch(batchPrompt)
            if (i != 0 and i != batch_size):
                metricsTracker.insert_latency_metrics(batchPrompt)
                # batchPrompt.printGeneratedResult()
        
        metricsTracker.export_to_csv(batch_size,run_id=args.run_id)
      
