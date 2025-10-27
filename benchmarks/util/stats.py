import numpy as np
import pandas as pd

OUTPUT_DIR = '/path/to/output'

class MetricsTracker:
    def __init__(self, model_name):
        # Initialize a DataFrame with specific metrics as columns
        self.model_name = model_name
        self.metrics_df = pd.DataFrame({
            'Batch Size': pd.Series(dtype='int'),
            'Prompt Token #': pd.Series(dtype='int'),
            'Output Token #': pd.Series(dtype='int'),
            'TTFT': pd.Series(dtype='float'),
            'Prefill GPU Power': pd.Series(dtype='float'),
            'Prefill GPU Util': pd.Series(dtype='float'),
            'Prefill Memory Util': pd.Series(dtype='float'),
            'Total Decode Time (s)': pd.Series(dtype='float'),
            'TPOT': pd.Series(dtype='float'),
            'Decode GPU Power': pd.Series(dtype='float'),
            'Decode GPU Util': pd.Series(dtype='float'),
            'Decode Memory Util': pd.Series(dtype='float'),
            'Total Time': pd.Series(dtype='float')
        })
        
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
            new_row['Prefill GPU Power'] = np.sum(np.array(batchPrompt.prefill_avg_power))
            
        if batchPrompt.prefill_avg_gpu_util is not None:
            new_row['Prefill GPU Util'] = np.sum(np.array(batchPrompt.prefill_avg_gpu_util))
            
        if batchPrompt.prefill_avg_mem_util is not None:
            new_row['Prefill Memory Util'] = np.sum(np.array(batchPrompt.prefill_avg_mem_util))

        if batchPrompt.time_rest_token_generation is not None and total_output_tokens > 0:
            trtg = np.median(np.array(batchPrompt.time_rest_token_generation))
            batchPrompt.time_per_output_token = [batchPrompt.time_rest_token_generation[i] / (batchPrompt.output_tokens[i]-1) for i in range(batch_size)]
            new_row['Total Decode Time (s)'] = trtg
            # new_row['TBT'] = trtg / total_output_tokens
            new_row['TPOT'] = np.median(np.array(batchPrompt.time_per_output_token))
            
        if batchPrompt.prefill_avg_power is not None:
            new_row['Decode GPU Power'] = np.sum(np.array(batchPrompt.tokenization_avg_power))
            
        if batchPrompt.prefill_avg_gpu_util is not None:
            new_row['Decode GPU Util'] = np.sum(np.array(batchPrompt.tokenization_avg_gpu_util))
            
        if batchPrompt.prefill_avg_mem_util is not None:
            new_row['Decode Memory Util'] = np.sum(np.array(batchPrompt.tokenization_avg_mem_util))

        if batchPrompt.time_first_token_generation is not None and batchPrompt.time_rest_token_generation is not None:
            new_row['Total Time'] = np.median(np.array(batchPrompt.time_first_token_generation).reshape((-1,batch_size)) + np.array(batchPrompt.time_rest_token_generation))
            
        new_row = pd.Series(new_row)
        self.metrics_df = pd.concat([self.metrics_df, new_row.to_frame().T], ignore_index=True)

        
    def export_to_csv(self, batch_size, result_dir=OUTPUT_DIR, run_id=None):
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
        file_name = result_dir + f"{postfix}_batch_{batch_size}.csv"
        self.metrics_df.to_csv(file_name,index=False)
        
    def _find_median(self, arr):
        # deprecated
        if len(arr) == 0:
            return 0
        
        sorted_arr = sorted(arr)
        n = len(sorted_arr)
        if n % 2 == 0:
            return (sorted_arr[n//2 - 1] + sorted_arr[n//2]) / 2
        else:
            return sorted_arr[n//2]
