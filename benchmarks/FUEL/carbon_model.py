
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
A100_EMBODIED_CARBON = 26.34 * 10**3
T4_EMBODIED_CARBON = 10.3 * 10**3
V100_EMBODIED_CARBON = 20 * 10**3
H100_EMBODIED_CARBON = 29.92 * 10**3
L40_EMBODIED_CARBON = 26.6 * 10**3
AMD_EPYC_7443_EMBODIED_CARBON = 9.98 * 10**3
Intel_Xeon_Platinum_8480_plus_EMBODIED_CARBON =  42.81 * 10**3
MEM_EMBODIED_CARBON_PER_G = 65


LIFETIME = 7 * 365 * 24 * 3600
# gCO2eq/kWh
carbon_intensity = 518
# Convert Carbon Intensity to gCO2eq/J
carbon_intensity_j = carbon_intensity / 3600 / 1000
ttft_slo = 1000
tpot_slo = 200
score_path = "/path/to/score"

score_thresh = 10
score_thresh_list = [-5, -4,-3, -2, -1, 0, 1,2, 3,4,5,6, 7,8,9, 10, 11,12,13,14,15, 20, 25, 30]


def cal_mem_op_emb(dict_name, cur_item):
    mem_use = cur_item["cpu_stats"]["avg_mem_util"] * 0.4
    if "l40" in dict_name:
        mem_emb = 504*MEM_EMBODIED_CARBON_PER_G / 4
    else:
        mem_emb = 1031*MEM_EMBODIED_CARBON_PER_G / 8
    cur_item["mem_operational_energy"] = cur_item["duration (s)"] * mem_use
    cur_item["mem_embodied_carbon"] = mem_emb * (cur_item['duration (s)'] / LIFETIME)

def cal_cpu_op_emb(dict_name, cur_item):
    cpu_util = cur_item["cpu_stats"]["avg_cpu_util"]

    if "l40" in dict_name:
        cpu_use = cpu_util * (200-96) + 96
        cpu_emb = AMD_EPYC_7443_EMBODIED_CARBON / 4
    else:
        cpu_use = 182 + (330-182)/10 * cpu_util
        cpu_emb = Intel_Xeon_Platinum_8480_plus_EMBODIED_CARBON *2 / 8
    cur_item["cpu_operational_energy"] = cur_item["duration (s)"] * cpu_use
    cur_item["cpu_embodied_carbon"] = cpu_emb * (cur_item['duration (s)'] / LIFETIME)
    
def calculate_ppl(cumulative_log_probs, output_len):
    ppl_list = []
    for log_prob, length in zip(cumulative_log_probs, output_len):
        if length > 0:
            avg_log_prob = log_prob / length
            ppl = math.exp(-avg_log_prob)
        else:
            ppl = float('inf')
        ppl_list.append(ppl)
    return ppl_list

def filter_sequences(cur_data, score_thresh, ttft_slo, tpot_slo, model_name):
    scores = []
    with open(score_path) as f:
        score_dict = json.load(f)
        if model_name not in score_dict:
            return 10, 10, 100, []
        scores = score_dict[model_name]
    
    assert len(scores) == len(cur_data['ttfts'])
    ttfts = [x * 1000 for x in cur_data['ttfts']]
    tpots = [(x-y)*1000/max(1, (o-1)) for x,y,o in zip(cur_data['e2els'], cur_data['ttfts'], cur_data["output_lens"])]
    slo_met_count = 0
    filtered_count = 0
    total_count = len(ttfts)

    total_filtered_tokens = 0
    filtered_idx_list = []
    for i in range(len(cur_data["output_lens"])):
        if ttfts[i] <= ttft_slo and tpots[i] <= tpot_slo:
            slo_met_count += 1
            if scores[i] >= score_thresh:
                filtered_idx_list.append(i)
                filtered_count += 1
                total_filtered_tokens += cur_data["output_lens"][i]
                total_filtered_tokens += cur_data["input_lens"][i]
    slo_percentage = (slo_met_count / total_count) * 100
    filtered_percentage = (filtered_count / total_count) * 100
    return slo_percentage, filtered_percentage, total_filtered_tokens, filtered_idx_list


def load_model_data_multigpu(dict_name, model_name, input_len, output_len, QPS, pathprefix, num_gpus=1, dataset="shareGPT"):
    data_list = []
    qps_ppl = {}
    

    for q in QPS:
        relative_path = f"{model_name}/{model_name}-qps-{q}.json"
        file_path = os.path.join(pathprefix, relative_path)
        if not os.path.exists(file_path):
            continue
        with open(file_path) as f:
            data = json.load(f)
            total_tokens = data["total_output_tokens"] + data["total_input_tokens"]
            cpu_utils_array = np.array(data["cpu_trace"]["cpu_utils"])
            mem_util_array = np.array(data["cpu_trace"]["mem_utils"])
            if len(data["binded_cpus"]) == 0 or data["binded_cpus"] is not None:
                selected_cpu_utils = cpu_utils_array
            else:
                selected_cpu_utils = cpu_utils_array[:, data["binded_cpus"]]

            cpu_util_row_means = np.mean(selected_cpu_utils, axis=1)
            cpu_util_median = np.median(cpu_util_row_means)
            mem_util_median = np.median(mem_util_array)

            write_io = data["cpu_stats"]["disk_io_stats"]["write_count"] / data['duration']
            read_io = data["cpu_stats"]["disk_io_stats"]["read_count"] / data['duration']

            ppl = calculate_ppl(data["cumulative_logprob"], data["output_lens"])
            qps_ppl[q] = ppl
            
            for score in score_thresh_list:
                slo_percentage, filtered_percentage, total_filtered_tokens, filtered_indices = filter_sequences(data, score, ttft_slo, tpot_slo, model_name)
                cur_item = {
                    'QPS': q,
                    'duration (s)': data['duration'],
                    'total_tokens': total_tokens,
                    'TTFT (ms)': data['median_ttft_ms'],
                    'TPOT (ms)': data['median_tpot_ms'],
                    "gpu_power (mW)": [data["gpu_power_stats"][str(i)]["avg_power"] for i in range(num_gpus)],
                    "gpu_util (%)": [data["gpu_power_stats"][str(i)]["avg_gpu_util"] for i in range(num_gpus)],
                    'total throughput (token/s)': data['total_token_throughput'],
                    "model": model_name,
                    "dataset": dataset,
                    "input_len": input_len,
                    "output_len": output_len,
                    "cpu_utils": cpu_util_median,
                    "mem_utils": mem_util_median,
                    "write_io": write_io,
                    "read_io": read_io,
                    "score_thresh": score,
                    "slo_attn": slo_percentage,
                    "filtered_percentage": filtered_percentage,
                    "filtered_total_tokens": total_filtered_tokens
                }
                total_power = 0.0
                avg_gpu_util = 0.0
                cal_mem_op_emb(dict_name, cur_item)
                cal_cpu_op_emb(dict_name, cur_item)
                for i in range(num_gpus):
                    cur_item[f"gpu_power {i}"] = data["gpu_power_stats"][str(i)]["avg_power"]
                    cur_item[f"gpu_util {i}"] = data["gpu_power_stats"][str(i)]["avg_gpu_util"]
                    total_power += cur_item[f"gpu_power {i}"]
                    avg_gpu_util += cur_item[f"gpu_util {i}"] 
                avg_gpu_util /= num_gpus
                cur_item["gpu_power (mW)"] = total_power
                cur_item["gpu_util (%)"] = avg_gpu_util
                data_list.append(cur_item)

    df = pd.DataFrame(data_list)
    df['gpu_energy_per_token'] = df['gpu_power (mW)'] * df['duration (s)'] / df['total_tokens'] / 1000
    df['gpu_total_energy'] = df['gpu_power (mW)'] * df['duration (s)'] / 1000
    df['total_energy'] = df['gpu_total_energy'] + df["cpu_operational_energy"] + df["mem_operational_energy"]
    df["energy_per_token"] = df['total_energy'] / df['total_tokens']
    df["filtered_energy_per_token"] =  df['total_energy'] / df['filtered_total_tokens']
    df["total_operational_carbon"] = df['total_energy']* carbon_intensity_j
    df["gpu_embodied_carbon"] = H100_EMBODIED_CARBON * (df['duration (s)'] / LIFETIME)
    df["total_embodied_carbon"] = df["gpu_embodied_carbon"] + df["mem_embodied_carbon"] +df["cpu_embodied_carbon"]
    df['gpu_filtered_energy_per_token (J)'] = df['gpu_power (mW)'] * df['duration (s)'] / df['filtered_total_tokens'] / 1000
    df["total_carbon"] = df["total_embodied_carbon"] + df["total_operational_carbon"] 
    df["carbon_per_token"] = df["total_carbon"] / df['total_tokens']
    df["embodied_carbon_per_token"] = df["total_embodied_carbon"] / df['total_tokens']
    df["operational_carbon_per_token"] = df["total_operational_carbon"] / df['total_tokens']
    df["filtered_carbon_per_token"]  = df["total_carbon"] / df['filtered_total_tokens']
    df["filtered_embodied_carbon_per_token"]  = df["total_embodied_carbon"] / df['filtered_total_tokens']
    df["filtered_operational_carbon_per_token"]  = df["total_operational_carbon"] / df['filtered_total_tokens']
    
 
    return df, qps_ppl
