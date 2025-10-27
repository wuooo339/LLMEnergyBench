import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import os
from tqdm import tqdm
import seaborn as sns



device = "cuda:0"


def extract_user_input(text):
    # llama2 instruct
    match_inst = re.search(r'<s>\s*\[INST\]\s*<<SYS>>\s*(.*?)\s*<</SYS>>\s*(.*?)\s*\[/INST\]', text, re.DOTALL)
    if match_inst:
        return match_inst.group(2).strip()

    # qwen 
    match_im = re.search(r'<\|im_start\|>user\s*(.*?)<\|im_end\|>', text, re.DOTALL)
    if match_im:
        return match_im.group(1).strip()

    return text


def get_skywork_socres(filepath, tokenizer, scorer, eval_length=None, take_user_input=True):
    batch_size = 1
    with open(filepath) as f:
        data = json.load(f)
        print(len(data["generated_texts"]))
        if eval_length is None:
            eval_length = len(data["generated_texts"])
        # Replace \n with double spaces in generated texts and prompts
        processed_texts = [text for text in data["generated_texts"][:eval_length]]
        if take_user_input:
            processed_prompts = [extract_user_input(prompt) for prompt in data["prompt"][:eval_length]]
        else:
            processed_prompts = [prompt for prompt in data["prompt"][:eval_length]]
        scores = []
        for i in tqdm(range(0, len(processed_prompts), batch_size)):
            batch_prompts = processed_prompts[i:i+batch_size]
            batch_responses = processed_texts[i:i+batch_size]
            
            batch_inputs = []
            for prompt, response in zip(batch_prompts, batch_responses):
                conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
                formatted = tokenizer.apply_chat_template(conv, tokenize=False)
                tokenized = tokenizer(formatted, return_tensors="pt")
                batch_inputs.append(tokenized)

            batch_input = {k: torch.cat([conv[k] for conv in batch_inputs], dim=0).to(device) for k in batch_inputs[0].keys()}

            with torch.no_grad():
                batch_scores = scorer(**batch_input).logits[:, 0].tolist()
                scores.extend(batch_scores)
    return scores
                
def process_and_save_results(model_results_dict, save_path, tokenizer, scorer, eval_length=None, take_user_input=True):
    results_summary = {}
    
    for model_name, file_path in model_results_dict.items():
        scores = get_skywork_socres(file_path, tokenizer, scorer, eval_length, take_user_input)
        mean_value = np.mean(scores)
        median_value = np.median(scores)
        std_value = np.std(scores)

        results_summary[model_name] = scores
        print(f"\n{model_name} Statistics:")
        print(f"Mean: {mean_value}")
        print(f"Median: {median_value}")
        print(f"Standard Deviation: {std_value}")

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    existing_data.update(results_summary)

    with open(save_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    return results_summary


def visualize_results(results_path, dataset_name, model_keys):
    with open(results_path, "r") as f:
        results_data = json.load(f)

    plt.figure(figsize=(10, 6))
    for model_key in model_keys:
        if model_key in results_data:
            scores = results_data[model_key]
            plt.hist(scores, bins=30, alpha=0.5, label=f"{model_key}")

    plt.title(f"Score Distributions for {dataset_name}")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.legend()

    hist_output_path = f"{dataset_name}_score_distributions_hist.png"
    plt.savefig(hist_output_path)
    plt.show()

    plt.figure(figsize=(10, 6))
    for model_key in model_keys:
        if model_key in results_data:
            scores = results_data[model_key]
            sns.histplot(scores, bins=30, kde=True, alpha=0.5, label=f"{model_key}")

    plt.title(f"Score Distributions with KDE for {dataset_name}")
    plt.xlabel("Scores")
    plt.ylabel("Density")
    plt.legend()

    kde_output_path = f"{dataset_name}_score_distributions_kde.png"
    plt.savefig(kde_output_path)
    plt.show()

    return hist_output_path, kde_output_path


dataset_name = "areanahard8"

model_results_dict = {
    "qwen-7b-awq": "/path/to/file",
}


save_path = f"{dataset_name}_all_scores.json"

model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)


# Process and save results
results_summary = process_and_save_results(model_results_dict, save_path, rm_tokenizer, rm)


# Visualize the results
# visualize_results(save_path, dataset_name, ["qwen-7b", "qwen-14b", "qwen-32b", "llama2-7b", "llama2-13b"])

# visualize_results("./newsqa8_all_scores.json", "newsqa", ["qwen-7b", "qwen-14b", "qwen-32b", "qwen-7b-awq", "qwen-14b-awq", "qwen-32b-awq", "qwen-72b-awq"])

# visualize_results(save_path, "humaneval", ["llama2-7b", "llama2-13b", "llama2-7b-awq", "llama2-13b-awq"])
# visualize_results(save_path, "newsqa", ["qwen-7b-awq", "qwen-14b-awq", "qwen-32b-awq"])
# visualize_results(save_path, "alpaca", ["qwen-32b-awq", "qwen-32b-w8a8", "qwen-32b"])

