#!/bin/bash

# Benchmark serving script with GPU and CPU monitoring
# Make sure the vLLM server is already running at localhost:8000
set -euo pipefail

# Configuration
# All variables can be overridden by environment variables.
# Example:
#   MODEL=/path/to/model NUM_PROMPTS=512 ./run_benchmark.sh
# MODEL="/share-data/models/Qwen3-32B"
MODEL="${MODEL:-/share-data/models/Llama-3.1-70B-Instruct}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
BACKEND="${BACKEND:-vllm}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"  # Using ShareGPT dataset
NUM_PROMPTS="${NUM_PROMPTS:-2560}"         # Number of prompts to test
REQUEST_RATE="${REQUEST_RATE:-10}"         # Requests per second
OUTPUT_LEN="${OUTPUT_LEN:-1024}"           # max output length
MIN_PROMPT_LEN="${MIN_PROMPT_LEN:-}"       # 例如 100: 只测 >=100 tokens 的请求
RESULT_DIR="${RESULT_DIR:-/home/wzk/LLMEnergyBench/benchmark_results}"
# GPU monitoring settings
# Set GPU_IDS=auto to monitor all detected GPUs from nvidia-smi.
GPU_IDS="${GPU_IDS:-auto}"                 # Monitor GPUs
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"        # Ignore first 10% for power stats
DATASET_PATH="${DATASET_PATH:-/share-data/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}"
# Monitoring switches
ENABLE_KV_TRACE="${ENABLE_KV_TRACE:-1}"
# Free-form metadata tags written into result JSON.
PE_MODE_TAG="${PE_MODE_TAG:-unknown}"      # e.g. on/off
SPEC_MODE_TAG="${SPEC_MODE_TAG:-unknown}"  # e.g. draft_model/ngram/off
RESULT_STEM_OVERRIDE="${RESULT_STEM_OVERRIDE:-}"

# Create result directory if it doesn't exist (use absolute path)
mkdir -p "$RESULT_DIR"
# Change to benchmark directory
cd /home/wzk/LLMEnergyBench/benchmarks

auto_detect_gpu_ids() {
    if [[ "$GPU_IDS" != "auto" ]]; then
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] nvidia-smi not found; fallback GPU_IDS='0'"
        GPU_IDS="0"
        return 0
    fi

    local detected
    detected="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
        | awk -F',' '{gsub(/ /,"",$1); if ($1 ~ /^[0-9]+$/) print $1}' \
        | tr '\n' ' ' \
        | sed 's/[[:space:]]*$//')"

    if [[ -z "$detected" ]]; then
        echo "[WARN] Failed to auto-detect GPU IDs; fallback GPU_IDS='0'"
        GPU_IDS="0"
        return 0
    fi

    GPU_IDS="$detected"
}

auto_detect_gpu_ids
echo "Using GPU monitor IDs: $GPU_IDS"

# Build a readable result file prefix from model path.
MODEL_BASENAME="$(basename "$MODEL")"
RESULT_STEM="${RESULT_STEM_OVERRIDE:-${MODEL_BASENAME}-benchmark-$(date +%Y%m%d-%H%M%S)}"

CMD="python benchmark_serving.py \
    --backend $BACKEND \
    --model $MODEL \
    --host $HOST \
    --port $PORT \
    --dataset-name $DATASET_NAME \
    --dataset-path $DATASET_PATH \
    --sharegpt-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS \
    --request-rate $REQUEST_RATE \
    --save-result \
    --result-dir $RESULT_DIR \
    --result-filename ${RESULT_STEM}.json \
    --gpu-ids $GPU_IDS \
    --gpu-monitor-interval $GPU_MONITOR_INTERVAL \
    --warmup-ratio $WARMUP_RATIO \
    --monitor-cpu \
    --metadata pe_mode=$PE_MODE_TAG \
    --metadata spec_mode=$SPEC_MODE_TAG \
    --seed 42 \
    --trust-remote-code"

if [ "$ENABLE_KV_TRACE" = "1" ]; then
    CMD="$CMD --enable-kv-trace"
fi

# 只在 MIN_PROMPT_LEN 非空时添加该参数
if [ -n "$MIN_PROMPT_LEN" ]; then
    CMD="$CMD --min-prompt-len $MIN_PROMPT_LEN"
fi

# 执行命令
eval $CMD

echo ""
echo "Benchmark completed! Results saved to: $RESULT_DIR"
echo "Check the JSON file for detailed metrics including power consumption."
