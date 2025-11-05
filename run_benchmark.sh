#!/bin/bash

# Benchmark serving script with GPU and CPU monitoring
# Make sure the vLLM server is already running at localhost:8000

# Configuration
MODEL="/share-data/wzk-1/model/deepseek-v2-lite"
HOST="localhost"
PORT="8000"
BACKEND="vllm"
DATASET_NAME="sharegpt"  # Using ShareGPT dataset
NUM_PROMPTS=2560         # Number of prompts to test
REQUEST_RATE=128          # 10 requests per second
OUTPUT_LEN=4096           # max output length
MIN_PROMPT_LEN=""       # 最小prompt长度（留空表示不限制），例如 "--min-prompt-len 100" 只测试>=100 tokens的请求
RESULT_DIR="/home/user/offload/FUEL/benchmark_results"
# GPU monitoring settings
GPU_IDS="0 1 2 3"          # Monitor GPUs (adjust based on your setup)
GPU_MONITOR_INTERVAL=0.05  # Monitor interval in seconds
WARMUP_RATIO=0.1       # Ignore first 10% of requests for power stats
DATASET_PATH="/share-data/wzk-1/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

# Create result directory if it doesn't exist (use absolute path)
mkdir -p $RESULT_DIR
# Change to benchmark directory
cd /home/user/offload/FUEL/benchmarks

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
    --result-filename deepseek-v2-lite-benchmark-\$(date +%Y%m%d-%H%M%S).json \
    --gpu-ids $GPU_IDS \
    --gpu-monitor-interval $GPU_MONITOR_INTERVAL \
    --warmup-ratio $WARMUP_RATIO \
    --monitor-cpu \
    --enable-kv-trace \
    --seed 42 \
    --trust-remote-code"

# 只在 MIN_PROMPT_LEN 非空时添加该参数
if [ -n "$MIN_PROMPT_LEN" ]; then
    CMD="$CMD --min-prompt-len $MIN_PROMPT_LEN"
fi

# 执行命令
eval $CMD

echo ""
echo "Benchmark completed! Results saved to: $RESULT_DIR"
echo "Check the JSON file for detailed metrics including power consumption."