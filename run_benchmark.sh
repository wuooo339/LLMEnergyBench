#!/bin/bash

# Benchmark serving script with GPU and CPU monitoring
# Make sure the vLLM server is already running at localhost:8000

# Configuration
MODEL="/share-data/wzk-1/model/Qwen3-8B"
HOST="localhost"
PORT="8000"
BACKEND="vllm"
DATASET_NAME="sharegpt"  # Using ShareGPT dataset
NUM_PROMPTS=100        # Number of prompts to test
REQUEST_RATE=50        # 10 requests per second
OUTPUT_LEN=512        # max output length
RESULT_DIR="/home/user/offload/FUEL/benchmark_results"
# GPU monitoring settings
GPU_IDS="0 1"          # Monitor both GPUs (adjust based on your setup)
GPU_MONITOR_INTERVAL=0.1  # Monitor interval in seconds
WARMUP_RATIO=0.1       # Ignore first 10% of requests for power stats
DATASET_PATH="/share-data/wzk-1/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

# Create result directory if it doesn't exist (use absolute path)
mkdir -p $RESULT_DIR
# Change to benchmark directory
cd /home/user/offload/FUEL/benchmarks
python benchmark_serving.py \
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
    --result-filename qwen3-8b-benchmark-$(date +%Y%m%d-%H%M%S).json \
    --gpu-ids $GPU_IDS \
    --gpu-monitor-interval $GPU_MONITOR_INTERVAL \
    --warmup-ratio $WARMUP_RATIO \
    --monitor-cpu \
    --seed 42 \
    --trust-remote-code 

echo ""
echo "Benchmark completed! Results saved to: $RESULT_DIR"
echo "Check the JSON file for detailed metrics including power consumption."