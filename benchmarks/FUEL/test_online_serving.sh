wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

benchmark () {
  # Run the benchmark
  results_dir="/path/to/results/dir"
  model="/path/to/model"
  qps=$2
  input_len=$3
  output_len=$4
  tag=$5
  num_prompts=$6

  python ../benchmark_serving.py \
    --model $model \
    --backend vllm-demo \
    --endpoint /generate \
    --host localhost \
    --port 8301 \
    --save-result \
    --dataset-name arenahard \
    --dataset-path /path/to/dataset \
    --request-rate $qps \
    --result-dir $results_dir \
    --result-filename $tag-qps-$qps-input-$input_len-output-$output_len.json \
    --request_rate $qps \
    --gpu-ids $1 \
    --gpu-monitor-interval 0.2 \
    --warmup-ratio 0.2 \
    --use-deterministic-rate \
    --monitor-cpu \
    --gpu-monitor-truncate 1 \
    --disable-tqdm

  sleep 2
}

kill_server() {
  # Kill the server
  pkill -u uname -f "api_server.*8001"
}

main() {

  gpu=0
  # Start the server
  # For benchmark reasons, we need to use the demo api_server rather than OpenAI compatible api_server 
  # CUDA_VISIBLE_DEVICES=$gpu vllm serve openlm-research/open_llama_7b_v2 --config online_serving_config.yaml &
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.api_server \
  --model /path/to/model \
  --port 8301 \
  --host localhost \
  --disable-log-requests \
  --enable-chunked-prefill False \
  --max-model-len 15000 \
  --log-level warning &

  wait_for_server 8301 || exit 1
  sleep 1

  for i in 10; do
    qps=$i
    input_len=160
    output_len=140
    num_prompts=$(awk "BEGIN {print $qps * 30}")
    # if num_prompts < 30, set num_prompts to 30
    if (( $(echo "$num_prompts < 30" | bc -l) )); then
      num_prompts=30
    fi
    benchmark $gpu $qps $input_len $output_len "test_params" $num_prompts
    sleep 10
  done

  # Kill the server
  kill_server

  sleep 2


}

main "$@"