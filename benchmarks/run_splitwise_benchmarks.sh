for i in 1 3 5; do
  CUDA_VISIBLE_DEVICES=0 python benchmark_vllm_offline.py --model-shortcut meta-llama-7b --gpu-ids 0 --batch-size $i --num-iters 30 --run-id ShareGPT-decoding-sim-1T4 --use-splitwise --use-dummy-inputs --input-len 160 --output-len 140 --use-online-ttft --prefill-num-iters 1 --num-iters-warmup 0 --gpu-memory-utilization 0.99 --enforce-eager --max-model-len 2048
  sleep 20
done
