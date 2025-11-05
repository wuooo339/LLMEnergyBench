#!/bin/bash
# KV Cache 实时监控启动脚本
# 持续监控 vLLM KV Cache 使用情况并保存 trace 数据

cd /home/user/offload/FUEL

# 配置参数
HOST="localhost"
PORT="8000"
INTERVAL="1.0"              # 更新间隔（秒）
HISTORY_SIZE="60"           # 实时显示保留的历史记录数量
OUTPUT_DIR="./kv_cache_traces"  # trace 数据保存目录
SESSION_NAME="monitor"      # 会话名称

# 运行监控脚本
python monitor_kv_cache.py \
    --host "$HOST" \
    --port "$PORT" \
    --interval "$INTERVAL" \
    --history-size "$HISTORY_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --session-name "$SESSION_NAME"

# 使用说明:
# --host: vLLM 服务器地址 (默认 localhost)
# --port: vLLM 服务器端口 (默认 8000)
# --interval: 更新间隔秒数 (默认 1.0)
# --history-size: 实时显示保留的历史统计数量 (默认 60，不影响 trace 完整记录)
# --output-dir: trace 数据保存目录 (默认 ./kv_cache_traces)
# --session-name: 会话名称，用于文件命名 (默认 monitor)
#
# 退出监控: 按 Ctrl+C
# Trace 数据将自动保存为 JSON 文件，包含：
#   - metadata: 监控会话信息（开始/结束时间、总采样数等）
#   - static_config: KV Cache 静态配置（总 blocks、block 大小等）
#   - trace: 完整的时序数据（每个采样点的详细信息）
#   - summary: 统计摘要（平均值、峰值）
#
# Trace 数据包含指标:
#   - timestamp: Unix 时间戳
#   - elapsed_seconds: 从监控开始的经过时间
#   - datetime: 人类可读的时间
#   - cache_usage_perc: KV Cache 使用百分比
#   - used_blocks: 已使用的 blocks 数量
#   - free_blocks: 空闲的 blocks 数量
#   - used_tokens: 已使用的 tokens 数量
#   - requests_running: 运行中的请求数（并发数）
#   - requests_waiting: 等待中的请求数（队列数）
#
