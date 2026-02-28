#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"

patterns=(
  "./run_page_eviction_ab.sh"
  "/home/wzk/LLMEnergyBench/run_benchmark.sh"
  "python benchmark_serving.py"
  "vllm serve .*--port ${PORT}"
)

echo "[1/4] Matched processes:"
for p in "${patterns[@]}"; do
  pgrep -af "$p" || true
done

mapfile -t pids < <(
  {
    for p in "${patterns[@]}"; do
      pgrep -f "$p" || true
    done
  } | sort -u
)

if [[ ${#pids[@]} -gt 0 ]]; then
  echo "[2/4] SIGTERM: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
fi

echo "[3/4] Wait 10s, then SIGKILL leftovers"
for _ in {1..10}; do
  alive=()
  for pid in "${pids[@]:-}"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
  done
  [[ ${#alive[@]} -eq 0 ]] && break
  sleep 1
done

alive=()
for pid in "${pids[@]:-}"; do
  kill -0 "$pid" 2>/dev/null && alive+=("$pid")
done
if [[ ${#alive[@]} -gt 0 ]]; then
  echo "SIGKILL: ${alive[*]}"
  kill -KILL "${alive[@]}" 2>/dev/null || true
fi

echo "[4/4] Free listener on :${PORT}"
if command -v lsof >/dev/null 2>&1; then
  mapfile -t lpids < <(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)
  if [[ ${#lpids[@]} -gt 0 ]]; then
    kill -TERM "${lpids[@]}" 2>/dev/null || true
    sleep 1
    for pid in "${lpids[@]}"; do
      kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
    done
  fi
fi

echo "Done. Remaining related processes:"
mapfile -t remaining_pids < <(
  {
    for p in "${patterns[@]}"; do
      pgrep -f "$p" || true
    done
  } | sort -u
)

live_remaining=()
for pid in "${remaining_pids[@]:-}"; do
  [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
  kill -0 "$pid" 2>/dev/null || continue
  live_remaining+=("$pid")
done

if [[ ${#live_remaining[@]} -gt 0 ]]; then
  ps -o pid=,etimes=,cmd= -p "$(IFS=,; echo "${live_remaining[*]}")"
else
  echo "(none)"
fi
