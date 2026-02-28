#!/bin/bash

# Simple A/B benchmark for PageEviction:
# 1) run baseline (PE off)
# 2) run PageEviction (PE on)
# 3) print energy/throughput comparison
#
# This script uses run_benchmark.sh under the hood.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional clear mode:
# --clear / CLEAR_RESULTS=1: remove old files in RESULT_DIR before benchmark.
CLEAR_RESULTS="${CLEAR_RESULTS:-0}"
for arg in "$@"; do
  case "$arg" in
    --clear)
      CLEAR_RESULTS="1"
      ;;
    --no-clear)
      CLEAR_RESULTS="0"
      ;;
    *)
      echo "Unknown argument: ${arg}"
      echo "Usage: $0 [--clear|--no-clear]"
      exit 1
      ;;
  esac
done

# ===== Server configuration =====
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/share-data/models/Llama-3.1-70B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-/share-data/models/Llama-3.1-8B-Instruct}"
TP_SIZE="${TP_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
DTYPE="${DTYPE:-half}"
ENABLE_EAGER="${ENABLE_EAGER:-1}"
KV_CACHE_BUDGET="${KV_CACHE_BUDGET:-4096}"
# Allow chunked prefill while PE is on (requires vLLM support):
# 0 = strict PE behavior (default), 1 = allow chunked prefill with PE.
PE_CHUNKED_PREFILL="${PE_CHUNKED_PREFILL:-0}"

# Use speculative decode or not: 1/0
USE_SPEC_DECODE="${USE_SPEC_DECODE:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-4}"

# Auto launch vLLM server or assume you launch it manually: 1/0
START_SERVER="${START_SERVER:-1}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-600}"

# ===== Benchmark configuration =====
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/benchmark_results/page_eviction_ab}"
JSON_RESULT_DIR="${JSON_RESULT_DIR:-${RESULT_DIR}/json}"
SUMMARY_DIR="${SUMMARY_DIR:-${RESULT_DIR}/summary}"
PLOT_DIR="${PLOT_DIR:-${RESULT_DIR}/plots}"
METRICS_DIR="${METRICS_DIR:-${RESULT_DIR}/metrics}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${RESULT_DIR}/server_logs}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-256}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-8}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-512}"
BENCH_DATASET_PATH="${BENCH_DATASET_PATH:-/share-data/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}"
BENCH_GPU_IDS="${BENCH_GPU_IDS:-auto}"
BENCH_GPU_MONITOR_INTERVAL="${BENCH_GPU_MONITOR_INTERVAL:-0.05}"
BENCH_WARMUP_RATIO="${BENCH_WARMUP_RATIO:-0.1}"
ENABLE_KV_TRACE="${ENABLE_KV_TRACE:-1}"

clear_result_dir_if_needed() {
  if [[ "${CLEAR_RESULTS}" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "${RESULT_DIR}" ]]; then
    return 0
  fi

  local resolved_result_dir
  resolved_result_dir="$(cd "${RESULT_DIR}" 2>/dev/null && pwd -P || true)"
  if [[ -z "${resolved_result_dir}" || "${resolved_result_dir}" == "/" ]]; then
    echo "Refusing to clear RESULT_DIR='${RESULT_DIR}' (resolved='${resolved_result_dir}')"
    return 1
  fi

  echo "CLEAR_RESULTS=1; removing previous benchmark outputs in ${resolved_result_dir}"
  # Keep local helper scripts if user stores them in RESULT_DIR.
  find "${resolved_result_dir}" -mindepth 1 -maxdepth 1 ! -name "*.py" -exec rm -rf {} +
}

clear_result_dir_if_needed
mkdir -p "$RESULT_DIR" "$JSON_RESULT_DIR" "$SUMMARY_DIR" "$PLOT_DIR" "$SERVER_LOG_DIR" "$METRICS_DIR"

SERVER_PID=""
SERVER_LOG_OFF=""
SERVER_LOG_ON=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

wait_server_ready() {
  local deadline=$((SECONDS + WAIT_TIMEOUT_SEC))
  while ((SECONDS < deadline)); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Server not ready within ${WAIT_TIMEOUT_SEC}s."
  return 1
}

# ---- metrics auto capture (off/on) ----
METRICS_GREP_RE='vllm:page_eviction_num|page_eviction|paged_eviction|replace_block|score_collect|compress|kv_cache_budget'

metrics_url() {
  # Prefer BASE_URL/PORT if available; fallback to 8000.
  if [[ -n "${BASE_URL:-}" ]]; then
    echo "${BASE_URL%/}/metrics"
    return
  fi
  local port="${PORT:-8000}"
  local host="${METRICS_HOST:-127.0.0.1}"
  echo "http://${host}:${port}/metrics"
}

capture_metrics() {
  # usage: capture_metrics <tag:off|on> <phase:start|end>
  local tag="$1"
  local phase="$2"
  local out="${METRICS_DIR}/metrics_${tag}_${phase}.prom"
  local url
  url="$(metrics_url)"

  # Do not fail benchmark if metrics endpoint is flaky.
  if ! curl -sf "${url}" > "${out}"; then
    echo "metrics_fetch_failed url=${url} tag=${tag} phase=${phase}" > "${out}"
    return 0
  fi
}

diff_metrics() {
  # usage: diff_metrics <tag:off|on>
  local tag="$1"
  local start="${METRICS_DIR}/metrics_${tag}_start.prom"
  local end="${METRICS_DIR}/metrics_${tag}_end.prom"
  local diff_out="${METRICS_DIR}/metrics_${tag}_diff.txt"
  local start_filtered=""
  local end_filtered=""

  if [[ ! -f "${start}" || ! -f "${end}" ]]; then
    echo "metrics_diff_failed missing_start_or_end tag=${tag}" > "${diff_out}"
    return 0
  fi

  if grep -q "metrics_fetch_failed" "${start}" || grep -q "metrics_fetch_failed" "${end}"; then
    echo "metrics_diff_skipped metrics_fetch_failed tag=${tag}" > "${diff_out}"
    return 0
  fi

  start_filtered="$(mktemp)"
  end_filtered="$(mktemp)"
  grep -E "${METRICS_GREP_RE}" "${start}" | sort > "${start_filtered}" || true
  grep -E "${METRICS_GREP_RE}" "${end}" | sort > "${end_filtered}" || true

  diff -u "${start_filtered}" "${end_filtered}" > "${diff_out}" || true
  rm -f "${start_filtered}" "${end_filtered}"
}

compute_metrics_delta() {
  # usage: compute_metrics_delta <tag:off|on>
  local tag="$1"
  local start="${METRICS_DIR}/metrics_${tag}_start.prom"
  local end="${METRICS_DIR}/metrics_${tag}_end.prom"
  local out="${METRICS_DIR}/metrics_${tag}_delta.csv"

  if [[ ! -f "${start}" || ! -f "${end}" ]]; then
    {
      echo "metric_with_labels,start,end,delta"
      echo "metrics_delta_failed_missing_start_or_end_${tag},0,0,0"
    } > "${out}"
    return 0
  fi

  if grep -q "metrics_fetch_failed" "${start}" || grep -q "metrics_fetch_failed" "${end}"; then
    {
      echo "metric_with_labels,start,end,delta"
      echo "metrics_fetch_failed_${tag},0,0,0"
    } > "${out}"
    return 0
  fi

  python - "${start}" "${end}" "${out}" "${METRICS_GREP_RE}" "${tag}" <<'PY' || {
import csv
import re
import sys
from pathlib import Path

start_path = Path(sys.argv[1])
end_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
grep_re = re.compile(sys.argv[4])
tag = sys.argv[5]

line_re = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)$'
)


def parse_prom(path: Path):
    out = {}
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not grep_re.search(line):
            continue
        m = line_re.match(line)
        if not m:
            continue
        name = m.group(1)
        labels = m.group(2) or ""
        value = float(m.group(3))
        out[name + labels] = value
    return out


s = parse_prom(start_path)
e = parse_prom(end_path)
rows = []
for key in sorted(set(s.keys()) | set(e.keys())):
    sv = s.get(key, 0.0)
    ev = e.get(key, 0.0)
    dv = ev - sv
    if dv != 0.0:
        rows.append((key, sv, ev, dv))

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric_with_labels", "start", "end", "delta"])
    w.writerows(rows)

key_prefixes = [
    "vllm:page_eviction_num_prefill_reqs_scheduled",
    "vllm:page_eviction_num_prefill_reqs_query_len_gt_budget",
    "vllm:page_eviction_num_prefill_compress_invocations",
    "vllm:page_eviction_num_replace_block_req_ids",
    "vllm:page_eviction_num_score_collect_calls_ubatch_list",
    "vllm:page_eviction_num_score_collect_return_none_ubatch_list",
    "vllm:page_eviction_num_prefill_block_scores_returned",
    "vllm:page_eviction_num_decode_token_scores_returned",
]

print(f"[metrics-delta] tag={tag} saved={out_path}")
for prefix in key_prefixes:
    delta_total = sum(dv for (k, _, _, dv) in rows if k.startswith(prefix))
    if delta_total != 0.0:
        print(f"  {prefix} delta={delta_total}")
PY
    echo "metric_with_labels,start,end,delta" > "${out}"
    echo "metrics_delta_python_failed_${tag},0,0,0" >> "${out}"
    return 0
  }
}

compute_metrics_ab_delta() {
  # usage: compute_metrics_ab_delta
  local off_csv="${METRICS_DIR}/metrics_off_delta.csv"
  local on_csv="${METRICS_DIR}/metrics_on_delta.csv"
  local out_csv="${METRICS_DIR}/metrics_ab_delta.csv"

  if [[ ! -f "${off_csv}" || ! -f "${on_csv}" ]]; then
    {
      echo "metric_with_labels,off_delta,on_delta,ab_delta_on_minus_off"
      echo "metrics_ab_delta_failed_missing_off_or_on,0,0,0"
    } > "${out_csv}"
    return 0
  fi

  python - "${off_csv}" "${on_csv}" "${out_csv}" <<'PY' || {
import csv
import sys
from pathlib import Path

off_csv = Path(sys.argv[1])
on_csv = Path(sys.argv[2])
out_csv = Path(sys.argv[3])


def load(path: Path):
    data = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("metric_with_labels", "")
            if not key:
                continue
            try:
                data[key] = float(row.get("delta", "0") or 0.0)
            except ValueError:
                data[key] = 0.0
    return data


off = load(off_csv)
on = load(on_csv)
rows = []
for key in sorted(set(off.keys()) | set(on.keys())):
    off_delta = off.get(key, 0.0)
    on_delta = on.get(key, 0.0)
    ab_delta = on_delta - off_delta
    if off_delta != 0.0 or on_delta != 0.0 or ab_delta != 0.0:
        rows.append((key, off_delta, on_delta, ab_delta))

with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric_with_labels", "off_delta", "on_delta", "ab_delta_on_minus_off"])
    w.writerows(rows)
PY
    echo "metric_with_labels,off_delta,on_delta,ab_delta_on_minus_off" > "${out_csv}"
    echo "metrics_ab_delta_python_failed,0,0,0" >> "${out_csv}"
    return 0
  }
}

append_metrics_delta_to_summary() {
  # usage: append_metrics_delta_to_summary <summary_csv>
  local summary_csv="$1"
  local off_delta_csv="${METRICS_DIR}/metrics_off_delta.csv"
  local on_delta_csv="${METRICS_DIR}/metrics_on_delta.csv"

  if [[ -z "${summary_csv}" || ! -f "${summary_csv}" ]]; then
    echo "skip append metrics delta: summary CSV missing"
    return 0
  fi
  if [[ ! -f "${off_delta_csv}" || ! -f "${on_delta_csv}" ]]; then
    echo "skip append metrics delta: missing off/on metrics delta CSV"
    return 0
  fi

  python - "${summary_csv}" "${off_delta_csv}" "${on_delta_csv}" <<'PY' || true
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
off_delta = Path(sys.argv[2])
on_delta = Path(sys.argv[3])


def load_delta(path: Path):
    data = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("metric_with_labels", "")
            if not key:
                continue
            try:
                data[key] = float(row.get("delta", "0") or 0.0)
            except ValueError:
                data[key] = 0.0
    return data


off = load_delta(off_delta)
on = load_delta(on_delta)

prefixes = [
    "vllm:page_eviction_num_prefill_reqs_scheduled",
    "vllm:page_eviction_num_prefill_reqs_query_len_gt_budget",
    "vllm:page_eviction_num_prefill_compress_invocations",
    "vllm:page_eviction_num_replace_block_req_ids",
    "vllm:page_eviction_num_score_collect_calls_ubatch_list",
    "vllm:page_eviction_num_score_collect_return_none_ubatch_list",
    "vllm:page_eviction_num_prefill_block_scores_returned",
    "vllm:page_eviction_num_decode_token_scores_returned",
]


def sum_prefix(mapping, prefix):
    return sum(v for (k, v) in mapping.items() if k.startswith(prefix))


rows = []
for prefix in prefixes:
    off_v = sum_prefix(off, prefix)
    on_v = sum_prefix(on, prefix)
    rows.append([prefix + ".delta_total", off_v, on_v, on_v - off_v])

off_sched = sum_prefix(off, "vllm:page_eviction_num_prefill_reqs_scheduled")
off_gt = sum_prefix(off, "vllm:page_eviction_num_prefill_reqs_query_len_gt_budget")
on_sched = sum_prefix(on, "vllm:page_eviction_num_prefill_reqs_scheduled")
on_gt = sum_prefix(on, "vllm:page_eviction_num_prefill_reqs_query_len_gt_budget")
off_ratio = (off_gt / off_sched) if off_sched > 0 else 0.0
on_ratio = (on_gt / on_sched) if on_sched > 0 else 0.0
rows.append(["pe.prefill_query_len_gt_budget_ratio", off_ratio, on_ratio, on_ratio - off_ratio])

off_inv = sum_prefix(off, "vllm:page_eviction_num_prefill_compress_invocations")
on_inv = sum_prefix(on, "vllm:page_eviction_num_prefill_compress_invocations")
off_den = off_gt if off_gt > 0 else 1.0
on_den = on_gt if on_gt > 0 else 1.0
off_per_req = off_inv / off_den
on_per_req = on_inv / on_den
rows.append([
    "pe.compress_invocations_per_querylen_gt_budget_req",
    off_per_req,
    on_per_req,
    on_per_req - off_per_req,
])

with summary.open("a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["metric(metrics_delta)", "off", "on", "delta(on-off)"])
    writer.writerows(rows)
PY
}
# ---- metrics auto capture (off/on) ----

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
  fi
  SERVER_PID=""
}

start_server() {
  local pe_mode="$1"
  stop_server

  local log_file="${SERVER_LOG_DIR}/vllm_${pe_mode}_$(date +%Y%m%d-%H%M%S).log"
  set -- \
    vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    -tp "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE"

  if [[ "$ENABLE_EAGER" == "1" ]]; then
    set -- "$@" --enforce-eager
  fi

  if [[ "$USE_SPEC_DECODE" == "1" ]]; then
    set -- "$@" \
      --speculative-config \
      "{\"method\":\"draft_model\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}"
  fi

  if [[ "$pe_mode" == "on" ]]; then
    set -- "$@" --enable-paged-eviction --kv-cache-budget "$KV_CACHE_BUDGET"
    if [[ "$PE_CHUNKED_PREFILL" == "1" ]]; then
      set -- "$@" --enable-chunked-prefill-with-paged-eviction
    fi
  fi

  echo "Launching vLLM (${pe_mode}) ..."
  "$@" >"$log_file" 2>&1 &
  SERVER_PID=$!

  if [[ "$pe_mode" == "off" ]]; then
    SERVER_LOG_OFF="$log_file"
  else
    SERVER_LOG_ON="$log_file"
  fi

  wait_server_ready
  echo "Server ready (${pe_mode}), log: $log_file"
}

run_one_mode() {
  local pe_mode="$1"
  local spec_tag="off"
  if [[ "$USE_SPEC_DECODE" == "1" ]]; then
    spec_tag="draft_model"
  fi

  local model_base
  model_base="$(basename "$MODEL")"
  local stem="${model_base}-pe_${pe_mode}-spec_${spec_tag}-$(date +%Y%m%d-%H%M%S)"

  echo "Running benchmark: PE=${pe_mode}, SPEC=${spec_tag}"
  (
    export MODEL="$MODEL"
    export HOST="127.0.0.1"
    export PORT="$PORT"
    export NUM_PROMPTS="$BENCH_NUM_PROMPTS"
    export REQUEST_RATE="$BENCH_REQUEST_RATE"
    export OUTPUT_LEN="$BENCH_OUTPUT_LEN"
    export DATASET_PATH="$BENCH_DATASET_PATH"
    export GPU_IDS="$BENCH_GPU_IDS"
    export GPU_MONITOR_INTERVAL="$BENCH_GPU_MONITOR_INTERVAL"
    export WARMUP_RATIO="$BENCH_WARMUP_RATIO"
    export RESULT_DIR="$JSON_RESULT_DIR"
    export PE_MODE_TAG="$pe_mode"
    export SPEC_MODE_TAG="$spec_tag"
    export ENABLE_KV_TRACE="$ENABLE_KV_TRACE"
    export RESULT_STEM_OVERRIDE="$stem"
    "${ROOT_DIR}/run_benchmark.sh"
  )
  echo "${JSON_RESULT_DIR}/${stem}.json"
}

RESULT_OFF=""
RESULT_ON=""

if [[ "$START_SERVER" == "1" ]]; then
  start_server "off"
  capture_metrics "off" "start"
  RESULT_OFF="$(run_one_mode "off" | tail -n 1)"
  capture_metrics "off" "end"
  diff_metrics "off"
  compute_metrics_delta "off"

  start_server "on"
  capture_metrics "on" "start"
  RESULT_ON="$(run_one_mode "on" | tail -n 1)"
  capture_metrics "on" "end"
  diff_metrics "on"
  compute_metrics_delta "on"
  compute_metrics_ab_delta
else
  echo "START_SERVER=0 is not supported in this helper. Use START_SERVER=1."
  exit 1
fi

stop_server

AB_SUMMARY_OUTPUT="$(
python - "$RESULT_OFF" "$RESULT_ON" "$USE_SPEC_DECODE" "$SUMMARY_DIR" "$SERVER_LOG_OFF" "$SERVER_LOG_ON" "$KV_CACHE_BUDGET" <<'PY'
import json
import csv
import sys
import re
from pathlib import Path
from datetime import datetime

off_path = Path(sys.argv[1])
on_path = Path(sys.argv[2])
use_spec_decode = sys.argv[3] == "1"
result_dir = Path(sys.argv[4])
off_log_path = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
on_log_path = Path(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
kv_cache_budget = None
if len(sys.argv) > 7 and str(sys.argv[7]).strip():
    try:
        kv_cache_budget = int(sys.argv[7])
    except ValueError:
        kv_cache_budget = None

off = json.loads(off_path.read_text())
on = json.loads(on_path.read_text())

def get(d, path, default=0.0):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:.6g}"
    return str(v)

def delta_pct(base, new):
    if base is None or new is None:
        return "n/a"
    if base == 0:
        return "n/a"
    return f"{(new - base) / base * 100:+.2f}%"

def mean(values):
    if not values:
        return None
    return sum(values) / len(values)

def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    arr = sorted(values)
    rank = (len(arr) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(arr) - 1)
    frac = rank - low
    return arr[low] * (1.0 - frac) + arr[high] * frac


def parse_max_num_batched_tokens(log_path):
    if log_path is None or not log_path.exists():
        return None
    pattern = re.compile(r"max_num_batched_tokens=(\d+)")
    try:
        for line in log_path.read_text(errors="ignore").splitlines():
            m = pattern.search(line)
            if m:
                return int(m.group(1))
    except Exception:
        return None
    return None

def parse_engine_log_stats(log_path):
    stats = {
        "oom_count": 0,
        "engine_restart_count": 0,
    }
    if log_path is None or not log_path.exists():
        return stats
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return stats

    stats["oom_count"] = len(
        re.findall(r"(?:oom|out of memory|cuda error: out of memory)", text, re.IGNORECASE)
    )
    init_count = len(re.findall(r"Initializing a V1 LLM engine", text))
    if init_count <= 0:
        init_count = len(re.findall(r"init engine .* took", text))
    stats["engine_restart_count"] = max(init_count - 1, 0)
    return stats

def success_summary(result):
    num_prompts = int(result.get("num_prompts", 0))
    completed = int(result.get("completed", 0))
    if num_prompts <= 0:
        num_prompts = len(result.get("output_lens", []))
    if completed <= 0 and isinstance(result.get("errors"), list):
        completed = sum(1 for e in result.get("errors", []) if not e)
    completed = max(0, min(completed, num_prompts)) if num_prompts > 0 else max(completed, 0)
    failed = max(num_prompts - completed, 0) if num_prompts > 0 else 0
    return {
        "num_prompts": num_prompts,
        "completed": completed,
        "failed": failed,
        "success_rate": (completed / num_prompts) if num_prompts > 0 else None,
    }


def latency_stats(result, *, success_only=False, min_output_len_for_tpot=2, indices=None):
    output_lens = result.get("output_lens", [])
    ttfts = result.get("ttfts", [])
    e2els = result.get("e2els", [])
    errors = result.get("errors", [])

    n = min(len(output_lens), len(ttfts), len(e2els))
    has_error_list = isinstance(errors, list) and len(errors) >= n
    if indices is None:
        indices = range(n)

    ttft_ms = []
    e2e_ms = []
    tpot_ms = []
    decode_time_s_sum = 0.0
    decode_token_count = 0
    request_count = 0
    for idx in indices:
        if idx < 0 or idx >= n:
            continue
        if success_only and has_error_list and errors[idx]:
            continue
        request_count += 1

        out_len = output_lens[idx]
        ttft_s = ttfts[idx]
        e2e_s = e2els[idx]
        if out_len and out_len > 0:
            ttft_ms.append(ttft_s * 1000.0)
            e2e_ms.append(e2e_s * 1000.0)
        if out_len and out_len >= min_output_len_for_tpot:
            decode_time_s = e2e_s - ttft_s
            token_count = out_len - 1
            tpot_ms.append(decode_time_s / token_count * 1000.0)
            decode_time_s_sum += decode_time_s
            decode_token_count += token_count

    return {
        "mean_output_len": mean([
            output_lens[i] for i in indices if 0 <= i < n
        ]) if n > 0 else None,
        "p50_output_len": percentile([
            output_lens[i] for i in indices if 0 <= i < n
        ], 50) if n > 0 else None,
        "p90_output_len": percentile([
            output_lens[i] for i in indices if 0 <= i < n
        ], 90) if n > 0 else None,
        "p99_output_len": percentile([
            output_lens[i] for i in indices if 0 <= i < n
        ], 99) if n > 0 else None,
        "mean_ttft_ms": mean(ttft_ms),
        "p50_ttft_ms": percentile(ttft_ms, 50),
        "p90_ttft_ms": percentile(ttft_ms, 90),
        "p99_ttft_ms": percentile(ttft_ms, 99),
        "mean_e2e_latency_ms": mean(e2e_ms),
        "p50_e2e_latency_ms": percentile(e2e_ms, 50),
        "p90_e2e_latency_ms": percentile(e2e_ms, 90),
        "p99_e2e_latency_ms": percentile(e2e_ms, 99),
        "mean_tpot_ms": mean(tpot_ms),
        "p50_tpot_ms": percentile(tpot_ms, 50),
        "p90_tpot_ms": percentile(tpot_ms, 90),
        "weighted_tpot_ms": (
            (decode_time_s_sum / decode_token_count) * 1000.0
            if decode_token_count > 0 else None
        ),
        "p99_tpot_ms": percentile(tpot_ms, 99),
        "request_count": request_count,
        "tpot_sample_count": len(tpot_ms),
    }

def finish_reason_counts(result):
    if isinstance(result.get("finish_reason_counts"), dict):
        return result["finish_reason_counts"]
    finish_reasons = result.get("finish_reasons", [])
    stop_reasons = result.get("stop_reasons", [])
    errors = result.get("errors", [])
    n = min(len(finish_reasons), len(errors))
    counts = {"eos": 0, "length": 0, "stop": 0, "error": 0, "abort": 0, "unknown": 0}
    for i in range(n):
        fr = str(finish_reasons[i]).lower() if finish_reasons[i] is not None else ""
        sr = str(stop_reasons[i]).lower() if i < len(stop_reasons) and stop_reasons[i] is not None else ""
        err = errors[i]
        if err:
            counts["error"] += 1
        elif fr == "length":
            counts["length"] += 1
        elif fr == "error":
            counts["error"] += 1
        elif fr == "abort":
            counts["abort"] += 1
        elif fr == "eos" or "eos" in sr:
            counts["eos"] += 1
        elif fr == "stop":
            counts["stop"] += 1
        else:
            counts["unknown"] += 1
    return counts

def output_len_distribution(result):
    if isinstance(result.get("output_len_distribution"), dict):
        return result["output_len_distribution"]
    output_lens = result.get("output_lens", [])
    if not output_lens:
        return {}
    return {
        "mean": mean(output_lens),
        "p50": percentile(output_lens, 50),
        "p90": percentile(output_lens, 90),
        "p99": percentile(output_lens, 99),
        "ratio_out_len_lt_16": sum(1 for x in output_lens if x < 16) / len(output_lens),
        "ratio_out_len_lt_64": sum(1 for x in output_lens if x < 64) / len(output_lens),
        "ratio_out_len_lt_200": sum(1 for x in output_lens if x < 200) / len(output_lens),
    }

def memory_stats(result, log_stats):
    mem = result.get("memory_stats", {})
    peak_alloc = mem.get("peak_allocated_mb")
    peak_reserved = mem.get("peak_reserved_mb")
    if peak_alloc is None:
        gpu_power_stats = result.get("gpu_power_stats", {})
        peak_alloc = 0.0
        for gpu_stats in gpu_power_stats.values():
            trace = gpu_stats.get("memory_used_mb_trace") or []
            if trace:
                peak_alloc += max(trace)
        if peak_alloc == 0.0:
            peak_alloc = None
    if peak_reserved is None:
        peak_reserved = peak_alloc
    oom_count = mem.get("oom_count", 0)
    if not oom_count:
        oom_count = log_stats.get("oom_count", 0)
    restart_count = mem.get("engine_restart_count", 0)
    if not restart_count:
        restart_count = log_stats.get("engine_restart_count", 0)
    return {
        "peak_allocated_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
        "oom_count": oom_count,
        "engine_restart_count": restart_count,
    }

def normalize_bucket_stats(stats):
    if not isinstance(stats, dict):
        return {}
    out = dict(stats)
    alias_pairs = [
        ("ttft_ms_p50", "p50_ttft_ms"),
        ("ttft_ms_p90", "p90_ttft_ms"),
        ("ttft_ms_p99", "p99_ttft_ms"),
        ("e2e_ms_p50", "p50_e2e_latency_ms"),
        ("e2e_ms_p90", "p90_e2e_latency_ms"),
        ("e2e_ms_p99", "p99_e2e_latency_ms"),
        ("tpot_ms_p50", "p50_tpot_ms"),
        ("tpot_ms_p90", "p90_tpot_ms"),
        ("tpot_ms_p99", "p99_tpot_ms"),
    ]
    for canonical, legacy in alias_pairs:
        if canonical in out and legacy not in out:
            out[legacy] = out[canonical]
        elif legacy in out and canonical not in out:
            out[canonical] = out[legacy]
    return out

def prompt_len_bucket_stats(result, budget):
    stats = result.get("prompt_len_bucket_stats")
    if isinstance(stats, dict):
        return (
            normalize_bucket_stats(stats.get("prompt_len_le_budget", {})),
            normalize_bucket_stats(stats.get("prompt_len_gt_budget", {})),
            stats.get("budget", budget),
        )
    if budget is None or budget <= 0:
        return {}, {}, budget
    input_lens = result.get("input_lens", [])
    le_idx = [i for i, x in enumerate(input_lens) if x <= budget]
    gt_idx = [i for i, x in enumerate(input_lens) if x > budget]
    le = latency_stats(result, indices=le_idx)
    gt = latency_stats(result, indices=gt_idx)
    le_200 = latency_stats(result, min_output_len_for_tpot=200, indices=le_idx)
    gt_200 = latency_stats(result, min_output_len_for_tpot=200, indices=gt_idx)
    le["weighted_tpot_ms_out>=200"] = le_200.get("weighted_tpot_ms")
    gt["weighted_tpot_ms_out>=200"] = gt_200.get("weighted_tpot_ms")
    return (
        normalize_bucket_stats(le),
        normalize_bucket_stats(gt),
        budget,
    )

off_success = success_summary(off)
on_success = success_summary(on)
off_lat = latency_stats(off)
on_lat = latency_stats(on)
off_lat64 = latency_stats(off, min_output_len_for_tpot=64)
on_lat64 = latency_stats(on, min_output_len_for_tpot=64)
off_lat200 = latency_stats(off, min_output_len_for_tpot=200)
on_lat200 = latency_stats(on, min_output_len_for_tpot=200)
off_lat_success = latency_stats(off, success_only=True)
on_lat_success = latency_stats(on, success_only=True)
off_max_num_batched_tokens = parse_max_num_batched_tokens(off_log_path)
on_max_num_batched_tokens = parse_max_num_batched_tokens(on_log_path)
off_log_stats = parse_engine_log_stats(off_log_path)
on_log_stats = parse_engine_log_stats(on_log_path)
off_output_stats = output_len_distribution(off)
on_output_stats = output_len_distribution(on)
off_finish_counts = finish_reason_counts(off)
on_finish_counts = finish_reason_counts(on)
off_mem = memory_stats(off, off_log_stats)
on_mem = memory_stats(on, on_log_stats)
off_bucket_le, off_bucket_gt, budget_used = prompt_len_bucket_stats(off, kv_cache_budget)
on_bucket_le, on_bucket_gt, _ = prompt_len_bucket_stats(on, kv_cache_budget)

rows = [
    {"name": "request_throughput(req/s)", "path": "request_throughput"},
    {"name": "output_throughput(tok/s)", "path": "output_throughput"},
    {"name": "total_token_throughput(tok/s)", "path": "total_token_throughput"},
    {"name": "success.rate", "off": off_success["success_rate"], "on": on_success["success_rate"]},
    {"name": "success.completed", "off": off_success["completed"], "on": on_success["completed"]},
    {"name": "success.failed", "off": off_success["failed"], "on": on_success["failed"]},
    {"name": "mean_ttft_ms", "off": off_lat["mean_ttft_ms"], "on": on_lat["mean_ttft_ms"]},
    {"name": "p99_ttft_ms", "off": off_lat["p99_ttft_ms"], "on": on_lat["p99_ttft_ms"]},
    {"name": "mean_e2e_latency_ms", "off": off_lat["mean_e2e_latency_ms"], "on": on_lat["mean_e2e_latency_ms"]},
    {"name": "p99_e2e_latency_ms", "off": off_lat["p99_e2e_latency_ms"], "on": on_lat["p99_e2e_latency_ms"]},
    {"name": "mean_tpot_ms", "off": off_lat["mean_tpot_ms"], "on": on_lat["mean_tpot_ms"]},
    {"name": "weighted_tpot_ms", "off": off_lat["weighted_tpot_ms"], "on": on_lat["weighted_tpot_ms"]},
    {"name": "weighted_tpot_ms_out>=200", "off": off_lat200["weighted_tpot_ms"], "on": on_lat200["weighted_tpot_ms"]},
    {"name": "p99_tpot_ms", "off": off_lat["p99_tpot_ms"], "on": on_lat["p99_tpot_ms"]},
    {"name": "mean_tpot_ms_out>=64", "off": off_lat64["mean_tpot_ms"], "on": on_lat64["mean_tpot_ms"]},
    {"name": "weighted_tpot_ms_out>=64", "off": off_lat64["weighted_tpot_ms"], "on": on_lat64["weighted_tpot_ms"]},
    {"name": "p99_tpot_ms_out>=64", "off": off_lat64["p99_tpot_ms"], "on": on_lat64["p99_tpot_ms"]},
    {"name": "tpot_samples_out>=64", "off": off_lat64["tpot_sample_count"], "on": on_lat64["tpot_sample_count"]},
    {"name": "success_only.mean_tpot_ms", "off": off_lat_success["mean_tpot_ms"], "on": on_lat_success["mean_tpot_ms"]},
    {"name": "success_only.weighted_tpot_ms", "off": off_lat_success["weighted_tpot_ms"], "on": on_lat_success["weighted_tpot_ms"]},
    {"name": "success_only.p99_tpot_ms", "off": off_lat_success["p99_tpot_ms"], "on": on_lat_success["p99_tpot_ms"]},
    {"name": "success_only.tpot_samples", "off": off_lat_success["tpot_sample_count"], "on": on_lat_success["tpot_sample_count"]},
    {"name": "energy.total_energy(J)", "path": "energy_stats.total_energy"},
    {"name": "energy.energy_per_request(J)", "path": "energy_stats.energy_per_request"},
    {"name": "energy.energy_per_token(J)", "path": "energy_stats.energy_per_token"},
    {"name": "output_len.mean", "off": off_output_stats.get("mean"), "on": on_output_stats.get("mean")},
    {"name": "output_len.p50", "off": off_output_stats.get("p50"), "on": on_output_stats.get("p50")},
    {"name": "output_len.p90", "off": off_output_stats.get("p90"), "on": on_output_stats.get("p90")},
    {"name": "output_len.p99", "off": off_output_stats.get("p99"), "on": on_output_stats.get("p99")},
    {"name": "output_len.ratio_lt_16", "off": off_output_stats.get("ratio_out_len_lt_16"), "on": on_output_stats.get("ratio_out_len_lt_16")},
    {"name": "output_len.ratio_lt_64", "off": off_output_stats.get("ratio_out_len_lt_64"), "on": on_output_stats.get("ratio_out_len_lt_64")},
    {"name": "output_len.ratio_lt_200", "off": off_output_stats.get("ratio_out_len_lt_200"), "on": on_output_stats.get("ratio_out_len_lt_200")},
    {"name": "finish_reason.eos", "off": off_finish_counts.get("eos"), "on": on_finish_counts.get("eos")},
    {"name": "finish_reason.length", "off": off_finish_counts.get("length"), "on": on_finish_counts.get("length")},
    {"name": "finish_reason.stop", "off": off_finish_counts.get("stop"), "on": on_finish_counts.get("stop")},
    {"name": "finish_reason.error", "off": off_finish_counts.get("error"), "on": on_finish_counts.get("error")},
    {
        "name": "prefill.compress_calls_per_request.mean",
        "path": "page_eviction_monitoring_stats.prefill_compress_calls_per_request_mean",
        "default": None,
    },
    {
        "name": "prefill.compress_calls_per_request.p99",
        "path": "page_eviction_monitoring_stats.prefill_compress_calls_per_request_p99",
        "default": None,
    },
    {
        "name": "prefill.compress_calls_per_request.max",
        "path": "page_eviction_monitoring_stats.prefill_compress_calls_per_request_max",
        "default": None,
    },
    {"name": "prefill.compress_time_ms_total", "path": "page_eviction_monitoring_stats.prefill_compress_time_ms_total", "default": None},
    {"name": "prefill.compress_time_ms_per_event.p50", "path": "page_eviction_monitoring_stats.prefill_compress_time_ms_per_event_p50", "default": None},
    {"name": "prefill.compress_time_ms_per_event.p90", "path": "page_eviction_monitoring_stats.prefill_compress_time_ms_per_event_p90", "default": None},
    {"name": "prefill.compress_time_ms_per_event.p99", "path": "page_eviction_monitoring_stats.prefill_compress_time_ms_per_event_p99", "default": None},
    {"name": "prefill.keep_len.mean", "path": "page_eviction_monitoring_stats.prefill_keep_len_mean", "default": None},
    {"name": "prefill.keep_len.p90", "path": "page_eviction_monitoring_stats.prefill_keep_len_p90", "default": None},
    {"name": "prefill.kept_ratio.mean", "path": "page_eviction_monitoring_stats.prefill_kept_ratio_mean", "default": None},
    {"name": "prefill.kept_ratio.p90", "path": "page_eviction_monitoring_stats.prefill_kept_ratio_p90", "default": None},
    {
        "name": "max_num_batched_tokens.actual",
        "off": off_max_num_batched_tokens,
        "on": on_max_num_batched_tokens,
    },
    {"name": "pe.total_eviction_ops", "path": "page_eviction_monitoring_stats.total_eviction_ops"},
    {"name": "pe.total_evicted_blocks", "path": "page_eviction_monitoring_stats.total_evicted_blocks"},
    {"name": "pe.prefill_reqs_scheduled.total", "path": "page_eviction_monitoring_stats.total_prefill_reqs_scheduled", "default": None},
    {"name": "pe.prefill_reqs_query_len_gt_budget.total", "path": "page_eviction_monitoring_stats.total_prefill_reqs_query_len_gt_budget", "default": None},
    {"name": "pe.prefill_reqs_query_len_gt_budget.ratio", "path": "page_eviction_monitoring_stats.prefill_query_len_gt_budget_ratio", "default": None},
    {"name": "pe.replace_block_req_ids.total", "path": "page_eviction_monitoring_stats.total_replace_block_req_ids", "default": None},
    {"name": "pe.replace_block_req_ids_per_sample.p50", "path": "page_eviction_monitoring_stats.replace_block_req_ids_count_per_sample_p50", "default": None},
    {"name": "pe.replace_block_req_ids_per_sample.p90", "path": "page_eviction_monitoring_stats.replace_block_req_ids_count_per_sample_p90", "default": None},
    {"name": "pe.replace_block_req_ids_per_sample.p99", "path": "page_eviction_monitoring_stats.replace_block_req_ids_count_per_sample_p99", "default": None},
    {"name": "pe.score_collect.calls_single.total", "path": "page_eviction_monitoring_stats.total_score_collect_calls_single", "default": None},
    {"name": "pe.score_collect.calls_ubatch_list.total", "path": "page_eviction_monitoring_stats.total_score_collect_calls_ubatch_list", "default": None},
    {"name": "pe.score_collect.return_none_ubatch_list.total", "path": "page_eviction_monitoring_stats.total_score_collect_return_none_ubatch_list", "default": None},
    {"name": "pe.score_collect.return_none_ubatch_ratio", "path": "page_eviction_monitoring_stats.score_collect_return_none_ubatch_ratio", "default": None},
    {"name": "pe.prefill_block_scores_returned.total", "path": "page_eviction_monitoring_stats.total_prefill_block_scores_returned", "default": None},
    {"name": "pe.decode_token_scores_returned.total", "path": "page_eviction_monitoring_stats.total_decode_token_scores_returned", "default": None},
    {"name": "pe.prefill_compress_invocations.total", "path": "page_eviction_monitoring_stats.total_prefill_compress_invocations", "default": None},
    {"name": "pe.prefill_compress_invocations_per_request.mean", "path": "page_eviction_monitoring_stats.prefill_compress_invocations_per_request_mean", "default": None},
    {"name": "decode.eviction_ops_per_request.p50", "path": "page_eviction_monitoring_stats.decode_eviction_ops_per_request_p50", "default": None},
    {"name": "decode.eviction_ops_per_request.p90", "path": "page_eviction_monitoring_stats.decode_eviction_ops_per_request_p90", "default": None},
    {"name": "decode.eviction_ops_per_request.p99", "path": "page_eviction_monitoring_stats.decode_eviction_ops_per_request_p99", "default": None},
    {"name": "decode.evicted_blocks_per_op.p50", "path": "page_eviction_monitoring_stats.decode_evicted_blocks_per_op_p50", "default": None},
    {"name": "decode.evicted_blocks_per_op.p90", "path": "page_eviction_monitoring_stats.decode_evicted_blocks_per_op_p90", "default": None},
    {"name": "decode.evicted_blocks_per_op.p99", "path": "page_eviction_monitoring_stats.decode_evicted_blocks_per_op_p99", "default": None},
    {"name": "decode.eviction_time_ms_per_op.p50", "path": "page_eviction_monitoring_stats.decode_eviction_time_ms_per_op_p50", "default": None},
    {"name": "decode.eviction_time_ms_per_op.p90", "path": "page_eviction_monitoring_stats.decode_eviction_time_ms_per_op_p90", "default": None},
    {"name": "decode.eviction_time_ms_per_op.p99", "path": "page_eviction_monitoring_stats.decode_eviction_time_ms_per_op_p99", "default": None},
    {"name": "decode.pages_scored_per_op.p50", "path": "page_eviction_monitoring_stats.decode_pages_scored_per_op_p50", "default": None},
    {"name": "decode.pages_scored_per_op.p90", "path": "page_eviction_monitoring_stats.decode_pages_scored_per_op_p90", "default": None},
    {"name": "decode.pages_scored_per_op.p99", "path": "page_eviction_monitoring_stats.decode_pages_scored_per_op_p99", "default": None},
    {"name": "sched.active_concurrency.p50", "path": "page_eviction_monitoring_stats.active_concurrency_p50", "default": None},
    {"name": "sched.active_concurrency.p90", "path": "page_eviction_monitoring_stats.active_concurrency_p90", "default": None},
    {"name": "sched.num_prefill_tokens_scheduled.p50", "path": "page_eviction_monitoring_stats.num_prefill_tokens_scheduled_p50", "default": None},
    {"name": "sched.num_prefill_tokens_scheduled.p90", "path": "page_eviction_monitoring_stats.num_prefill_tokens_scheduled_p90", "default": None},
    {"name": "sched.num_prefill_tokens_scheduled.p99", "path": "page_eviction_monitoring_stats.num_prefill_tokens_scheduled_p99", "default": None},
    {"name": "sched.num_decode_tokens_scheduled.p50", "path": "page_eviction_monitoring_stats.num_decode_tokens_scheduled_p50", "default": None},
    {"name": "sched.num_decode_tokens_scheduled.p90", "path": "page_eviction_monitoring_stats.num_decode_tokens_scheduled_p90", "default": None},
    {"name": "sched.num_decode_tokens_scheduled.p99", "path": "page_eviction_monitoring_stats.num_decode_tokens_scheduled_p99", "default": None},
    {"name": "memory.peak_allocated_mb", "off": off_mem.get("peak_allocated_mb"), "on": on_mem.get("peak_allocated_mb")},
    {"name": "memory.peak_reserved_mb", "off": off_mem.get("peak_reserved_mb"), "on": on_mem.get("peak_reserved_mb")},
    {"name": "memory.oom_count", "off": off_mem.get("oom_count"), "on": on_mem.get("oom_count")},
    {"name": "memory.engine_restart_count", "off": off_mem.get("engine_restart_count"), "on": on_mem.get("engine_restart_count")},
]

if budget_used is not None and budget_used > 0:
    rows += [
        {"name": f"bucket<=budget({budget_used}).weighted_tpot_ms", "off": off_bucket_le.get("weighted_tpot_ms"), "on": on_bucket_le.get("weighted_tpot_ms")},
        {"name": f"bucket<=budget({budget_used}).weighted_tpot_ms_out>=200", "off": off_bucket_le.get("weighted_tpot_ms_out>=200"), "on": on_bucket_le.get("weighted_tpot_ms_out>=200")},
        {"name": f"bucket<=budget({budget_used}).ttft_ms_p50", "off": off_bucket_le.get("p50_ttft_ms"), "on": on_bucket_le.get("p50_ttft_ms")},
        {"name": f"bucket<=budget({budget_used}).ttft_ms_p90", "off": off_bucket_le.get("p90_ttft_ms"), "on": on_bucket_le.get("p90_ttft_ms")},
        {"name": f"bucket<=budget({budget_used}).ttft_ms_p99", "off": off_bucket_le.get("p99_ttft_ms"), "on": on_bucket_le.get("p99_ttft_ms")},
        {"name": f"bucket<=budget({budget_used}).e2e_ms_p50", "off": off_bucket_le.get("p50_e2e_latency_ms"), "on": on_bucket_le.get("p50_e2e_latency_ms")},
        {"name": f"bucket<=budget({budget_used}).e2e_ms_p90", "off": off_bucket_le.get("p90_e2e_latency_ms"), "on": on_bucket_le.get("p90_e2e_latency_ms")},
        {"name": f"bucket<=budget({budget_used}).e2e_ms_p99", "off": off_bucket_le.get("p99_e2e_latency_ms"), "on": on_bucket_le.get("p99_e2e_latency_ms")},
        {"name": f"bucket<=budget({budget_used}).tpot_ms_p50", "off": off_bucket_le.get("p50_tpot_ms"), "on": on_bucket_le.get("p50_tpot_ms")},
        {"name": f"bucket<=budget({budget_used}).tpot_ms_p90", "off": off_bucket_le.get("p90_tpot_ms"), "on": on_bucket_le.get("p90_tpot_ms")},
        {"name": f"bucket<=budget({budget_used}).tpot_ms_p99", "off": off_bucket_le.get("p99_tpot_ms"), "on": on_bucket_le.get("p99_tpot_ms")},
        {"name": f"bucket>budget({budget_used}).weighted_tpot_ms", "off": off_bucket_gt.get("weighted_tpot_ms"), "on": on_bucket_gt.get("weighted_tpot_ms")},
        {"name": f"bucket>budget({budget_used}).weighted_tpot_ms_out>=200", "off": off_bucket_gt.get("weighted_tpot_ms_out>=200"), "on": on_bucket_gt.get("weighted_tpot_ms_out>=200")},
        {"name": f"bucket>budget({budget_used}).ttft_ms_p50", "off": off_bucket_gt.get("p50_ttft_ms"), "on": on_bucket_gt.get("p50_ttft_ms")},
        {"name": f"bucket>budget({budget_used}).ttft_ms_p90", "off": off_bucket_gt.get("p90_ttft_ms"), "on": on_bucket_gt.get("p90_ttft_ms")},
        {"name": f"bucket>budget({budget_used}).ttft_ms_p99", "off": off_bucket_gt.get("p99_ttft_ms"), "on": on_bucket_gt.get("p99_ttft_ms")},
        {"name": f"bucket>budget({budget_used}).e2e_ms_p50", "off": off_bucket_gt.get("p50_e2e_latency_ms"), "on": on_bucket_gt.get("p50_e2e_latency_ms")},
        {"name": f"bucket>budget({budget_used}).e2e_ms_p90", "off": off_bucket_gt.get("p90_e2e_latency_ms"), "on": on_bucket_gt.get("p90_e2e_latency_ms")},
        {"name": f"bucket>budget({budget_used}).e2e_ms_p99", "off": off_bucket_gt.get("p99_e2e_latency_ms"), "on": on_bucket_gt.get("p99_e2e_latency_ms")},
        {"name": f"bucket>budget({budget_used}).tpot_ms_p50", "off": off_bucket_gt.get("p50_tpot_ms"), "on": on_bucket_gt.get("p50_tpot_ms")},
        {"name": f"bucket>budget({budget_used}).tpot_ms_p90", "off": off_bucket_gt.get("p90_tpot_ms"), "on": on_bucket_gt.get("p90_tpot_ms")},
        {"name": f"bucket>budget({budget_used}).tpot_ms_p99", "off": off_bucket_gt.get("p99_tpot_ms"), "on": on_bucket_gt.get("p99_tpot_ms")},
    ]

if use_spec_decode:
    rows.append({
        "name": "spec.acceptance_rate",
        "path": "spec_decode_monitoring_stats.acceptance_rate",
    })
else:
    rows.append({"name": "spec.acceptance_rate", "off": None, "on": None})

rows_by_name = {row["name"]: row for row in rows}

console_metric_names = [
    "request_throughput(req/s)",
    "output_throughput(tok/s)",
    "total_token_throughput(tok/s)",
    "success.rate",
    "mean_ttft_ms",
    "mean_e2e_latency_ms",
    "mean_tpot_ms",
    "weighted_tpot_ms",
    "energy.energy_per_request(J)",
    "energy.energy_per_token(J)",
]
if use_spec_decode:
    console_metric_names.append("spec.acceptance_rate")

console_rows = []
for name in console_metric_names:
    row = rows_by_name.get(name)
    if row is None:
        continue
    if "path" in row:
        default = row.get("default", 0.0)
        v0 = get(off, row["path"], default)
        v1 = get(on, row["path"], default)
    else:
        v0 = row.get("off")
        v1 = row.get("on")
    console_rows.append({
        "name": name,
        "off": fmt(v0),
        "on": fmt(v1),
        "delta": delta_pct(v0, v1),
    })

print("\n=== PageEviction A/B Summary ===")
print(f"PE off result: {off_path}")
print(f"PE on  result: {on_path}")
print("")
print(f"{'metric':35} {'off':>14} {'on':>14} {'delta(on-off)':>14}")
print("-" * 80)

for row in console_rows:
    print(f"{row['name']:35} {row['off']:>14} {row['on']:>14} {row['delta']:>14}")
print("-" * 80)

csv_rows = []
for row in rows:
    name = row["name"]
    if "path" in row:
        default = row.get("default", 0.0)
        v0 = get(off, row["path"], default)
        v1 = get(on, row["path"], default)
    else:
        v0 = row.get("off")
        v1 = row.get("on")
    delta = delta_pct(v0, v1)
    csv_rows.append([name, fmt(v0), fmt(v1), delta])

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_path = result_dir / f"page_eviction_ab_summary_{ts}.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "off", "on", "delta(on-off)"])
    writer.writerows(csv_rows)

print(f"CSV summary: {csv_path}")
print(f"CSV_SUMMARY_PATH={csv_path}")
PY
)"

echo "$AB_SUMMARY_OUTPUT"

CSV_SUMMARY_PATH="$(echo "$AB_SUMMARY_OUTPUT" | awk -F= '/^CSV_SUMMARY_PATH=/{print $2}' | tail -n 1)"
append_metrics_delta_to_summary "$CSV_SUMMARY_PATH"
DRAW_SCRIPT="${DRAW_SCRIPT:-}"
if [[ -z "$DRAW_SCRIPT" ]]; then
  for candidate in \
    "${ROOT_DIR}/draw_compare.py" \
    "${ROOT_DIR}/benchmark_results/page_eviction_ab/draw_compare.py"; do
    if [[ -f "$candidate" ]]; then
      DRAW_SCRIPT="$candidate"
      break
    fi
  done
fi

if [[ -n "$CSV_SUMMARY_PATH" && -n "$DRAW_SCRIPT" && -f "$DRAW_SCRIPT" ]]; then
  PLOT_OUTPUT_PATH="${PLOT_DIR}/$(basename "${CSV_SUMMARY_PATH%.csv}.png")"
  if python "$DRAW_SCRIPT" "$CSV_SUMMARY_PATH" --output "$PLOT_OUTPUT_PATH"; then
    echo "PNG summary: $PLOT_OUTPUT_PATH"
  else
    echo "Warning: Failed to draw summary plot from $CSV_SUMMARY_PATH"
  fi
else
  echo "Warning: CSV summary path or draw script missing; skip plotting."
fi

echo ""
echo "Done. A/B results are saved under: $RESULT_DIR"
echo "JSON results: $JSON_RESULT_DIR"
echo "CSV summaries: $SUMMARY_DIR"
echo "PNG plots: $PLOT_DIR"
