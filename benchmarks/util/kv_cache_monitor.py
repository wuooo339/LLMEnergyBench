import threading
import time
import queue
import numpy as np
import aiohttp
import asyncio
import re
from typing import Dict, Any, Optional


class KVCacheMonitor:
    """
    Monitor KV cache statistics from vLLM server at regular intervals.
    Compatible with GPU/CPU monitoring for DQN state space construction.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", interval: float = 0.5, truncate: float = 0):
        """
        Initialize KV cache monitor.
        
        Args:
            base_url: Base URL of the vLLM server
            interval: Sampling interval in seconds (recommend 0.5-1.0s for KV cache)
                     Note: KV cache metrics update slower than GPU/CPU, so using
                     the same 0.05s interval may not capture meaningful changes
            truncate: Seconds to truncate from beginning and end of monitoring
        """
        self.base_url = base_url
        self.metrics_url = f"{base_url}/metrics"
        self.interval = max(interval, 0.1)  # Minimum 100ms to avoid overwhelming the server
        self.truncate = truncate
        self.done = False
        self.thread = None
        
        # Queues for results
        self.results_queue = queue.Queue()  # Average statistics
        self.stats_queue = queue.Queue()    # Detailed statistics
        self.hist_queue = queue.Queue()     # Time series data
        
        # Static configuration (fetched once)
        self.static_config = {}
        
        # Error tracking
        self.fetch_errors = 0
        self.successful_fetches = 0
        
    def start(self):
        """Start monitoring in background thread."""
        self.done = False
        self.thread = threading.Thread(target=self._monitor_kv_cache)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop monitoring."""
        if self.thread and self.thread.is_alive():
            self.done = True
            self.thread.join(timeout=5)
            
    def _fetch_metrics_sync(self) -> Optional[str]:
        """
        Fetch metrics from server synchronously.
        Returns metrics text or None if failed.
        """
        try:
            import requests
            response = requests.get(self.metrics_url, timeout=3)
            if response.status_code == 200:
                self.successful_fetches += 1
                return response.text
            else:
                self.fetch_errors += 1
        except requests.exceptions.Timeout:
            self.fetch_errors += 1
            # print(f"[KV Monitor] Timeout fetching metrics")
        except requests.exceptions.ConnectionError:
            self.fetch_errors += 1
            # print(f"[KV Monitor] Connection error")
        except Exception as e:
            self.fetch_errors += 1
            # print(f"[KV Monitor] Error: {e}")
        return None
        
    def _parse_metrics(self, metrics_text: str) -> Dict[str, Any]:
        """Parse Prometheus-format metrics."""
        result = {}

        def _sum_counter(metric_name: str) -> float:
            # Prometheus counters are exposed with `_total`.
            pattern = re.compile(
                rf"^{re.escape(metric_name)}_total(?:\{{[^}}]*\}})?\s+([0-9eE+.\-]+)$",
                re.MULTILINE,
            )
            return float(sum(float(m.group(1)) for m in pattern.finditer(metrics_text)))

        def _sum_counter_with_label(
            metric_name: str,
            label_name: str,
            label_value: str,
        ) -> float:
            pattern = re.compile(
                rf'^{re.escape(metric_name)}_total\{{[^}}]*{re.escape(label_name)}="{re.escape(label_value)}"[^}}]*\}}\s+([0-9eE+.\-]+)$',
                re.MULTILINE,
            )
            return float(sum(float(m.group(1)) for m in pattern.finditer(metrics_text)))
        
        # Static configuration patterns
        # Note: In vLLM, all these configs are in cache_config_info labels
        static_patterns = {
            'total_gpu_blocks': r'vllm:cache_config_info\{[^}]*num_gpu_blocks="(\d+)"',
            'block_size': r'vllm:cache_config_info\{[^}]*block_size="(\d+)"',
            'num_layers': r'vllm:cache_config_info\{[^}]*num_layers="(\d+)"',
            'num_kv_heads': r'vllm:cache_config_info\{[^}]*num_kv_heads="(\d+)"',
            'head_size': r'vllm:cache_config_info\{[^}]*head_size="(\d+)"',
        }
        
        # Dynamic statistics patterns
        # Note: vLLM uses 'kv_cache_usage_perc' not 'gpu_cache_usage_perc'
        dynamic_patterns = {
            'cache_usage_perc': r'vllm:kv_cache_usage_perc\{[^}]*\}\s+(\d+\.?\d*)',
            'num_requests_running': r'vllm:num_requests_running\{[^}]*\}\s+(\d+\.?\d*)',
            'num_requests_waiting': r'vllm:num_requests_waiting\{[^}]*\}\s+(\d+\.?\d*)',
            'num_requests_swapped': r'vllm:num_requests_swapped\{[^}]*\}\s+(\d+\.?\d*)',
        }
        
        # Extract static config (only once)
        if not self.static_config:
            for key, pattern in static_patterns.items():
                match = re.search(pattern, metrics_text)
                if match:
                    try:
                        self.static_config[key] = int(match.group(1))
                    except ValueError:
                        self.static_config[key] = match.group(1)
            
            # Calculate derived static values
            if 'block_size' in self.static_config:
                self.static_config['tokens_per_block'] = self.static_config['block_size']
            if 'total_gpu_blocks' in self.static_config and 'tokens_per_block' in self.static_config:
                self.static_config['total_kv_cache_tokens'] = (
                    self.static_config['total_gpu_blocks'] * self.static_config['tokens_per_block']
                )
        
        # Extract dynamic metrics
        for key, pattern in dynamic_patterns.items():
            match = re.search(pattern, metrics_text)
            if match:
                result[key] = float(match.group(1))

        # PageEviction counters (cumulative since server start).
        result['page_eviction_ops_total'] = _sum_counter(
            'vllm:page_eviction_num_eviction_ops'
        )
        result['page_eviction_ops_prefill_total'] = _sum_counter(
            'vllm:page_eviction_num_eviction_ops_prefill'
        )
        result['page_eviction_ops_decode_total'] = _sum_counter(
            'vllm:page_eviction_num_eviction_ops_decode'
        )
        result['page_eviction_blocks_total'] = _sum_counter(
            'vllm:page_eviction_num_evicted_blocks'
        )
        result['page_eviction_blocks_prefill_total'] = _sum_counter(
            'vllm:page_eviction_num_evicted_blocks_prefill'
        )
        result['page_eviction_blocks_decode_total'] = _sum_counter(
            'vllm:page_eviction_num_evicted_blocks_decode'
        )
        result['page_eviction_prefill_reqs_scheduled_total'] = _sum_counter(
            'vllm:page_eviction_num_prefill_reqs_scheduled'
        )
        result['page_eviction_prefill_reqs_query_len_gt_budget_total'] = _sum_counter(
            'vllm:page_eviction_num_prefill_reqs_query_len_gt_budget'
        )
        result['page_eviction_replace_block_req_ids_total'] = _sum_counter(
            'vllm:page_eviction_num_replace_block_req_ids'
        )
        result['page_eviction_score_collect_calls_single_total'] = _sum_counter(
            'vllm:page_eviction_num_score_collect_calls_single'
        )
        result['page_eviction_score_collect_calls_ubatch_list_total'] = _sum_counter(
            'vllm:page_eviction_num_score_collect_calls_ubatch_list'
        )
        result['page_eviction_score_collect_return_none_ubatch_list_total'] = _sum_counter(
            'vllm:page_eviction_num_score_collect_return_none_ubatch_list'
        )
        result['page_eviction_prefill_block_scores_returned_total'] = _sum_counter(
            'vllm:page_eviction_num_prefill_block_scores_returned'
        )
        result['page_eviction_decode_token_scores_returned_total'] = _sum_counter(
            'vllm:page_eviction_num_decode_token_scores_returned'
        )
        result['page_eviction_prefill_compress_invocations_total'] = _sum_counter(
            'vllm:page_eviction_num_prefill_compress_invocations'
        )
        result['request_success_total'] = _sum_counter('vllm:request_success')
        result['request_success_stop_total'] = _sum_counter_with_label(
            'vllm:request_success', 'finished_reason', 'stop'
        )
        result['request_success_length_total'] = _sum_counter_with_label(
            'vllm:request_success', 'finished_reason', 'length'
        )
        result['request_success_abort_total'] = _sum_counter_with_label(
            'vllm:request_success', 'finished_reason', 'abort'
        )
        result['request_success_error_total'] = _sum_counter_with_label(
            'vllm:request_success', 'finished_reason', 'error'
        )
        result['prompt_tokens_total'] = _sum_counter('vllm:prompt_tokens')
        result['generation_tokens_total'] = _sum_counter('vllm:generation_tokens')

        # Optional fine-grained PageEviction counters (if exposed by vLLM build).
        result['page_eviction_prefill_compress_time_seconds_total'] = _sum_counter(
            'vllm:page_eviction_prefill_compress_time_seconds'
        )
        result['page_eviction_prefill_keep_len_tokens_total'] = _sum_counter(
            'vllm:page_eviction_prefill_keep_len_tokens'
        )
        result['page_eviction_prefill_prompt_len_tokens_total'] = _sum_counter(
            'vllm:page_eviction_prefill_prompt_len_tokens'
        )
        result['page_eviction_decode_eviction_time_seconds_total'] = _sum_counter(
            'vllm:page_eviction_decode_eviction_time_seconds'
        )
        result['page_eviction_decode_pages_scored_total'] = _sum_counter(
            'vllm:page_eviction_decode_pages_scored'
        )

        # Spec decode counters (cumulative since server start).
        result['spec_decode_drafts_total'] = _sum_counter('vllm:spec_decode_num_drafts')
        result['spec_decode_draft_tokens_total'] = _sum_counter(
            'vllm:spec_decode_num_draft_tokens'
        )
        result['spec_decode_accepted_tokens_total'] = _sum_counter(
            'vllm:spec_decode_num_accepted_tokens'
        )
        
        # Calculate derived dynamic values
        if 'cache_usage_perc' in result and 'total_gpu_blocks' in self.static_config:
            total_blocks = self.static_config['total_gpu_blocks']
            usage_percent = result['cache_usage_perc']
            used_blocks = int(total_blocks * usage_percent / 100)
            free_blocks = total_blocks - used_blocks
            
            result['used_gpu_blocks'] = used_blocks
            result['free_gpu_blocks'] = free_blocks
            
            if 'tokens_per_block' in self.static_config:
                result['used_kv_cache_tokens'] = used_blocks * self.static_config['tokens_per_block']
                result['free_kv_cache_tokens'] = free_blocks * self.static_config['tokens_per_block']
        
        return result
        
    def _monitor_kv_cache(self):
        """Main monitoring loop running in background thread."""
        # Time series data
        cache_usage_readings = []
        used_blocks_readings = []
        free_blocks_readings = []
        used_tokens_readings = []
        requests_running_readings = []
        requests_waiting_readings = []
        active_concurrency_readings = []
        page_eviction_ops_total_readings = []
        page_eviction_ops_prefill_total_readings = []
        page_eviction_ops_decode_total_readings = []
        page_eviction_blocks_total_readings = []
        page_eviction_blocks_prefill_total_readings = []
        page_eviction_blocks_decode_total_readings = []
        page_eviction_prefill_reqs_scheduled_total_readings = []
        page_eviction_prefill_reqs_query_len_gt_budget_total_readings = []
        page_eviction_replace_block_req_ids_total_readings = []
        page_eviction_score_collect_calls_single_total_readings = []
        page_eviction_score_collect_calls_ubatch_list_total_readings = []
        page_eviction_score_collect_return_none_ubatch_list_total_readings = []
        page_eviction_prefill_block_scores_returned_total_readings = []
        page_eviction_decode_token_scores_returned_total_readings = []
        page_eviction_prefill_compress_invocations_total_readings = []
        page_eviction_ops_delta_readings = []
        page_eviction_ops_prefill_delta_readings = []
        page_eviction_ops_decode_delta_readings = []
        page_eviction_blocks_delta_readings = []
        page_eviction_blocks_prefill_delta_readings = []
        page_eviction_blocks_decode_delta_readings = []
        page_eviction_prefill_reqs_scheduled_delta_readings = []
        page_eviction_prefill_reqs_query_len_gt_budget_delta_readings = []
        page_eviction_replace_block_req_ids_delta_readings = []
        page_eviction_score_collect_calls_single_delta_readings = []
        page_eviction_score_collect_calls_ubatch_list_delta_readings = []
        page_eviction_score_collect_return_none_ubatch_list_delta_readings = []
        page_eviction_prefill_block_scores_returned_delta_readings = []
        page_eviction_decode_token_scores_returned_delta_readings = []
        page_eviction_prefill_compress_invocations_delta_readings = []
        request_success_total_readings = []
        request_success_stop_total_readings = []
        request_success_length_total_readings = []
        request_success_abort_total_readings = []
        request_success_error_total_readings = []
        request_success_delta_readings = []
        prefill_ops_per_completed_req_readings = []
        decode_ops_per_completed_req_readings = []
        prefill_compress_time_ms_per_event_readings = []
        prefill_keep_len_readings = []
        prefill_kept_ratio_readings = []
        decode_evicted_blocks_per_op_readings = []
        decode_eviction_time_ms_per_op_readings = []
        decode_pages_scored_per_op_readings = []
        spec_draft_tokens_total_readings = []
        spec_accepted_tokens_total_readings = []
        spec_draft_tokens_delta_readings = []
        spec_accepted_tokens_delta_readings = []
        spec_acceptance_rate_readings = []
        prompt_tokens_total_readings = []
        generation_tokens_total_readings = []
        prompt_tokens_delta_readings = []
        generation_tokens_delta_readings = []
        page_eviction_prefill_compress_time_seconds_total_readings = []
        page_eviction_prefill_keep_len_tokens_total_readings = []
        page_eviction_prefill_prompt_len_tokens_total_readings = []
        page_eviction_decode_eviction_time_seconds_total_readings = []
        page_eviction_decode_pages_scored_total_readings = []
        timestamps = []

        prev_counters: Optional[dict[str, float]] = None
        
        print(f"[KV Monitor] Starting monitoring with interval={self.interval}s")
        
        while not self.done:
            start_time = time.time()
            
            # Fetch and parse metrics
            metrics_text = self._fetch_metrics_sync()
            if metrics_text:
                metrics = self._parse_metrics(metrics_text)
                
                # Collect time series data only if we got valid metrics
                if metrics:
                    timestamps.append(time.time())
                    cache_usage_readings.append(metrics.get('cache_usage_perc', 0))
                    used_blocks_readings.append(metrics.get('used_gpu_blocks', 0))
                    free_blocks_readings.append(metrics.get('free_gpu_blocks', 0))
                    used_tokens_readings.append(metrics.get('used_kv_cache_tokens', 0))
                    requests_running = metrics.get('num_requests_running', 0)
                    requests_waiting = metrics.get('num_requests_waiting', 0)
                    requests_running_readings.append(requests_running)
                    requests_waiting_readings.append(requests_waiting)
                    active_concurrency_readings.append(requests_running + requests_waiting)
                    page_eviction_ops_total = metrics.get('page_eviction_ops_total', 0.0)
                    page_eviction_ops_prefill_total = metrics.get(
                        'page_eviction_ops_prefill_total', 0.0
                    )
                    page_eviction_ops_decode_total = metrics.get(
                        'page_eviction_ops_decode_total', 0.0
                    )
                    page_eviction_blocks_total = metrics.get('page_eviction_blocks_total', 0.0)
                    page_eviction_blocks_prefill_total = metrics.get(
                        'page_eviction_blocks_prefill_total', 0.0
                    )
                    page_eviction_blocks_decode_total = metrics.get(
                        'page_eviction_blocks_decode_total', 0.0
                    )
                    page_eviction_prefill_reqs_scheduled_total = metrics.get(
                        'page_eviction_prefill_reqs_scheduled_total', 0.0
                    )
                    page_eviction_prefill_reqs_query_len_gt_budget_total = metrics.get(
                        'page_eviction_prefill_reqs_query_len_gt_budget_total', 0.0
                    )
                    page_eviction_replace_block_req_ids_total = metrics.get(
                        'page_eviction_replace_block_req_ids_total', 0.0
                    )
                    page_eviction_score_collect_calls_single_total = metrics.get(
                        'page_eviction_score_collect_calls_single_total', 0.0
                    )
                    page_eviction_score_collect_calls_ubatch_list_total = metrics.get(
                        'page_eviction_score_collect_calls_ubatch_list_total', 0.0
                    )
                    page_eviction_score_collect_return_none_ubatch_list_total = metrics.get(
                        'page_eviction_score_collect_return_none_ubatch_list_total', 0.0
                    )
                    page_eviction_prefill_block_scores_returned_total = metrics.get(
                        'page_eviction_prefill_block_scores_returned_total', 0.0
                    )
                    page_eviction_decode_token_scores_returned_total = metrics.get(
                        'page_eviction_decode_token_scores_returned_total', 0.0
                    )
                    page_eviction_prefill_compress_invocations_total = metrics.get(
                        'page_eviction_prefill_compress_invocations_total', 0.0
                    )
                    request_success_total = metrics.get('request_success_total', 0.0)
                    request_success_stop_total = metrics.get(
                        'request_success_stop_total', 0.0
                    )
                    request_success_length_total = metrics.get(
                        'request_success_length_total', 0.0
                    )
                    request_success_abort_total = metrics.get(
                        'request_success_abort_total', 0.0
                    )
                    request_success_error_total = metrics.get(
                        'request_success_error_total', 0.0
                    )
                    spec_draft_tokens_total = metrics.get('spec_decode_draft_tokens_total', 0.0)
                    spec_accepted_tokens_total = metrics.get('spec_decode_accepted_tokens_total', 0.0)
                    prompt_tokens_total = metrics.get('prompt_tokens_total', 0.0)
                    generation_tokens_total = metrics.get('generation_tokens_total', 0.0)
                    prefill_compress_time_total = metrics.get(
                        'page_eviction_prefill_compress_time_seconds_total', 0.0
                    )
                    prefill_keep_len_total = metrics.get(
                        'page_eviction_prefill_keep_len_tokens_total', 0.0
                    )
                    prefill_prompt_len_total = metrics.get(
                        'page_eviction_prefill_prompt_len_tokens_total', 0.0
                    )
                    decode_eviction_time_total = metrics.get(
                        'page_eviction_decode_eviction_time_seconds_total', 0.0
                    )
                    decode_pages_scored_total = metrics.get(
                        'page_eviction_decode_pages_scored_total', 0.0
                    )

                    page_eviction_ops_total_readings.append(page_eviction_ops_total)
                    page_eviction_ops_prefill_total_readings.append(
                        page_eviction_ops_prefill_total
                    )
                    page_eviction_ops_decode_total_readings.append(
                        page_eviction_ops_decode_total
                    )
                    page_eviction_blocks_total_readings.append(page_eviction_blocks_total)
                    page_eviction_blocks_prefill_total_readings.append(
                        page_eviction_blocks_prefill_total
                    )
                    page_eviction_blocks_decode_total_readings.append(
                        page_eviction_blocks_decode_total
                    )
                    page_eviction_prefill_reqs_scheduled_total_readings.append(
                        page_eviction_prefill_reqs_scheduled_total
                    )
                    page_eviction_prefill_reqs_query_len_gt_budget_total_readings.append(
                        page_eviction_prefill_reqs_query_len_gt_budget_total
                    )
                    page_eviction_replace_block_req_ids_total_readings.append(
                        page_eviction_replace_block_req_ids_total
                    )
                    page_eviction_score_collect_calls_single_total_readings.append(
                        page_eviction_score_collect_calls_single_total
                    )
                    page_eviction_score_collect_calls_ubatch_list_total_readings.append(
                        page_eviction_score_collect_calls_ubatch_list_total
                    )
                    page_eviction_score_collect_return_none_ubatch_list_total_readings.append(
                        page_eviction_score_collect_return_none_ubatch_list_total
                    )
                    page_eviction_prefill_block_scores_returned_total_readings.append(
                        page_eviction_prefill_block_scores_returned_total
                    )
                    page_eviction_decode_token_scores_returned_total_readings.append(
                        page_eviction_decode_token_scores_returned_total
                    )
                    page_eviction_prefill_compress_invocations_total_readings.append(
                        page_eviction_prefill_compress_invocations_total
                    )
                    request_success_total_readings.append(request_success_total)
                    request_success_stop_total_readings.append(request_success_stop_total)
                    request_success_length_total_readings.append(request_success_length_total)
                    request_success_abort_total_readings.append(request_success_abort_total)
                    request_success_error_total_readings.append(request_success_error_total)
                    spec_draft_tokens_total_readings.append(spec_draft_tokens_total)
                    spec_accepted_tokens_total_readings.append(spec_accepted_tokens_total)
                    prompt_tokens_total_readings.append(prompt_tokens_total)
                    generation_tokens_total_readings.append(generation_tokens_total)
                    page_eviction_prefill_compress_time_seconds_total_readings.append(
                        prefill_compress_time_total
                    )
                    page_eviction_prefill_keep_len_tokens_total_readings.append(
                        prefill_keep_len_total
                    )
                    page_eviction_prefill_prompt_len_tokens_total_readings.append(
                        prefill_prompt_len_total
                    )
                    page_eviction_decode_eviction_time_seconds_total_readings.append(
                        decode_eviction_time_total
                    )
                    page_eviction_decode_pages_scored_total_readings.append(
                        decode_pages_scored_total
                    )

                    if prev_counters is None:
                        page_eviction_ops_delta_readings.append(0.0)
                        page_eviction_ops_prefill_delta_readings.append(0.0)
                        page_eviction_ops_decode_delta_readings.append(0.0)
                        page_eviction_blocks_delta_readings.append(0.0)
                        page_eviction_blocks_prefill_delta_readings.append(0.0)
                        page_eviction_blocks_decode_delta_readings.append(0.0)
                        page_eviction_prefill_reqs_scheduled_delta_readings.append(0.0)
                        page_eviction_prefill_reqs_query_len_gt_budget_delta_readings.append(0.0)
                        page_eviction_replace_block_req_ids_delta_readings.append(0.0)
                        page_eviction_score_collect_calls_single_delta_readings.append(0.0)
                        page_eviction_score_collect_calls_ubatch_list_delta_readings.append(0.0)
                        page_eviction_score_collect_return_none_ubatch_list_delta_readings.append(0.0)
                        page_eviction_prefill_block_scores_returned_delta_readings.append(0.0)
                        page_eviction_decode_token_scores_returned_delta_readings.append(0.0)
                        page_eviction_prefill_compress_invocations_delta_readings.append(0.0)
                        request_success_delta_readings.append(0.0)
                        spec_draft_tokens_delta_readings.append(0.0)
                        spec_accepted_tokens_delta_readings.append(0.0)
                        prompt_tokens_delta_readings.append(0.0)
                        generation_tokens_delta_readings.append(0.0)
                    else:
                        page_eviction_ops_delta = max(
                            0.0,
                            page_eviction_ops_total
                            - prev_counters['page_eviction_ops_total'],
                        )
                        page_eviction_ops_prefill_delta = max(
                            0.0,
                            page_eviction_ops_prefill_total
                            - prev_counters['page_eviction_ops_prefill_total'],
                        )
                        page_eviction_ops_decode_delta = max(
                            0.0,
                            page_eviction_ops_decode_total
                            - prev_counters['page_eviction_ops_decode_total'],
                        )
                        page_eviction_blocks_prefill_delta = max(
                            0.0,
                            page_eviction_blocks_prefill_total
                            - prev_counters['page_eviction_blocks_prefill_total'],
                        )
                        page_eviction_blocks_decode_delta = max(
                            0.0,
                            page_eviction_blocks_decode_total
                            - prev_counters['page_eviction_blocks_decode_total'],
                        )
                        page_eviction_prefill_reqs_scheduled_delta = max(
                            0.0,
                            page_eviction_prefill_reqs_scheduled_total
                            - prev_counters['page_eviction_prefill_reqs_scheduled_total'],
                        )
                        page_eviction_prefill_reqs_query_len_gt_budget_delta = max(
                            0.0,
                            page_eviction_prefill_reqs_query_len_gt_budget_total
                            - prev_counters[
                                'page_eviction_prefill_reqs_query_len_gt_budget_total'
                            ],
                        )
                        page_eviction_replace_block_req_ids_delta = max(
                            0.0,
                            page_eviction_replace_block_req_ids_total
                            - prev_counters['page_eviction_replace_block_req_ids_total'],
                        )
                        page_eviction_score_collect_calls_single_delta = max(
                            0.0,
                            page_eviction_score_collect_calls_single_total
                            - prev_counters['page_eviction_score_collect_calls_single_total'],
                        )
                        page_eviction_score_collect_calls_ubatch_list_delta = max(
                            0.0,
                            page_eviction_score_collect_calls_ubatch_list_total
                            - prev_counters[
                                'page_eviction_score_collect_calls_ubatch_list_total'
                            ],
                        )
                        page_eviction_score_collect_return_none_ubatch_list_delta = max(
                            0.0,
                            page_eviction_score_collect_return_none_ubatch_list_total
                            - prev_counters[
                                'page_eviction_score_collect_return_none_ubatch_list_total'
                            ],
                        )
                        page_eviction_prefill_block_scores_returned_delta = max(
                            0.0,
                            page_eviction_prefill_block_scores_returned_total
                            - prev_counters[
                                'page_eviction_prefill_block_scores_returned_total'
                            ],
                        )
                        page_eviction_decode_token_scores_returned_delta = max(
                            0.0,
                            page_eviction_decode_token_scores_returned_total
                            - prev_counters[
                                'page_eviction_decode_token_scores_returned_total'
                            ],
                        )
                        page_eviction_prefill_compress_invocations_delta = max(
                            0.0,
                            page_eviction_prefill_compress_invocations_total
                            - prev_counters[
                                'page_eviction_prefill_compress_invocations_total'
                            ],
                        )
                        request_success_delta = max(
                            0.0,
                            request_success_total
                            - prev_counters['request_success_total'],
                        )
                        prompt_tokens_delta = max(
                            0.0,
                            prompt_tokens_total - prev_counters['prompt_tokens_total'],
                        )
                        generation_tokens_delta = max(
                            0.0,
                            generation_tokens_total
                            - prev_counters['generation_tokens_total'],
                        )
                        prefill_compress_time_delta = max(
                            0.0,
                            prefill_compress_time_total
                            - prev_counters[
                                'page_eviction_prefill_compress_time_seconds_total'
                            ],
                        )
                        prefill_keep_len_delta = max(
                            0.0,
                            prefill_keep_len_total
                            - prev_counters[
                                'page_eviction_prefill_keep_len_tokens_total'
                            ],
                        )
                        prefill_prompt_len_delta = max(
                            0.0,
                            prefill_prompt_len_total
                            - prev_counters[
                                'page_eviction_prefill_prompt_len_tokens_total'
                            ],
                        )
                        decode_eviction_time_delta = max(
                            0.0,
                            decode_eviction_time_total
                            - prev_counters[
                                'page_eviction_decode_eviction_time_seconds_total'
                            ],
                        )
                        decode_pages_scored_delta = max(
                            0.0,
                            decode_pages_scored_total
                            - prev_counters[
                                'page_eviction_decode_pages_scored_total'
                            ],
                        )
                        page_eviction_ops_delta_readings.append(page_eviction_ops_delta)
                        page_eviction_ops_prefill_delta_readings.append(
                            page_eviction_ops_prefill_delta
                        )
                        page_eviction_ops_decode_delta_readings.append(
                            page_eviction_ops_decode_delta
                        )
                        page_eviction_blocks_prefill_delta_readings.append(
                            page_eviction_blocks_prefill_delta
                        )
                        page_eviction_blocks_decode_delta_readings.append(
                            page_eviction_blocks_decode_delta
                        )
                        page_eviction_prefill_reqs_scheduled_delta_readings.append(
                            page_eviction_prefill_reqs_scheduled_delta
                        )
                        page_eviction_prefill_reqs_query_len_gt_budget_delta_readings.append(
                            page_eviction_prefill_reqs_query_len_gt_budget_delta
                        )
                        page_eviction_replace_block_req_ids_delta_readings.append(
                            page_eviction_replace_block_req_ids_delta
                        )
                        page_eviction_score_collect_calls_single_delta_readings.append(
                            page_eviction_score_collect_calls_single_delta
                        )
                        page_eviction_score_collect_calls_ubatch_list_delta_readings.append(
                            page_eviction_score_collect_calls_ubatch_list_delta
                        )
                        page_eviction_score_collect_return_none_ubatch_list_delta_readings.append(
                            page_eviction_score_collect_return_none_ubatch_list_delta
                        )
                        page_eviction_prefill_block_scores_returned_delta_readings.append(
                            page_eviction_prefill_block_scores_returned_delta
                        )
                        page_eviction_decode_token_scores_returned_delta_readings.append(
                            page_eviction_decode_token_scores_returned_delta
                        )
                        page_eviction_prefill_compress_invocations_delta_readings.append(
                            page_eviction_prefill_compress_invocations_delta
                        )
                        page_eviction_blocks_delta_readings.append(
                            max(
                                0.0,
                                page_eviction_blocks_total
                                - prev_counters['page_eviction_blocks_total'],
                            )
                        )
                        request_success_delta_readings.append(request_success_delta)
                        prompt_tokens_delta_readings.append(prompt_tokens_delta)
                        generation_tokens_delta_readings.append(generation_tokens_delta)
                        spec_draft_tokens_delta_readings.append(
                            max(
                                0.0,
                                spec_draft_tokens_total
                                - prev_counters['spec_decode_draft_tokens_total'],
                            )
                        )
                        if request_success_delta > 0:
                            prefill_ops_per_completed_req_readings.append(
                                page_eviction_ops_prefill_delta
                                / request_success_delta
                            )
                            decode_ops_per_completed_req_readings.append(
                                page_eviction_ops_decode_delta
                                / request_success_delta
                            )
                        if page_eviction_ops_prefill_delta > 0:
                            prefill_compress_time_ms_per_event_readings.append(
                                (prefill_compress_time_delta / page_eviction_ops_prefill_delta)
                                * 1000.0
                            )
                            prefill_keep_len_readings.append(
                                prefill_keep_len_delta / page_eviction_ops_prefill_delta
                            )
                            if prefill_prompt_len_delta > 0:
                                prefill_kept_ratio_readings.append(
                                    prefill_keep_len_delta / prefill_prompt_len_delta
                                )
                        if page_eviction_ops_decode_delta > 0:
                            decode_evicted_blocks_per_op_readings.append(
                                page_eviction_blocks_decode_delta
                                / page_eviction_ops_decode_delta
                            )
                            decode_eviction_time_ms_per_op_readings.append(
                                (decode_eviction_time_delta / page_eviction_ops_decode_delta)
                                * 1000.0
                            )
                            decode_pages_scored_per_op_readings.append(
                                decode_pages_scored_delta / page_eviction_ops_decode_delta
                            )
                        spec_accepted_tokens_delta_readings.append(
                            max(
                                0.0,
                                spec_accepted_tokens_total
                                - prev_counters['spec_decode_accepted_tokens_total'],
                            )
                        )

                    if spec_draft_tokens_total > 0:
                        spec_acceptance_rate_readings.append(
                            spec_accepted_tokens_total / spec_draft_tokens_total
                        )
                    else:
                        spec_acceptance_rate_readings.append(0.0)

                    prev_counters = {
                        'page_eviction_ops_total': page_eviction_ops_total,
                        'page_eviction_ops_prefill_total': page_eviction_ops_prefill_total,
                        'page_eviction_ops_decode_total': page_eviction_ops_decode_total,
                        'page_eviction_blocks_total': page_eviction_blocks_total,
                        'page_eviction_blocks_prefill_total': page_eviction_blocks_prefill_total,
                        'page_eviction_blocks_decode_total': page_eviction_blocks_decode_total,
                        'page_eviction_prefill_reqs_scheduled_total': (
                            page_eviction_prefill_reqs_scheduled_total
                        ),
                        'page_eviction_prefill_reqs_query_len_gt_budget_total': (
                            page_eviction_prefill_reqs_query_len_gt_budget_total
                        ),
                        'page_eviction_replace_block_req_ids_total': (
                            page_eviction_replace_block_req_ids_total
                        ),
                        'page_eviction_score_collect_calls_single_total': (
                            page_eviction_score_collect_calls_single_total
                        ),
                        'page_eviction_score_collect_calls_ubatch_list_total': (
                            page_eviction_score_collect_calls_ubatch_list_total
                        ),
                        'page_eviction_score_collect_return_none_ubatch_list_total': (
                            page_eviction_score_collect_return_none_ubatch_list_total
                        ),
                        'page_eviction_prefill_block_scores_returned_total': (
                            page_eviction_prefill_block_scores_returned_total
                        ),
                        'page_eviction_decode_token_scores_returned_total': (
                            page_eviction_decode_token_scores_returned_total
                        ),
                        'page_eviction_prefill_compress_invocations_total': (
                            page_eviction_prefill_compress_invocations_total
                        ),
                        'request_success_total': request_success_total,
                        'spec_decode_draft_tokens_total': spec_draft_tokens_total,
                        'spec_decode_accepted_tokens_total': spec_accepted_tokens_total,
                        'prompt_tokens_total': prompt_tokens_total,
                        'generation_tokens_total': generation_tokens_total,
                        'page_eviction_prefill_compress_time_seconds_total': prefill_compress_time_total,
                        'page_eviction_prefill_keep_len_tokens_total': prefill_keep_len_total,
                        'page_eviction_prefill_prompt_len_tokens_total': prefill_prompt_len_total,
                        'page_eviction_decode_eviction_time_seconds_total': decode_eviction_time_total,
                        'page_eviction_decode_pages_scored_total': decode_pages_scored_total,
                    }
            
            # Sleep for remaining interval time
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)
        
        print(f"[KV Monitor] Stopped. Collected {len(cache_usage_readings)} samples")
        print(f"[KV Monitor] Successful fetches: {self.successful_fetches}, Errors: {self.fetch_errors}")
        
        # Truncate data if needed
        if self.truncate > 0:
            samples_to_truncate = int(self.truncate / self.interval)
            if samples_to_truncate * 2 < len(cache_usage_readings):
                cache_usage_readings = cache_usage_readings[samples_to_truncate:-samples_to_truncate]
                used_blocks_readings = used_blocks_readings[samples_to_truncate:-samples_to_truncate]
                free_blocks_readings = free_blocks_readings[samples_to_truncate:-samples_to_truncate]
                used_tokens_readings = used_tokens_readings[samples_to_truncate:-samples_to_truncate]
                requests_running_readings = requests_running_readings[samples_to_truncate:-samples_to_truncate]
                requests_waiting_readings = requests_waiting_readings[samples_to_truncate:-samples_to_truncate]
                page_eviction_ops_total_readings = page_eviction_ops_total_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                page_eviction_ops_prefill_total_readings = (
                    page_eviction_ops_prefill_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_ops_decode_total_readings = (
                    page_eviction_ops_decode_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_blocks_total_readings = page_eviction_blocks_total_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                page_eviction_ops_delta_readings = page_eviction_ops_delta_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                page_eviction_ops_prefill_delta_readings = (
                    page_eviction_ops_prefill_delta_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_ops_decode_delta_readings = (
                    page_eviction_ops_decode_delta_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_blocks_delta_readings = (
                    page_eviction_blocks_delta_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                request_success_total_readings = request_success_total_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                request_success_delta_readings = request_success_delta_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                spec_draft_tokens_total_readings = spec_draft_tokens_total_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                spec_accepted_tokens_total_readings = (
                    spec_accepted_tokens_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                spec_draft_tokens_delta_readings = spec_draft_tokens_delta_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                spec_accepted_tokens_delta_readings = (
                    spec_accepted_tokens_delta_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                spec_acceptance_rate_readings = spec_acceptance_rate_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                active_concurrency_readings = active_concurrency_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                page_eviction_blocks_prefill_total_readings = (
                    page_eviction_blocks_prefill_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_blocks_decode_total_readings = (
                    page_eviction_blocks_decode_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_blocks_prefill_delta_readings = (
                    page_eviction_blocks_prefill_delta_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_blocks_decode_delta_readings = (
                    page_eviction_blocks_decode_delta_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                request_success_stop_total_readings = request_success_stop_total_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                request_success_length_total_readings = (
                    request_success_length_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                request_success_abort_total_readings = request_success_abort_total_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                request_success_error_total_readings = request_success_error_total_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                prompt_tokens_total_readings = prompt_tokens_total_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                generation_tokens_total_readings = generation_tokens_total_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                prompt_tokens_delta_readings = prompt_tokens_delta_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                generation_tokens_delta_readings = generation_tokens_delta_readings[
                    samples_to_truncate:-samples_to_truncate
                ]
                page_eviction_prefill_compress_time_seconds_total_readings = (
                    page_eviction_prefill_compress_time_seconds_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_prefill_keep_len_tokens_total_readings = (
                    page_eviction_prefill_keep_len_tokens_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_prefill_prompt_len_tokens_total_readings = (
                    page_eviction_prefill_prompt_len_tokens_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_decode_eviction_time_seconds_total_readings = (
                    page_eviction_decode_eviction_time_seconds_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
                page_eviction_decode_pages_scored_total_readings = (
                    page_eviction_decode_pages_scored_total_readings[
                        samples_to_truncate:-samples_to_truncate
                    ]
                )
        
        def _series_delta(values: list[float]) -> float:
            return (
                values[-1] - values[0]
                if len(values) >= 2
                else 0.0
            )

        def _pct(values: list[float], p: float) -> float:
            return float(np.percentile(values, p)) if values else 0.0

        # Calculate average statistics
        avg_cache_usage = np.mean(cache_usage_readings) if cache_usage_readings else 0
        avg_used_blocks = np.mean(used_blocks_readings) if used_blocks_readings else 0
        avg_free_blocks = np.mean(free_blocks_readings) if free_blocks_readings else 0
        avg_used_tokens = np.mean(used_tokens_readings) if used_tokens_readings else 0
        avg_requests_running = np.mean(requests_running_readings) if requests_running_readings else 0
        avg_requests_waiting = np.mean(requests_waiting_readings) if requests_waiting_readings else 0
        avg_page_eviction_ops_per_sample = (
            np.mean(page_eviction_ops_delta_readings)
            if page_eviction_ops_delta_readings
            else 0
        )
        avg_page_eviction_blocks_per_sample = (
            np.mean(page_eviction_blocks_delta_readings)
            if page_eviction_blocks_delta_readings
            else 0
        )
        avg_spec_acceptance_rate = (
            np.mean(spec_acceptance_rate_readings) if spec_acceptance_rate_readings else 0
        )
        total_page_eviction_ops = _series_delta(page_eviction_ops_total_readings)
        total_page_eviction_ops_prefill = _series_delta(
            page_eviction_ops_prefill_total_readings
        )
        total_page_eviction_ops_decode = _series_delta(
            page_eviction_ops_decode_total_readings
        )
        total_page_eviction_blocks = _series_delta(page_eviction_blocks_total_readings)
        total_page_eviction_blocks_prefill = _series_delta(
            page_eviction_blocks_prefill_total_readings
        )
        total_page_eviction_blocks_decode = _series_delta(
            page_eviction_blocks_decode_total_readings
        )
        total_page_eviction_prefill_reqs_scheduled = _series_delta(
            page_eviction_prefill_reqs_scheduled_total_readings
        )
        total_page_eviction_prefill_reqs_query_len_gt_budget = _series_delta(
            page_eviction_prefill_reqs_query_len_gt_budget_total_readings
        )
        total_page_eviction_replace_block_req_ids = _series_delta(
            page_eviction_replace_block_req_ids_total_readings
        )
        total_page_eviction_score_collect_calls_single = _series_delta(
            page_eviction_score_collect_calls_single_total_readings
        )
        total_page_eviction_score_collect_calls_ubatch_list = _series_delta(
            page_eviction_score_collect_calls_ubatch_list_total_readings
        )
        total_page_eviction_score_collect_return_none_ubatch_list = _series_delta(
            page_eviction_score_collect_return_none_ubatch_list_total_readings
        )
        total_page_eviction_prefill_block_scores_returned = _series_delta(
            page_eviction_prefill_block_scores_returned_total_readings
        )
        total_page_eviction_decode_token_scores_returned = _series_delta(
            page_eviction_decode_token_scores_returned_total_readings
        )
        total_page_eviction_prefill_compress_invocations = _series_delta(
            page_eviction_prefill_compress_invocations_total_readings
        )
        total_request_success = _series_delta(request_success_total_readings)
        total_request_success_stop = _series_delta(request_success_stop_total_readings)
        total_request_success_length = _series_delta(
            request_success_length_total_readings
        )
        total_request_success_abort = _series_delta(
            request_success_abort_total_readings
        )
        total_request_success_error = _series_delta(
            request_success_error_total_readings
        )
        prefill_calls_per_request_mean = (
            (total_page_eviction_ops_prefill / total_request_success)
            if total_request_success > 0
            else 0
        )
        prefill_calls_per_request_p99 = (
            float(np.percentile(prefill_ops_per_completed_req_readings, 99))
            if prefill_ops_per_completed_req_readings
            else 0
        )
        prefill_calls_per_request_max = (
            float(max(prefill_ops_per_completed_req_readings))
            if prefill_ops_per_completed_req_readings
            else 0.0
        )
        total_spec_draft_tokens = _series_delta(spec_draft_tokens_total_readings)
        total_spec_accepted_tokens = _series_delta(spec_accepted_tokens_total_readings)
        total_spec_acceptance_rate = (
            (total_spec_accepted_tokens / total_spec_draft_tokens)
            if total_spec_draft_tokens > 0
            else 0
        )
        total_prompt_tokens = _series_delta(prompt_tokens_total_readings)
        total_generation_tokens = _series_delta(generation_tokens_total_readings)
        total_prefill_compress_time_seconds = _series_delta(
            page_eviction_prefill_compress_time_seconds_total_readings
        )
        total_prefill_keep_len_tokens = _series_delta(
            page_eviction_prefill_keep_len_tokens_total_readings
        )
        total_prefill_prompt_len_tokens = _series_delta(
            page_eviction_prefill_prompt_len_tokens_total_readings
        )
        total_decode_eviction_time_seconds = _series_delta(
            page_eviction_decode_eviction_time_seconds_total_readings
        )
        total_decode_pages_scored = _series_delta(
            page_eviction_decode_pages_scored_total_readings
        )
        active_concurrency_p50 = _pct(active_concurrency_readings, 50)
        active_concurrency_p90 = _pct(active_concurrency_readings, 90)
        prefill_tokens_scheduled_p50 = _pct(prompt_tokens_delta_readings, 50)
        prefill_tokens_scheduled_p90 = _pct(prompt_tokens_delta_readings, 90)
        prefill_tokens_scheduled_p99 = _pct(prompt_tokens_delta_readings, 99)
        decode_tokens_scheduled_p50 = _pct(generation_tokens_delta_readings, 50)
        decode_tokens_scheduled_p90 = _pct(generation_tokens_delta_readings, 90)
        decode_tokens_scheduled_p99 = _pct(generation_tokens_delta_readings, 99)
        prefill_query_len_gt_budget_ratio = (
            (
                total_page_eviction_prefill_reqs_query_len_gt_budget
                / total_page_eviction_prefill_reqs_scheduled
            )
            if total_page_eviction_prefill_reqs_scheduled > 0
            else 0.0
        )
        score_collect_return_none_ubatch_ratio = (
            (
                total_page_eviction_score_collect_return_none_ubatch_list
                / total_page_eviction_score_collect_calls_ubatch_list
            )
            if total_page_eviction_score_collect_calls_ubatch_list > 0
            else 0.0
        )
        
        # Put average results in queue
        self.results_queue.put({
            'avg_cache_usage_perc': avg_cache_usage,
            'avg_used_blocks': avg_used_blocks,
            'avg_free_blocks': avg_free_blocks,
            'avg_used_tokens': avg_used_tokens,
            'avg_requests_running': avg_requests_running,
            'avg_requests_waiting': avg_requests_waiting,
            'avg_page_eviction_ops_per_sample': avg_page_eviction_ops_per_sample,
            'avg_page_eviction_blocks_per_sample': avg_page_eviction_blocks_per_sample,
            'total_page_eviction_ops': total_page_eviction_ops,
            'total_page_eviction_ops_prefill': total_page_eviction_ops_prefill,
            'total_page_eviction_ops_decode': total_page_eviction_ops_decode,
            'total_page_eviction_blocks': total_page_eviction_blocks,
            'total_page_eviction_blocks_prefill': total_page_eviction_blocks_prefill,
            'total_page_eviction_blocks_decode': total_page_eviction_blocks_decode,
            'total_page_eviction_prefill_reqs_scheduled': (
                total_page_eviction_prefill_reqs_scheduled
            ),
            'total_page_eviction_prefill_reqs_query_len_gt_budget': (
                total_page_eviction_prefill_reqs_query_len_gt_budget
            ),
            'prefill_query_len_gt_budget_ratio': prefill_query_len_gt_budget_ratio,
            'total_page_eviction_replace_block_req_ids': (
                total_page_eviction_replace_block_req_ids
            ),
            'replace_block_req_ids_count_per_sample_p50': _pct(
                page_eviction_replace_block_req_ids_delta_readings, 50
            ),
            'replace_block_req_ids_count_per_sample_p90': _pct(
                page_eviction_replace_block_req_ids_delta_readings, 90
            ),
            'replace_block_req_ids_count_per_sample_p99': _pct(
                page_eviction_replace_block_req_ids_delta_readings, 99
            ),
            'total_page_eviction_score_collect_calls_single': (
                total_page_eviction_score_collect_calls_single
            ),
            'total_page_eviction_score_collect_calls_ubatch_list': (
                total_page_eviction_score_collect_calls_ubatch_list
            ),
            'total_page_eviction_score_collect_return_none_ubatch_list': (
                total_page_eviction_score_collect_return_none_ubatch_list
            ),
            'score_collect_return_none_ubatch_ratio': (
                score_collect_return_none_ubatch_ratio
            ),
            'total_page_eviction_prefill_block_scores_returned': (
                total_page_eviction_prefill_block_scores_returned
            ),
            'total_page_eviction_decode_token_scores_returned': (
                total_page_eviction_decode_token_scores_returned
            ),
            'total_page_eviction_prefill_compress_invocations': (
                total_page_eviction_prefill_compress_invocations
            ),
            'prefill_compress_invocations_per_request_mean': (
                (total_page_eviction_prefill_compress_invocations / total_request_success)
                if total_request_success > 0
                else 0.0
            ),
            'total_request_success': total_request_success,
            'total_request_success_stop': total_request_success_stop,
            'total_request_success_length': total_request_success_length,
            'total_request_success_abort': total_request_success_abort,
            'total_request_success_error': total_request_success_error,
            'prefill_compress_calls_per_request_mean': prefill_calls_per_request_mean,
            'prefill_compress_calls_per_request_p99': prefill_calls_per_request_p99,
            'prefill_compress_calls_per_request_max': prefill_calls_per_request_max,
            'total_spec_draft_tokens': total_spec_draft_tokens,
            'total_spec_accepted_tokens': total_spec_accepted_tokens,
            'spec_decode_acceptance_rate': total_spec_acceptance_rate,
            'avg_spec_decode_acceptance_rate': avg_spec_acceptance_rate,
            'active_concurrency_p50': active_concurrency_p50,
            'active_concurrency_p90': active_concurrency_p90,
            'prefill_tokens_scheduled_p50': prefill_tokens_scheduled_p50,
            'prefill_tokens_scheduled_p90': prefill_tokens_scheduled_p90,
            'prefill_tokens_scheduled_p99': prefill_tokens_scheduled_p99,
            'decode_tokens_scheduled_p50': decode_tokens_scheduled_p50,
            'decode_tokens_scheduled_p90': decode_tokens_scheduled_p90,
            'decode_tokens_scheduled_p99': decode_tokens_scheduled_p99,
            'total_prompt_tokens_scheduled': total_prompt_tokens,
            'total_decode_tokens_scheduled': total_generation_tokens,
            'prefill_compress_time_ms_total': total_prefill_compress_time_seconds * 1000.0,
            'prefill_compress_time_ms_per_event_p50': _pct(
                prefill_compress_time_ms_per_event_readings, 50
            ),
            'prefill_compress_time_ms_per_event_p90': _pct(
                prefill_compress_time_ms_per_event_readings, 90
            ),
            'prefill_compress_time_ms_per_event_p99': _pct(
                prefill_compress_time_ms_per_event_readings, 99
            ),
            'prefill_keep_len_mean': (
                float(np.mean(prefill_keep_len_readings))
                if prefill_keep_len_readings
                else 0.0
            ),
            'prefill_keep_len_p90': _pct(prefill_keep_len_readings, 90),
            'prefill_kept_ratio_mean': (
                float(np.mean(prefill_kept_ratio_readings))
                if prefill_kept_ratio_readings
                else 0.0
            ),
            'prefill_kept_ratio_p90': _pct(prefill_kept_ratio_readings, 90),
            'decode_eviction_ops_per_request_p50': _pct(
                decode_ops_per_completed_req_readings, 50
            ),
            'decode_eviction_ops_per_request_p90': _pct(
                decode_ops_per_completed_req_readings, 90
            ),
            'decode_eviction_ops_per_request_p99': _pct(
                decode_ops_per_completed_req_readings, 99
            ),
            'decode_evicted_blocks_per_op_p50': _pct(
                decode_evicted_blocks_per_op_readings, 50
            ),
            'decode_evicted_blocks_per_op_p90': _pct(
                decode_evicted_blocks_per_op_readings, 90
            ),
            'decode_evicted_blocks_per_op_p99': _pct(
                decode_evicted_blocks_per_op_readings, 99
            ),
            'decode_eviction_time_ms_per_op_p50': _pct(
                decode_eviction_time_ms_per_op_readings, 50
            ),
            'decode_eviction_time_ms_per_op_p90': _pct(
                decode_eviction_time_ms_per_op_readings, 90
            ),
            'decode_eviction_time_ms_per_op_p99': _pct(
                decode_eviction_time_ms_per_op_readings, 99
            ),
            'decode_pages_scored_per_op_p50': _pct(
                decode_pages_scored_per_op_readings, 50
            ),
            'decode_pages_scored_per_op_p90': _pct(
                decode_pages_scored_per_op_readings, 90
            ),
            'decode_pages_scored_per_op_p99': _pct(
                decode_pages_scored_per_op_readings, 99
            ),
            'decode_eviction_time_ms_total': total_decode_eviction_time_seconds * 1000.0,
            'decode_pages_scored_total': total_decode_pages_scored,
        })
        
        # Calculate detailed statistics
        stats = {}
        if cache_usage_readings:
            stats['cache_usage'] = {
                'min': float(np.min(cache_usage_readings)),
                'p5': float(np.percentile(cache_usage_readings, 5)),
                'p25': float(np.percentile(cache_usage_readings, 25)),
                'median': float(np.median(cache_usage_readings)),
                'p75': float(np.percentile(cache_usage_readings, 75)),
                'p95': float(np.percentile(cache_usage_readings, 95)),
                'max': float(np.max(cache_usage_readings)),
                'std': float(np.std(cache_usage_readings)),
            }
        
        if used_blocks_readings:
            stats['used_blocks'] = {
                'min': int(np.min(used_blocks_readings)),
                'p25': int(np.percentile(used_blocks_readings, 25)),
                'median': int(np.median(used_blocks_readings)),
                'p75': int(np.percentile(used_blocks_readings, 75)),
                'max': int(np.max(used_blocks_readings)),
            }
        
        if used_tokens_readings:
            stats['used_tokens'] = {
                'min': int(np.min(used_tokens_readings)),
                'median': int(np.median(used_tokens_readings)),
                'max': int(np.max(used_tokens_readings)),
            }
        
        if requests_running_readings:
            stats['requests_running'] = {
                'min': float(np.min(requests_running_readings)),
                'median': float(np.median(requests_running_readings)),
                'max': float(np.max(requests_running_readings)),
            }
        if active_concurrency_readings:
            stats['active_concurrency'] = {
                'p50': float(np.percentile(active_concurrency_readings, 50)),
                'p90': float(np.percentile(active_concurrency_readings, 90)),
            }

        if page_eviction_ops_delta_readings:
            stats['page_eviction_ops_delta'] = {
                'mean': float(np.mean(page_eviction_ops_delta_readings)),
                'max': float(np.max(page_eviction_ops_delta_readings)),
                'total': float(total_page_eviction_ops),
            }
        if page_eviction_ops_prefill_delta_readings:
            stats['page_eviction_ops_prefill_delta'] = {
                'mean': float(np.mean(page_eviction_ops_prefill_delta_readings)),
                'max': float(np.max(page_eviction_ops_prefill_delta_readings)),
                'total': float(total_page_eviction_ops_prefill),
            }
        if page_eviction_ops_decode_delta_readings:
            stats['page_eviction_ops_decode_delta'] = {
                'mean': float(np.mean(page_eviction_ops_decode_delta_readings)),
                'max': float(np.max(page_eviction_ops_decode_delta_readings)),
                'total': float(total_page_eviction_ops_decode),
            }
        if page_eviction_prefill_reqs_scheduled_delta_readings:
            stats['page_eviction_prefill_reqs_scheduled_delta'] = {
                'mean': float(np.mean(page_eviction_prefill_reqs_scheduled_delta_readings)),
                'max': float(np.max(page_eviction_prefill_reqs_scheduled_delta_readings)),
                'total': float(total_page_eviction_prefill_reqs_scheduled),
            }
        if page_eviction_prefill_reqs_query_len_gt_budget_delta_readings:
            stats['page_eviction_prefill_reqs_query_len_gt_budget_delta'] = {
                'mean': float(np.mean(page_eviction_prefill_reqs_query_len_gt_budget_delta_readings)),
                'max': float(np.max(page_eviction_prefill_reqs_query_len_gt_budget_delta_readings)),
                'total': float(total_page_eviction_prefill_reqs_query_len_gt_budget),
                'ratio_total': float(prefill_query_len_gt_budget_ratio),
            }
        if page_eviction_replace_block_req_ids_delta_readings:
            stats['page_eviction_replace_block_req_ids_delta'] = {
                'p50': float(np.percentile(page_eviction_replace_block_req_ids_delta_readings, 50)),
                'p90': float(np.percentile(page_eviction_replace_block_req_ids_delta_readings, 90)),
                'p99': float(np.percentile(page_eviction_replace_block_req_ids_delta_readings, 99)),
                'total': float(total_page_eviction_replace_block_req_ids),
            }
        if page_eviction_score_collect_calls_single_delta_readings:
            stats['page_eviction_score_collect_calls_single_delta'] = {
                'mean': float(np.mean(page_eviction_score_collect_calls_single_delta_readings)),
                'total': float(total_page_eviction_score_collect_calls_single),
            }
        if page_eviction_score_collect_calls_ubatch_list_delta_readings:
            stats['page_eviction_score_collect_calls_ubatch_list_delta'] = {
                'mean': float(np.mean(page_eviction_score_collect_calls_ubatch_list_delta_readings)),
                'total': float(total_page_eviction_score_collect_calls_ubatch_list),
            }
        if page_eviction_score_collect_return_none_ubatch_list_delta_readings:
            stats['page_eviction_score_collect_return_none_ubatch_list_delta'] = {
                'mean': float(np.mean(page_eviction_score_collect_return_none_ubatch_list_delta_readings)),
                'total': float(total_page_eviction_score_collect_return_none_ubatch_list),
                'ratio_total': float(score_collect_return_none_ubatch_ratio),
            }
        if page_eviction_prefill_compress_invocations_delta_readings:
            stats['page_eviction_prefill_compress_invocations_delta'] = {
                'mean': float(np.mean(page_eviction_prefill_compress_invocations_delta_readings)),
                'max': float(np.max(page_eviction_prefill_compress_invocations_delta_readings)),
                'total': float(total_page_eviction_prefill_compress_invocations),
            }
        if prefill_ops_per_completed_req_readings:
            stats['prefill_compress_calls_per_request'] = {
                'mean': float(prefill_calls_per_request_mean),
                'p99': float(prefill_calls_per_request_p99),
                'max': float(prefill_calls_per_request_max),
                'samples': len(prefill_ops_per_completed_req_readings),
            }
        if prefill_compress_time_ms_per_event_readings:
            stats['prefill_compress_time_ms_per_event'] = {
                'p50': float(np.percentile(prefill_compress_time_ms_per_event_readings, 50)),
                'p90': float(np.percentile(prefill_compress_time_ms_per_event_readings, 90)),
                'p99': float(np.percentile(prefill_compress_time_ms_per_event_readings, 99)),
            }
        if prefill_keep_len_readings:
            stats['prefill_keep_len'] = {
                'mean': float(np.mean(prefill_keep_len_readings)),
                'p90': float(np.percentile(prefill_keep_len_readings, 90)),
            }
        if prefill_kept_ratio_readings:
            stats['prefill_kept_ratio'] = {
                'mean': float(np.mean(prefill_kept_ratio_readings)),
                'p90': float(np.percentile(prefill_kept_ratio_readings, 90)),
            }
        if decode_ops_per_completed_req_readings:
            stats['decode_eviction_ops_per_request'] = {
                'p50': float(np.percentile(decode_ops_per_completed_req_readings, 50)),
                'p90': float(np.percentile(decode_ops_per_completed_req_readings, 90)),
                'p99': float(np.percentile(decode_ops_per_completed_req_readings, 99)),
            }
        if decode_evicted_blocks_per_op_readings:
            stats['decode_evicted_blocks_per_op'] = {
                'p50': float(np.percentile(decode_evicted_blocks_per_op_readings, 50)),
                'p90': float(np.percentile(decode_evicted_blocks_per_op_readings, 90)),
                'p99': float(np.percentile(decode_evicted_blocks_per_op_readings, 99)),
            }
        if decode_eviction_time_ms_per_op_readings:
            stats['decode_eviction_time_ms_per_op'] = {
                'p50': float(np.percentile(decode_eviction_time_ms_per_op_readings, 50)),
                'p90': float(np.percentile(decode_eviction_time_ms_per_op_readings, 90)),
                'p99': float(np.percentile(decode_eviction_time_ms_per_op_readings, 99)),
            }
        if decode_pages_scored_per_op_readings:
            stats['decode_pages_scored_per_op'] = {
                'p50': float(np.percentile(decode_pages_scored_per_op_readings, 50)),
                'p90': float(np.percentile(decode_pages_scored_per_op_readings, 90)),
                'p99': float(np.percentile(decode_pages_scored_per_op_readings, 99)),
            }
        if prompt_tokens_delta_readings:
            stats['prefill_tokens_scheduled'] = {
                'p50': float(np.percentile(prompt_tokens_delta_readings, 50)),
                'p90': float(np.percentile(prompt_tokens_delta_readings, 90)),
                'p99': float(np.percentile(prompt_tokens_delta_readings, 99)),
            }
        if generation_tokens_delta_readings:
            stats['decode_tokens_scheduled'] = {
                'p50': float(np.percentile(generation_tokens_delta_readings, 50)),
                'p90': float(np.percentile(generation_tokens_delta_readings, 90)),
                'p99': float(np.percentile(generation_tokens_delta_readings, 99)),
            }

        if page_eviction_blocks_delta_readings:
            stats['page_eviction_blocks_delta'] = {
                'mean': float(np.mean(page_eviction_blocks_delta_readings)),
                'max': float(np.max(page_eviction_blocks_delta_readings)),
                'total': float(total_page_eviction_blocks),
            }

        if spec_draft_tokens_delta_readings or spec_accepted_tokens_delta_readings:
            stats['spec_decode'] = {
                'draft_tokens_total': float(total_spec_draft_tokens),
                'accepted_tokens_total': float(total_spec_accepted_tokens),
                'acceptance_rate': float(total_spec_acceptance_rate),
            }
        
        # Add static configuration
        stats['static_config'] = self.static_config
        
        self.stats_queue.put(stats)
        
        # Put time series data in queue for DQN state construction
        self.hist_queue.put({
            'cache_usage': cache_usage_readings,
            'used_blocks': used_blocks_readings,
            'free_blocks': free_blocks_readings,
            'used_tokens': used_tokens_readings,
            'requests_running': requests_running_readings,
            'requests_waiting': requests_waiting_readings,
            'active_concurrency': active_concurrency_readings,
            'page_eviction_ops_total': page_eviction_ops_total_readings,
            'page_eviction_ops_prefill_total': page_eviction_ops_prefill_total_readings,
            'page_eviction_ops_decode_total': page_eviction_ops_decode_total_readings,
            'page_eviction_blocks_total': page_eviction_blocks_total_readings,
            'page_eviction_blocks_prefill_total': page_eviction_blocks_prefill_total_readings,
            'page_eviction_blocks_decode_total': page_eviction_blocks_decode_total_readings,
            'page_eviction_prefill_reqs_scheduled_total': (
                page_eviction_prefill_reqs_scheduled_total_readings
            ),
            'page_eviction_prefill_reqs_query_len_gt_budget_total': (
                page_eviction_prefill_reqs_query_len_gt_budget_total_readings
            ),
            'page_eviction_replace_block_req_ids_total': (
                page_eviction_replace_block_req_ids_total_readings
            ),
            'page_eviction_score_collect_calls_single_total': (
                page_eviction_score_collect_calls_single_total_readings
            ),
            'page_eviction_score_collect_calls_ubatch_list_total': (
                page_eviction_score_collect_calls_ubatch_list_total_readings
            ),
            'page_eviction_score_collect_return_none_ubatch_list_total': (
                page_eviction_score_collect_return_none_ubatch_list_total_readings
            ),
            'page_eviction_prefill_block_scores_returned_total': (
                page_eviction_prefill_block_scores_returned_total_readings
            ),
            'page_eviction_decode_token_scores_returned_total': (
                page_eviction_decode_token_scores_returned_total_readings
            ),
            'page_eviction_prefill_compress_invocations_total': (
                page_eviction_prefill_compress_invocations_total_readings
            ),
            'page_eviction_ops_delta': page_eviction_ops_delta_readings,
            'page_eviction_ops_prefill_delta': page_eviction_ops_prefill_delta_readings,
            'page_eviction_ops_decode_delta': page_eviction_ops_decode_delta_readings,
            'page_eviction_blocks_delta': page_eviction_blocks_delta_readings,
            'page_eviction_blocks_prefill_delta': page_eviction_blocks_prefill_delta_readings,
            'page_eviction_blocks_decode_delta': page_eviction_blocks_decode_delta_readings,
            'page_eviction_prefill_reqs_scheduled_delta': (
                page_eviction_prefill_reqs_scheduled_delta_readings
            ),
            'page_eviction_prefill_reqs_query_len_gt_budget_delta': (
                page_eviction_prefill_reqs_query_len_gt_budget_delta_readings
            ),
            'page_eviction_replace_block_req_ids_delta': (
                page_eviction_replace_block_req_ids_delta_readings
            ),
            'page_eviction_score_collect_calls_single_delta': (
                page_eviction_score_collect_calls_single_delta_readings
            ),
            'page_eviction_score_collect_calls_ubatch_list_delta': (
                page_eviction_score_collect_calls_ubatch_list_delta_readings
            ),
            'page_eviction_score_collect_return_none_ubatch_list_delta': (
                page_eviction_score_collect_return_none_ubatch_list_delta_readings
            ),
            'page_eviction_prefill_block_scores_returned_delta': (
                page_eviction_prefill_block_scores_returned_delta_readings
            ),
            'page_eviction_decode_token_scores_returned_delta': (
                page_eviction_decode_token_scores_returned_delta_readings
            ),
            'page_eviction_prefill_compress_invocations_delta': (
                page_eviction_prefill_compress_invocations_delta_readings
            ),
            'request_success_total': request_success_total_readings,
            'request_success_stop_total': request_success_stop_total_readings,
            'request_success_length_total': request_success_length_total_readings,
            'request_success_abort_total': request_success_abort_total_readings,
            'request_success_error_total': request_success_error_total_readings,
            'request_success_delta': request_success_delta_readings,
            'prefill_compress_calls_per_request': prefill_ops_per_completed_req_readings,
            'decode_eviction_ops_per_request': decode_ops_per_completed_req_readings,
            'prefill_compress_time_ms_per_event': prefill_compress_time_ms_per_event_readings,
            'prefill_keep_len': prefill_keep_len_readings,
            'prefill_kept_ratio': prefill_kept_ratio_readings,
            'decode_evicted_blocks_per_op': decode_evicted_blocks_per_op_readings,
            'decode_eviction_time_ms_per_op': decode_eviction_time_ms_per_op_readings,
            'decode_pages_scored_per_op': decode_pages_scored_per_op_readings,
            'prompt_tokens_total': prompt_tokens_total_readings,
            'generation_tokens_total': generation_tokens_total_readings,
            'prompt_tokens_delta': prompt_tokens_delta_readings,
            'generation_tokens_delta': generation_tokens_delta_readings,
            'page_eviction_prefill_compress_time_seconds_total': (
                page_eviction_prefill_compress_time_seconds_total_readings
            ),
            'page_eviction_prefill_keep_len_tokens_total': (
                page_eviction_prefill_keep_len_tokens_total_readings
            ),
            'page_eviction_prefill_prompt_len_tokens_total': (
                page_eviction_prefill_prompt_len_tokens_total_readings
            ),
            'page_eviction_decode_eviction_time_seconds_total': (
                page_eviction_decode_eviction_time_seconds_total_readings
            ),
            'page_eviction_decode_pages_scored_total': (
                page_eviction_decode_pages_scored_total_readings
            ),
            'spec_decode_draft_tokens_total': spec_draft_tokens_total_readings,
            'spec_decode_accepted_tokens_total': spec_accepted_tokens_total_readings,
            'spec_decode_acceptance_rate': spec_acceptance_rate_readings,
        })
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current KV cache state for DQN.
        This can be called at any time to get the latest state.
        
        Returns:
            Dictionary with current KV cache metrics for state space
        """
        metrics_text = self._fetch_metrics_sync()
        if metrics_text:
            metrics = self._parse_metrics(metrics_text)
            
            # Normalize values for DQN state space
            state = {}
            
            # Cache usage percentage (0-100) -> normalize to 0-1
            if 'cache_usage_perc' in metrics:
                state['kv_cache_usage'] = metrics['cache_usage_perc'] / 100.0
            
            # Used blocks ratio
            if 'used_gpu_blocks' in metrics and 'total_gpu_blocks' in self.static_config:
                state['kv_blocks_used_ratio'] = (
                    metrics['used_gpu_blocks'] / self.static_config['total_gpu_blocks']
                )
            
            # Request queue state
            if 'num_requests_running' in metrics:
                state['num_requests_running'] = metrics['num_requests_running']
            if 'num_requests_waiting' in metrics:
                state['num_requests_waiting'] = metrics['num_requests_waiting']

            state['page_eviction_ops_total'] = metrics.get(
                'page_eviction_ops_total', 0.0
            )
            state['page_eviction_blocks_total'] = metrics.get(
                'page_eviction_blocks_total', 0.0
            )
            draft_total = metrics.get('spec_decode_draft_tokens_total', 0.0)
            accepted_total = metrics.get('spec_decode_accepted_tokens_total', 0.0)
            state['spec_decode_draft_tokens_total'] = draft_total
            state['spec_decode_accepted_tokens_total'] = accepted_total
            state['spec_decode_acceptance_rate'] = (
                accepted_total / draft_total if draft_total > 0 else 0.0
            )
            
            # Total requests in system
            running = metrics.get('num_requests_running', 0)
            waiting = metrics.get('num_requests_waiting', 0)
            state['total_active_requests'] = running + waiting
            
            return state
        
        return {}
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
