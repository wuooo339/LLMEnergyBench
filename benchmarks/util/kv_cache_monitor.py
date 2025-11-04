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
        timestamps = []
        
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
                    requests_running_readings.append(metrics.get('num_requests_running', 0))
                    requests_waiting_readings.append(metrics.get('num_requests_waiting', 0))
            
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
        
        # Calculate average statistics
        avg_cache_usage = np.mean(cache_usage_readings) if cache_usage_readings else 0
        avg_used_blocks = np.mean(used_blocks_readings) if used_blocks_readings else 0
        avg_free_blocks = np.mean(free_blocks_readings) if free_blocks_readings else 0
        avg_used_tokens = np.mean(used_tokens_readings) if used_tokens_readings else 0
        avg_requests_running = np.mean(requests_running_readings) if requests_running_readings else 0
        avg_requests_waiting = np.mean(requests_waiting_readings) if requests_waiting_readings else 0
        
        # Put average results in queue
        self.results_queue.put({
            'avg_cache_usage_perc': avg_cache_usage,
            'avg_used_blocks': avg_used_blocks,
            'avg_free_blocks': avg_free_blocks,
            'avg_used_tokens': avg_used_tokens,
            'avg_requests_running': avg_requests_running,
            'avg_requests_waiting': avg_requests_waiting,
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
            
            # Total requests in system
            running = metrics.get('num_requests_running', 0)
            waiting = metrics.get('num_requests_waiting', 0)
            state['total_active_requests'] = running + waiting
            
            return state
        
        return {}
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()

