import threading
import time
import queue
import numpy as np
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlDeviceGetClockInfo, NVMLError

class GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.025, truncate=0, monitor_clcock=False):
        '''
        truncates the last `truncate` seconds of the monitoring data
        '''
        self.gpu_id = gpu_id
        self.done = False
        self.results_queue = queue.Queue()
        self.stats_queue = queue.Queue()
        self.hist_queue = queue.Queue()
        self.interval = interval
        self.thread = None
        self.truncate = truncate
        self.monitor_clock = monitor_clcock
        
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(gpu_id)
        except NVMLError as e:
            print(f"NVML Error: {e}")
            self.gpu_handle = None

    def start(self):
        if self.gpu_handle is None:
            print("GPU handle not initialized. Monitoring cannot start.")
            return
        self.done = False
        self.thread = threading.Thread(target=self._monitor_gpu)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.done = True
            self.thread.join()

    def _monitor_gpu(self):
        gpu_power_readings = []
        gpu_utilization_readings = []
        memory_utilization_readings = []
        if self.monitor_clock:
            gpu_clock_readings = []
        
        while not self.done:
            power = nvmlDeviceGetPowerUsage(self.gpu_handle)
            utilization = nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory = nvmlDeviceGetMemoryInfo(self.gpu_handle)

            gpu_power_readings.append(power)
            gpu_utilization_readings.append(utilization.gpu)
            # DO NOT USE utilization.memory, it is not what we want
            # utilization.memory is "Percent of time over the past second in which any framebuffer memory has been read or stored."
            memory_utilization_readings.append(memory.used / memory.total * 100)

            if self.monitor_clock:
                graphics_clock = nvmlDeviceGetClockInfo(self.gpu_handle, 0)
                sm_clock = nvmlDeviceGetClockInfo(self.gpu_handle, 1)
                memory_clock = nvmlDeviceGetClockInfo(self.gpu_handle, 2)
                gpu_clock_readings.append((graphics_clock, sm_clock, memory_clock))

            time.sleep(self.interval)

        if self.truncate > 0:
            seconds_to_truncate = int(self.truncate/self.interval)
            if seconds_to_truncate * 2 > len(gpu_power_readings):
                print(f"[Warning] Truncate value too high. This will lead to empty readings.")
            gpu_power_readings = gpu_power_readings[seconds_to_truncate:-seconds_to_truncate]
            gpu_utilization_readings = gpu_utilization_readings[seconds_to_truncate:-seconds_to_truncate]
            memory_utilization_readings = memory_utilization_readings[seconds_to_truncate:-seconds_to_truncate]

        avg_power = sum(gpu_power_readings) / len(gpu_power_readings) if gpu_power_readings else 0
        avg_gpu_util = sum(gpu_utilization_readings) / len(gpu_utilization_readings) if gpu_utilization_readings else 0
        avg_mem_util = sum(memory_utilization_readings) / len(memory_utilization_readings) if memory_utilization_readings else 0

        self.results_queue.put((avg_power, avg_gpu_util, avg_mem_util))

        min_power = min(gpu_power_readings) if gpu_power_readings else 0
        power_5p = np.percentile(gpu_power_readings, 5) if gpu_power_readings else 0
        power_25p = np.percentile(gpu_power_readings, 25) if gpu_power_readings else 0
        median_power = np.median(gpu_power_readings) if gpu_power_readings else 0
        power_75p = np.percentile(gpu_power_readings, 75) if gpu_power_readings else 0
        power_95p = np.percentile(gpu_power_readings, 95) if gpu_power_readings else 0
        max_power = max(gpu_power_readings) if gpu_power_readings else 0
        power_std = np.std(np.array(gpu_power_readings)/1000) if gpu_power_readings else 0

        self.stats_queue.put({
            "min_power": min_power,
            "power_5p": power_5p,
            "power_25p": power_25p,
            "median_power": median_power,
            "power_75p": power_75p,
            "power_95p": power_95p,
            "max_power": max_power,
            "power_std": power_std
        })

        if self.monitor_clock:
            self.hist_queue.put([gpu_power_readings, memory_utilization_readings, gpu_clock_readings])
        else:
            self.hist_queue.put([gpu_power_readings, memory_utilization_readings])

    def __del__(self):
        if self.gpu_handle:
            self.stop()
            nvmlShutdown()