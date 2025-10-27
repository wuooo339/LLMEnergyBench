import threading
import time
import queue
import numpy as np
import psutil

class CPUMonitor:
    def __init__(self, interval=0.025, truncate=0):
        '''
        truncates the last `truncate` seconds of the monitoring data.
        '''
        self.done = False
        self.results_queue = queue.Queue()
        self.stats_queue = queue.Queue()
        self.hist_queue = queue.Queue()
        self.interval = interval
        self.thread = None
        self.truncate = truncate

    def start(self):
        self.done = False
        self.thread = threading.Thread(target=self._monitor_cpu)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.done = True
            self.thread.join()

    def _monitor_cpu(self):
        cpu_utilization_readings = []
        cpu_freq_readings = []
        disk_io_readings = []
        mem_utilization_readings = []

        while not self.done:
            # CPU utilization per core
            cpu_utilization = psutil.cpu_percent(interval=None, percpu=True)
            cpu_utilization_readings.append(cpu_utilization)
            
            # CPU frequency per core
            cpu_freq = psutil.cpu_freq(percpu=True)
            if cpu_freq:
                cpu_freq_readings.append([f.current for f in cpu_freq])

            # Disk I/O counts
            disk_io = psutil.disk_io_counters()
            disk_io_readings.append((disk_io.read_count, disk_io.write_count))

            # Memory usage
            mem = psutil.virtual_memory()
            mem_utilization_readings.append(mem.percent)
            
            time.sleep(self.interval)

        if self.truncate > 0:
            seconds_to_truncate = int(self.truncate / self.interval)
            if seconds_to_truncate * 2 > len(cpu_utilization_readings):
                print(f"[Warning] Truncate value too high. This will lead to empty readings.")
            cpu_utilization_readings = cpu_utilization_readings[seconds_to_truncate:-seconds_to_truncate]
            cpu_freq_readings = cpu_freq_readings[seconds_to_truncate:-seconds_to_truncate]
            disk_io_readings = disk_io_readings[seconds_to_truncate:-seconds_to_truncate]
            mem_utilization_readings = mem_utilization_readings[seconds_to_truncate:-seconds_to_truncate]
        # CPU stats
        avg_cpu_util = np.mean(cpu_utilization_readings) if cpu_utilization_readings else 0
        avg_cpu_freq = np.mean(cpu_freq_readings) if cpu_freq_readings else 0\
        
        # Calculate stats for each core's utilization and frequency
        core_cpu_stats = []
        for core_util in zip(*cpu_utilization_readings):
            core_cpu_stats.append({
                "avg_cpu_util": np.mean(core_util),
                "max_cpu_util": np.max(core_util),
                "min_cpu_util": np.min(core_util),
                "cpu_util_std": np.std(core_util)
            })

        cpu_freq_stats = []
        for cpu_freq in zip(*cpu_freq_readings):
            cpu_freq_stats.append({
                "avg_cpu_freq": np.mean(cpu_freq),
                "max_cpu_freq": np.max(cpu_freq),
                "min_cpu_freq": np.min(cpu_freq),
                "cpu_freq_std": np.std(cpu_freq)
            })

        # Disk I/O stats
        disk_io_stats = {
            "read_count": disk_io_readings[-1][0] - disk_io_readings[0][0],
            "write_count": disk_io_readings[-1][1] - disk_io_readings[0][1]
        }

        # Memory utilization stats
        avg_mem_util = np.mean(mem_utilization_readings) if mem_utilization_readings else 0

        # Preparing the final stats dictionary
        stats = {
            "avg_cpu_util": avg_cpu_util,
            "avg_cpu_freq": avg_cpu_freq,
            "core_cpu_stats": core_cpu_stats,
            "cpu_freq_stats": cpu_freq_stats,
            "disk_io_stats": disk_io_stats,
            "avg_mem_util": avg_mem_util
        }

        self.results_queue.put(avg_cpu_util)
        self.stats_queue.put(stats)

        # Store history for further processing
        self.hist_queue.put([cpu_utilization_readings, cpu_freq_readings, disk_io_readings, mem_utilization_readings])
        

if __name__ == "__main__":
    print(psutil.sensors_temperatures())
    logical_cores = psutil.cpu_count(logical=True)
    print(f"Logical cores (including hyperthreading): {logical_cores}")

    physical_cores = psutil.cpu_count(logical=False)
    print(f"Physical cores: {physical_cores}")
    
    # Starting the CPU monitor with the new features enabled
    cpu_monitor = CPUMonitor(interval=0.2, truncate=2)
    cpu_monitor.start()
    
    time.sleep(10)
    cpu_monitor.stop()

    avg_cpu_util = cpu_monitor.results_queue.get()
    print(f"Average CPU Utilization: {avg_cpu_util}%")

    stats = cpu_monitor.stats_queue.get()
    print(f"Stats: {stats}")
