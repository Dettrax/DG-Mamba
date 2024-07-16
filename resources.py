import subprocess
import psutil
import time
from gpustat.core import GPUStatCollection

def clear_screen():
    print("\033[H\033[J", end='')

try:
    while True:
        clear_screen()  # Clear the screen and move the cursor to the top
        
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)  # Measures over 1 second
        cpu_per_core_usage = psutil.cpu_percent(interval=1, percpu=True)

        # Sort CPU usage per core in descending order
        sorted_cpu_usage = sorted(enumerate(cpu_per_core_usage), key=lambda x: x[1], reverse=False)

        # Get RAM utilization
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024 ** 3)
        ram_total_gb = ram.total / (1024 ** 3)

        # Get GPU utilization via gpustat
        gpu_stats = GPUStatCollection.new_query()
        gpu_usage = [gpu.utilization for gpu in gpu_stats.gpus]
        gpu_mem_usage = [gpu.memory_used / gpu.memory_total * 100 for gpu in gpu_stats.gpus]

        # Get detailed GPU info via nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        nvidia_smi_output = result.stdout

        # Display CPU, RAM, and GPU stats
        print(f"Overall CPU Usage: {cpu_usage}%")
        print(f"RAM Usage: {ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB")
        # for i, (gpu_util, mem_util) in enumerate(zip(gpu_usage, gpu_mem_usage)):
        #     print(f"GPU {i} Usage: {gpu_util}%  |  Memory Usage: {mem_util:.2f}%")
        #
        # # Display nvidia-smi output
        # print("\nDetailed GPU Stats via nvidia-smi:")
        # print(nvidia_smi_output)

        time.sleep(1)  # Refresh rate of 1 second
except KeyboardInterrupt:
    print("Monitoring stopped.")
