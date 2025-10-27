import os

def parse_cpu_mask(mask):
    binary_mask = bin(int(mask, 16))[2:]
    
    binary_mask = binary_mask[::-1]

    cpu_indices = [i for i, bit in enumerate(binary_mask) if bit == '1']
    
    return cpu_indices

def get_slurm_cpu_bind():
    cpu_bind = os.environ.get('SLURM_CPU_BIND', None)
    
    if cpu_bind is not None and "mask_cpu" in cpu_bind:
        mask = cpu_bind.split(":")[1]
        return parse_cpu_mask(mask)
    else:
        return []

        
if __name__ == "__main__":
    cpu_indices = get_slurm_cpu_bind()
    print(len(cpu_indices))
    print(f"Assigned CPU Indices: {cpu_indices}")
    cpus_on_node = os.environ.get('SLURM_CPUS_ON_NODE', None)
    if cpus_on_node:
        print(f"CPUs on node: {cpus_on_node}")
    else:
        print("No CPU allocation information available.")
