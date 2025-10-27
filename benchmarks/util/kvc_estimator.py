def estimate_kv_cache_size(layers, hidden_size, heads, input_tokens, output_tokens, precision="fp16", unit="GB"):
    """
    Estimate the KV cache size for a single request in a large language model.

    Parameters:
    - layers (int): Number of transformer layers in the model.
    - hidden_size (int): Dimension of the hidden states in the model.
    - heads (int): Number of attention heads in the model.
    - input_tokens (int): Number of input tokens in the request.
    - output_tokens (int): Number of output tokens to generate.
    - precision (str): Precision type. Either "fp32" (default) or "fp16".

    Returns:
    - float: Estimated KV cache size in MB/GB.
    """

    # Step 1: Compute the size of a single attention head (hidden size per head)
    head_dim = hidden_size // heads

    # Step 2: Calculate the KV cache size for one token per layer (2x because both key and value are stored)
    kv_cache_per_token_per_layer = 2 * head_dim * heads  # 2x for key and value

    # Step 3: Compute the total sequence length (input tokens + output tokens)
    total_tokens = input_tokens + output_tokens

    # Step 4: Compute the total KV cache size for the entire sequence across all layers
    kv_cache_total_floats = kv_cache_per_token_per_layer * layers * total_tokens

    # Step 5: Determine the size per float based on precision
    if precision == "fp16":
        bytes_per_float = 2  # FP16 precision: 2 bytes per float
    else:
        bytes_per_float = 4  # FP32 precision: 4 bytes per float

    # Step 6: Calculate the total size in bytes and convert to MB/GB
    kv_cache_total_bytes = kv_cache_total_floats * bytes_per_float
    if unit == "MB":
        kv_cache_size = kv_cache_total_bytes / (1024 ** 2)
    else:
        kv_cache_size = kv_cache_total_bytes / (1024 ** 3)  # Convert bytes to GB

    return kv_cache_size


def estimate_bandwidth_need(kv_cache_size, request_rate, unit="GB"):
    """
    Estimate the bandwidth needed to support a given request rate.

    Parameters:
    - kv_cache_size (float): Estimated KV cache size in MB/GB.
    - request_rate (float): Number of requests per second.
    - unit (str): Bandwidth unit. Either "GB" (default) or "MB".

    Returns:
    - float: Estimated bandwidth needed in Gbps.
    """

    # Step 1: Convert the KV cache size to bytes
    if unit == "GB":
        kv_cache_size_bytes = kv_cache_size * (1024 ** 3)
    else:
        kv_cache_size_bytes = kv_cache_size * (1024 ** 2)  
        
    # Step 2: Calculate the bandwidth needed in bytes per second
    bandwidth_bytes_per_sec = kv_cache_size_bytes * request_rate

    # Step 3: Convert the bandwidth to Gbps
    bandwidth_gbps = bandwidth_bytes_per_sec * 8 / (1024 ** 3)  # Convert bytes to Gbps

    return bandwidth_gbps