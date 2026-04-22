"""
Utility functions for seed setting, device selection, and JSON serialization.

This module provides helper functions to ensure reproducibility across runs,
automatically detect and configure the computation device (CPU/GPU), and
save/load dictionaries as JSON files.
"""

import random
import json
import numpy as np
import torch


def set_seed_and_device(seed: int = 42) -> torch.device:
    """
    Set the global random seed for reproducibility and return the appropriate device.

    The function sets the seed for Python's random module, NumPy, and PyTorch.
    If a CUDA-capable GPU is available, it also configures cuDNN for deterministic
    behavior and returns the CUDA device. Otherwise, it returns the CPU device.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.

    Returns:
        torch.device: The device to be used for tensor operations (cuda or cpu).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
        print(f"[+] Device selected: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("[+] Device selected: cpu")
    
    print(f"[+] Seed globally set to {seed}")
    return device


def save_json(data: dict, filepath: str) -> None:
    """
    Save a dictionary as a JSON file with indentation for readability.

    Args:
        data (dict): The dictionary to serialize.
        filepath (str): The destination file path.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> dict:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: The parsed dictionary.
    """
    with open(filepath, 'r') as f:
        return json.load(f)