"""
Helper Functions for Seed Setting, Device Selection, and JSON I/O

This module provides:
- set_seed_and_device(): globally sets random seeds for Python, NumPy, and PyTorch,
  and returns the appropriate torch.device (CUDA if available, else CPU).
- save_json(): write a dictionary to a JSON file with pretty formatting.
- load_json(): read a JSON file and return its content as a dictionary.

These functions ensure reproducibility and facilitate saving/loading
experiment results and best parameters.
"""

import random
import json
import os
import numpy as np
import torch


def set_seed_and_device(seed: int = 42, device_override: str = None) -> torch.device:
    """
    Set the global random seed for reproducibility and return the appropriate device.

    The function sets the seed for:
        - Python's built-in random module
        - NumPy
        - PyTorch (CPU and CUDA if available)
    It also configures cuDNN to deterministic mode when CUDA is present.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
        device_override (str, optional): Force a specific device ('cuda' or 'cpu').
            If None, automatically selects CUDA if available, otherwise CPU.

    Returns:
        torch.device: The device to be used for tensor operations.
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Configure device
    if device_override is not None:
        if device_override.lower() == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif device_override.lower() == 'cpu':
            device = torch.device('cpu')
        else:
            print(f"[!] Warning: device_override '{device_override}' not valid. Falling back to auto.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Additional CUDA settings for reproducibility
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[+] Device selected: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print("[+] Device selected: cpu")

    print(f"[+] Global seed set to {seed}")
    return device


def save_json(data: dict, filepath: str) -> None:
    """
    Save a dictionary as a JSON file with indentation for readability.

    Args:
        data (dict): The dictionary to serialize.
        filepath (str): The destination file path (directory will be created if needed).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[+] JSON saved to: {filepath}")


def load_json(filepath: str) -> dict:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: The parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"[+] JSON loaded from: {filepath}")
    return data