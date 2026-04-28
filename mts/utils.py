import json
from typing import Any

import yaml
import random
import numpy as np
import torch


def load_json_data(filepath: str) -> Any:
    with open(filepath, encoding="utf-8") as inp:
        return json.load(inp)


def load_yaml_data(filepath: str) -> Any:
    with open(filepath, encoding="utf-8") as inp:
        return yaml.load(inp, Loader=yaml.FullLoader)


def save_json_data(data: Any, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as outp:
        return json.dump(data, outp, indent=2, ensure_ascii=False)


def set_seed(seed: int):
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    np.random.seed(seed)  # For numpy operations
    random.seed(seed)  # For random operations
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables performance optimizations for deterministic results
