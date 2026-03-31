import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data


class RealisticCrystalDatasetStub:
    pass


try:
    add_safe_globals([RealisticCrystalDatasetStub])
except Exception:
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RealisticCrystalDataset(torch.utils.data.Dataset):
    def __init__(self, data_list=None, **kwargs):
        self.data_list = data_list or []
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def load_dataset(path: str):
    import data_prepare  # local stub module
    obj = torch.load(path, weights_only=False)
    if hasattr(obj, 'data_list'):
        return list(obj.data_list)
    if isinstance(obj, (list, tuple)):
        return list(obj)
    raise TypeError(f'Unsupported dataset object type: {type(obj)}')


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def tensor_to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)
