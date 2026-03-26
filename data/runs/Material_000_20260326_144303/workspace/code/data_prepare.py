"""
Minimal stub for loading .pt files that reference data_prepare module.
"""
import sys
import types
import torch
from torch_geometric.data import Data


class RealisticCrystalDataset:
    """Stub class to allow loading pickled datasets."""
    pass


def load_dataset(path):
    """Load a .pt file that uses RealisticCrystalDataset."""
    # Register stub in sys.modules if not already done
    if 'data_prepare' not in sys.modules:
        dm = types.ModuleType('data_prepare')
        dm.RealisticCrystalDataset = RealisticCrystalDataset
        sys.modules['data_prepare'] = dm

    dataset = torch.load(path, map_location='cpu', weights_only=False)
    return dataset.data_list
