import argparse
import json
import os
import platform
import shutil
import torch
import torch_geometric
from common import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    total, used, free = shutil.disk_usage('.')
    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'torch_geometric_version': torch_geometric.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'disk_total_gb': total / 1e9,
        'disk_free_gb': free / 1e9,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info['gpu_total_memory_gb'] = props.total_memory / 1e9
    save_json(info, args.out)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
