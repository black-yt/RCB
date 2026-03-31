import argparse
import json
import numpy as np
from common import load_dataset, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    data_list = load_dataset(args.input)
    num_nodes = np.array([int(d.num_nodes) for d in data_list])
    num_edges = np.array([int(d.num_edges) for d in data_list])
    y = np.array([int(d.y.item()) if hasattr(d, 'y') else -1 for d in data_list])
    out = {
        'path': args.input,
        'num_graphs': int(len(data_list)),
        'node_feature_dim': int(data_list[0].x.shape[1]),
        'edge_feature_dim': int(data_list[0].edge_attr.shape[1]) if getattr(data_list[0], 'edge_attr', None) is not None else 0,
        'label_counts': {str(int(k)): int((y == k).sum()) for k in np.unique(y)},
        'num_nodes': {
            'mean': float(num_nodes.mean()), 'std': float(num_nodes.std()), 'min': int(num_nodes.min()), 'max': int(num_nodes.max())
        },
        'num_edges': {
            'mean': float(num_edges.mean()), 'std': float(num_edges.std()), 'min': int(num_edges.min()), 'max': int(num_edges.max())
        },
    }
    save_json(out, args.out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
