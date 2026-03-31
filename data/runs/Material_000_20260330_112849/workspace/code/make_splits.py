import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from common import load_dataset, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--strategy', default='stratified')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--out', required=True)
    parser.add_argument('--val_size', type=float, default=0.2)
    args = parser.parse_args()
    ensure_dir(args.out)
    data_list = load_dataset(args.input)
    y = np.array([int(d.y.item()) for d in data_list])
    for seed in range(args.seeds):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
        out = {'seed': seed, 'train_idx': train_idx.tolist(), 'val_idx': val_idx.tolist()}
        with open(Path(args.out) / f'split_seed{seed}.json', 'w') as f:
            json.dump(out, f, indent=2)
        print(f'saved split {seed}: train={len(train_idx)} val={len(val_idx)} positives_train={int(y[train_idx].sum())} positives_val={int(y[val_idx].sum())}')


if __name__ == '__main__':
    main()
