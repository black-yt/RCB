import os
import yaml
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'flow')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
REPORT_IMG_DIR = os.path.join(BASE_DIR, 'report', 'images')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)


def load_meta_paths(base_dir=DATA_DIR, max_models=50):
    meta_paths = []
    for root, dirs, files in os.walk(base_dir):
        if '_meta.yaml' in files:
            meta_paths.append(os.path.join(root, '_meta.yaml'))
    meta_paths = sorted(meta_paths)[:max_models]
    return meta_paths


def summarize_connectome_and_task(meta_paths):
    records = []
    for mp in meta_paths:
        with open(mp, 'r') as f:
            meta = yaml.safe_load(f)
        cfg = meta.get('config', {})
        net = cfg.get('network', {})
        task = cfg.get('task', {})
        connectome = net.get('connectome', {})
        dynamics = net.get('dynamics', {})
        node_cfg = net.get('node_config', {})
        edge_cfg = net.get('edge_config', {})

        rec = {
            'path': mp,
            'connectome_type': connectome.get('type'),
            'connectome_file': connectome.get('file'),
            'connectome_extent': connectome.get('extent'),
            'neuron_model': dynamics.get('type'),
            'activation': dynamics.get('activation', {}).get('type'),
            'rest_init_mean': node_cfg.get('bias', {}).get('mean'),
            'rest_init_std': node_cfg.get('bias', {}).get('std'),
            'tau_value': node_cfg.get('time_const', {}).get('value'),
            'syn_sign_type': edge_cfg.get('sign', {}).get('type'),
            'syn_strength_scale': edge_cfg.get('syn_strength', {}).get('scale'),
        }

        dataset = task.get('dataset', {})
        decoder = task.get('decoder', {}).get('flow', {})

        rec.update({
            'dataset_type': dataset.get('type'),
            'dataset_tasks': ','.join(dataset.get('tasks', [])),
            'dataset_dt': dataset.get('dt'),
            'dataset_n_frames': dataset.get('n_frames'),
            'decoder_type': decoder.get('type'),
            'decoder_kernel_size': decoder.get('kernel_size'),
            'decoder_const_weight': decoder.get('const_weight'),
            'decoder_dropout': decoder.get('p_dropout'),
            'batch_size': task.get('batch_size'),
            'n_iters': task.get('n_iters'),
            'fold': task.get('fold'),
        })
        records.append(rec)

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, 'meta_summary.csv'), index=False)

    plt.figure(figsize=(8, 4))
    sns.histplot(df['tau_value'].dropna(), bins=20, kde=True)
    plt.xlabel('Membrane time constant (tau)')
    plt.ylabel('Count')
    plt.title('Distribution of neuron time constants across DMN ensemble')
    plt.tight_layout()
    fig_path = os.path.join(REPORT_IMG_DIR, 'tau_distribution.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.countplot(x='decoder_kernel_size', data=df)
    plt.xlabel('Decoder kernel size')
    plt.ylabel('Number of models')
    plt.title('Decoder kernel sizes across ensemble')
    plt.tight_layout()
    fig2_path = os.path.join(REPORT_IMG_DIR, 'decoder_kernel_sizes.png')
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    return df


def summarize_validation_losses(meta_paths):
    losses = []
    for mp in meta_paths:
        model_dir = os.path.dirname(mp)
        val_loss_file = os.path.join(model_dir, 'validation_loss.h5')
        if not os.path.exists(val_loss_file):
            continue
        with h5py.File(val_loss_file, 'r') as f:
            # assume dataset 'loss' or take first dataset
            if 'loss' in f:
                arr = f['loss'][()]
            else:
                key = list(f.keys())[0]
                arr = f[key][()]
        losses.append({'meta_path': mp, 'final_loss': float(np.array(arr).ravel()[-1])})

    import pandas as pd
    df_loss = pd.DataFrame(losses)
    df_loss.to_csv(os.path.join(OUTPUT_DIR, 'validation_losses.csv'), index=False)

    plt.figure(figsize=(6, 4))
    sns.histplot(df_loss['final_loss'], bins=15, kde=True)
    plt.xlabel('Final validation loss')
    plt.ylabel('Number of models')
    plt.title('Distribution of final validation loss across ensemble')
    plt.tight_layout()
    fig_path = os.path.join(REPORT_IMG_DIR, 'validation_loss_distribution.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()

    return df_loss


def inspect_single_model_dynamics(meta_path):
    # This is a placeholder illustrating how one might load and inspect a trained DMN.
    # The checkpoint format is not specified; we treat it as a Torch state_dict if possible
    model_dir = os.path.dirname(meta_path)
    chkpt = os.path.join(model_dir, 'best_chkpt')
    example_trace_path = os.path.join(OUTPUT_DIR, 'example_voltage_trace.npy')

    if os.path.exists(chkpt):
        try:
            state = torch.load(chkpt, map_location='cpu')
            # If state contains voltage statistics, try to extract a small tensor
            if isinstance(state, dict):
                # heuristic: pick the first small tensor to visualize as "activity" surrogate
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.numel() >= 100:
                        arr = v.detach().cpu().flatten().numpy()[:1000]
                        np.save(example_trace_path, arr)
                        plt.figure(figsize=(8, 3))
                        plt.plot(arr)
                        plt.xlabel('Index (surrogate time/neurons)')
                        plt.ylabel('Value')
                        plt.title(f'Example tensor slice from checkpoint: {k}')
                        plt.tight_layout()
                        fig_path = os.path.join(REPORT_IMG_DIR, 'example_checkpoint_trace.png')
                        plt.savefig(fig_path, dpi=200)
                        plt.close()
                        break
        except Exception as e:
            # if loading fails, skip dynamic visualization
            print(f'Failed to load checkpoint {chkpt}: {e}')


if __name__ == '__main__':
    meta_paths = load_meta_paths()
    df_meta = summarize_connectome_and_task(meta_paths)
    df_loss = summarize_validation_losses(meta_paths)
    if len(meta_paths) > 0:
        inspect_single_model_dynamics(meta_paths[0])
