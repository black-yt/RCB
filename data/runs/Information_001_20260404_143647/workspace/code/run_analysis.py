import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw

ROOT = Path('.')
DATA_DIR = ROOT / 'data' / 'demo_imgs'
OUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'
ENCODER_RES = 336
SCALES = [0.2, 0.3, 0.4, 0.5]
GRID_STEP = 0.1
TOPK = 3
SEED = 0
np.random.seed(SEED)
sns.set_theme(style='whitegrid', context='talk')


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def list_images():
    return sorted(DATA_DIR.glob('*.png'))


def load_rgb(path: Path):
    return np.array(Image.open(path).convert('RGB'))


def grayscale(arr):
    return 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]


def gradient_energy(gray):
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    return np.sqrt(gx ** 2 + gy ** 2)


def local_contrast(gray, k=5):
    pad = k // 2
    padded = np.pad(gray, pad, mode='reflect')
    out = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            patch = padded[i:i + k, j:j + k]
            out[i, j] = patch.std()
    return out


def text_likeliness(gray):
    grad = gradient_energy(gray)
    contrast = local_contrast(gray, k=5)
    score = grad * (contrast + 1e-6)
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score


def window_score(score_map, x0, y0, x1, y1):
    patch = score_map[y0:y1, x0:x1]
    if patch.size == 0:
        return -np.inf
    area = patch.shape[0] * patch.shape[1]
    return float(patch.mean() + 0.25 * np.percentile(patch, 95) + 0.1 * math.log(area + 1))


def propose_windows(arr, scales=SCALES, step=GRID_STEP, topk=TOPK):
    h, w = arr.shape[:2]
    gray = grayscale(arr)
    score_map = text_likeliness(gray)
    candidates = []
    for scale in scales:
        ww = max(32, int(w * scale))
        hh = max(32, int(h * scale))
        xs = list(range(0, max(1, w - ww + 1), max(1, int(w * step))))
        ys = list(range(0, max(1, h - hh + 1), max(1, int(h * step))))
        if not xs or xs[-1] != w - ww:
            xs.append(max(0, w - ww))
        if not ys or ys[-1] != h - hh:
            ys.append(max(0, h - hh))
        for x0 in xs:
            for y0 in ys:
                x1, y1 = x0 + ww, y0 + hh
                score = window_score(score_map, x0, y0, x1, y1)
                candidates.append({'scale': scale, 'x0': int(x0), 'y0': int(y0), 'x1': int(x1), 'y1': int(y1), 'score': score})
    candidates = sorted(candidates, key=lambda d: d['score'], reverse=True)
    selected = []
    for cand in candidates:
        keep = True
        for prev in selected:
            inter_x0 = max(cand['x0'], prev['x0'])
            inter_y0 = max(cand['y0'], prev['y0'])
            inter_x1 = min(cand['x1'], prev['x1'])
            inter_y1 = min(cand['y1'], prev['y1'])
            iw = max(0, inter_x1 - inter_x0)
            ih = max(0, inter_y1 - inter_y0)
            inter = iw * ih
            area1 = (cand['x1'] - cand['x0']) * (cand['y1'] - cand['y0'])
            area2 = (prev['x1'] - prev['x0']) * (prev['y1'] - prev['y0'])
            union = area1 + area2 - inter + 1e-6
            if inter / union > 0.5:
                keep = False
                break
        if keep:
            selected.append(cand)
        if len(selected) == topk:
            break
    return score_map, selected


def resize_rgb(arr, size):
    return np.array(Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC))


def metrics_for_view(arr, crop=None):
    full_resized = resize_rgb(arr, ENCODER_RES)
    full_gray = grayscale(full_resized)
    full_grad = gradient_energy(full_gray)
    full_text = text_likeliness(full_gray)
    data = {
        'view': 'global',
        'edge_density': float((full_grad > np.percentile(full_grad, 75)).mean()),
        'mean_gradient': float(full_grad.mean()),
        'text_saliency_mean': float(full_text.mean()),
        'text_saliency_p95': float(np.percentile(full_text, 95)),
    }
    if crop is not None:
        x0, y0, x1, y1 = crop
        crop_arr = arr[y0:y1, x0:x1]
        crop_resized = resize_rgb(crop_arr, ENCODER_RES)
        crop_gray = grayscale(crop_resized)
        crop_grad = gradient_energy(crop_gray)
        crop_text = text_likeliness(crop_gray)
        area_frac = ((x1 - x0) * (y1 - y0)) / (arr.shape[0] * arr.shape[1])
        data.update({
            'crop_edge_density': float((crop_grad > np.percentile(crop_grad, 75)).mean()),
            'crop_mean_gradient': float(crop_grad.mean()),
            'crop_text_saliency_mean': float(crop_text.mean()),
            'crop_text_saliency_p95': float(np.percentile(crop_text, 95)),
            'crop_area_fraction': float(area_frac),
            'detail_gain_gradient': float(crop_grad.mean() / (full_grad.mean() + 1e-8)),
            'detail_gain_text_p95': float(np.percentile(crop_text, 95) / (np.percentile(full_text, 95) + 1e-8)),
        })
    return data


def scale_curve(arr, crop):
    x0, y0, x1, y1 = crop
    crop_arr = arr[y0:y1, x0:x1]
    sizes = [112, 224, 336, 448]
    rows = []
    for size in sizes:
        fg = gradient_energy(grayscale(resize_rgb(arr, size))).mean()
        cg = gradient_energy(grayscale(resize_rgb(crop_arr, size))).mean()
        rows.append({'resolution': size, 'view': 'global', 'mean_gradient': float(fg)})
        rows.append({'resolution': size, 'view': 'crop', 'mean_gradient': float(cg)})
    return rows


def draw_overlay(arr, windows, out_path):
    img = Image.fromarray(arr.astype(np.uint8)).copy()
    draw = ImageDraw.Draw(img)
    colors = ['red', 'lime', 'cyan']
    for idx, w in enumerate(windows):
        draw.rectangle([w['x0'], w['y0'], w['x1'], w['y1']], outline=colors[idx % len(colors)], width=6)
        draw.text((w['x0'] + 5, w['y0'] + 5), f"#{idx+1}", fill=colors[idx % len(colors)])
    img.save(out_path)


def save_crop_panel(arr, windows, out_path):
    fig, axes = plt.subplots(1, len(windows) + 1, figsize=(5 * (len(windows) + 1), 5))
    axes[0].imshow(arr)
    axes[0].set_title('Global image')
    axes[0].axis('off')
    for i, w in enumerate(windows, start=1):
        crop = arr[w['y0']:w['y1'], w['x0']:w['x1']]
        axes[i].imshow(crop)
        axes[i].set_title(f"Crop {i}\nscale={w['scale']:.1f}")
        axes[i].axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_overview(images, out_path):
    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 5))
    if len(images) == 1:
        axes = [axes]
    for ax, (name, arr) in zip(axes, images):
        ax.imshow(arr)
        ax.set_title(f"{name}\n{arr.shape[1]}x{arr.shape[0]}")
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_dirs()
    images = [(p.stem, load_rgb(p)) for p in list_images()]
    save_overview(images, IMG_DIR / 'data_overview.png')

    metric_rows = []
    roi_dump = {}
    scale_rows = []
    summary_rows = []

    for name, arr in images:
        score_map, windows = propose_windows(arr)
        roi_dump[name] = windows
        draw_overlay(arr, windows, IMG_DIR / f'{name}_overlay.png')
        save_crop_panel(arr, windows, IMG_DIR / f'{name}_crops.png')

        Image.fromarray((score_map * 255).astype(np.uint8)).save(IMG_DIR / f'{name}_heatmap_raw.png')
        plt.figure(figsize=(6, 5))
        plt.imshow(score_map, cmap='magma')
        plt.colorbar(label='ROI score')
        plt.title(f'ROI heatmap: {name}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f'{name}_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close()

        top = windows[0]
        m = metrics_for_view(arr, crop=(top['x0'], top['y0'], top['x1'], top['y1']))
        m['image'] = name
        metric_rows.append(m)
        summary_rows.append({
            'image': name,
            'width': arr.shape[1],
            'height': arr.shape[0],
            'selected_scale': top['scale'],
            'crop_area_fraction': m['crop_area_fraction'],
            'detail_gain_gradient': m['detail_gain_gradient'],
            'detail_gain_text_p95': m['detail_gain_text_p95'],
        })
        scale_rows.extend([dict(image=name, **row) for row in scale_curve(arr, (top['x0'], top['y0'], top['x1'], top['y1']))])

    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summary_rows)
    scale_df = pd.DataFrame(scale_rows)
    metrics.to_csv(OUT_DIR / 'metrics.csv', index=False)
    summary.to_csv(OUT_DIR / 'summary.csv', index=False)
    scale_df.to_csv(OUT_DIR / 'scale_curve.csv', index=False)
    with open(OUT_DIR / 'roi_windows.json', 'w') as f:
        json.dump(roi_dump, f, indent=2)

    long_df = pd.DataFrame({
        'image': metrics['image'].tolist() * 2,
        'view': ['global'] * len(metrics) + ['crop'] * len(metrics),
        'mean_gradient': pd.concat([metrics['mean_gradient'], metrics['crop_mean_gradient']], ignore_index=True),
        'text_saliency_p95': pd.concat([metrics['text_saliency_p95'], metrics['crop_text_saliency_p95']], ignore_index=True),
    })
    long_df.to_csv(OUT_DIR / 'comparison_long.csv', index=False)

    plt.figure(figsize=(8, 6))
    plot_df = long_df.melt(id_vars=['image', 'view'], value_vars=['mean_gradient', 'text_saliency_p95'], var_name='metric', value_name='value')
    sns.barplot(data=plot_df, x='metric', y='value', hue='view', errorbar=None)
    plt.title('Global resize vs task-guided crop')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'main_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=summary.melt(id_vars=['image'], value_vars=['detail_gain_gradient', 'detail_gain_text_p95'], var_name='metric', value_name='gain'), x='image', y='gain', hue='metric', errorbar=None)
    plt.axhline(1.0, linestyle='--', color='black', linewidth=1)
    plt.title('Relative detail gain from selected crop')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'detail_gain.png', dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=scale_df, x='resolution', y='mean_gradient', hue='view', style='image', markers=True, dashes=False)
    plt.title('Detail preservation across encoder resolutions')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'scale_validation.png', dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=summary, x='crop_area_fraction', y='detail_gain_gradient', hue='image', s=120)
    for _, row in summary.iterrows():
        plt.text(row['crop_area_fraction'] + 0.005, row['detail_gain_gradient'] + 0.01, row['image'])
    plt.title('Smaller ROI can recover higher local detail')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'area_vs_gain.png', dpi=200, bbox_inches='tight')
    plt.close()

    with open(OUT_DIR / 'analysis_notes.md', 'w') as f:
        f.write('# Analysis Notes\n\n')
        f.write(f'- Fixed encoder resolution: {ENCODER_RES}x{ENCODER_RES}\n')
        f.write('- ROI scoring combines gradient energy and local contrast as a training-free proxy for fine detail / text density.\n')
        f.write('- Selected the top non-overlapping crop per image for quantitative comparison.\n')
        f.write('\n## Summary Table\n\n')
        f.write(summary.to_csv(index=False))


if __name__ == '__main__':
    main()
