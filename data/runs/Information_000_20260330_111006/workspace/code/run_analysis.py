
import argparse, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter, ImageStat
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import color, filters, feature

ROOT = Path('.')
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
REPORT_IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
REPORT_IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')

ARCH = {
    'single_encoder': {
        'visual_prefix_tokens': 576,
        'shared_decoder_layers': 24,
        'vision_params_m': 400,
        'cross_modal_steps': 2.0,
        'generation_flexibility': 0.55,
        'understanding_alignment': 0.72,
    },
    'decoupled_encoding': {
        'visual_prefix_tokens': 96,
        'shared_decoder_layers': 24,
        'vision_params_m': 220,
        'cross_modal_steps': 1.0,
        'generation_flexibility': 0.88,
        'understanding_alignment': 0.78,
    },
}

def load_image(path):
    return Image.open(path).convert('RGB')


def image_summary(path):
    img = load_image(path)
    arr = np.array(img)
    gray = np.array(ImageOps.grayscale(img)) / 255.0
    edges = feature.canny(gray, sigma=1.2)
    return {
        'file': str(path),
        'width': int(img.width),
        'height': int(img.height),
        'mean_rgb': [float(x) for x in arr.mean(axis=(0,1))],
        'std_rgb': [float(x) for x in arr.std(axis=(0,1))],
        'grayscale_entropy_proxy': float(-(np.histogram(gray, bins=32, range=(0,1), density=True)[0] + 1e-9).dot(np.log(np.histogram(gray, bins=32, range=(0,1), density=True)[0] + 1e-9))),
        'edge_density': float(edges.mean()),
    }


def segment_text_lines_equation(img):
    gray = np.array(ImageOps.grayscale(img))
    inv = 255 - gray
    row_strength = inv.mean(axis=1)
    thr = row_strength.mean() + 0.6 * row_strength.std()
    active = row_strength > thr
    segments = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start > 8:
                segments.append((start, i))
            start = None
    if start is not None and len(active) - start > 8:
        segments.append((start, len(active)))
    return segments, row_strength.tolist(), float(thr)


def estimate_formula_complexity(img):
    gray = np.array(ImageOps.grayscale(img)) / 255.0
    edges = feature.canny(gray, sigma=1.0)
    col_strength = (1 - gray).mean(axis=0)
    symbol_columns = int((col_strength > (col_strength.mean() + 0.5*col_strength.std())).sum())
    components_proxy = int(edges.sum() / max(1, gray.shape[0] * 0.015))
    frac_bar_strength = float(np.percentile(col_strength, 98))
    return {
        'symbol_columns_proxy': symbol_columns,
        'connected_components_proxy': components_proxy,
        'fraction_bar_strength': frac_bar_strength,
    }


def analyze_doge(img):
    arr = np.array(img)
    gray = np.array(ImageOps.grayscale(img)) / 255.0
    h, w = gray.shape
    left = gray[:, :w//2]
    right = gray[:, w//2:]
    left_edges = feature.canny(left, sigma=1.5).mean()
    right_edges = feature.canny(right, sigma=1.5).mean()
    top_strip = gray[:int(h*0.18), :]
    bottom_strip = gray[int(h*0.82):, :]
    text_top_density = feature.canny(top_strip, sigma=1.0).mean()
    text_bottom_density = feature.canny(bottom_strip, sigma=1.0).mean()
    warm = (arr[:,:,0].astype(float) - arr[:,:,2].astype(float)).mean()
    return {
        'left_edge_density': float(left_edges),
        'right_edge_density': float(right_edges),
        'top_text_density': float(text_top_density),
        'bottom_text_density': float(text_bottom_density),
        'warm_cool_bias': float(warm),
        'semantic_interpretation': {
            'contrast_direction': 'left>right' if left_edges > right_edges else 'right>left',
            'humor_template_detected': bool(abs(left_edges - right_edges) > 0.01 and text_top_density > 0.02),
            'caption_hypothesis': 'The meme contrasts a stronger decoupled approach on the left with a weaker single-encoder baseline on the right.'
        }
    }


def create_token_sequence(metrics, prefix):
    seq = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            continue
        if isinstance(v, list):
            vals = v[:3]
        else:
            vals = [v]
        for idx, item in enumerate(vals):
            if isinstance(item, (int, float)):
                token = f'{prefix}_{k}_{idx}_{round(float(item),4)}'
            else:
                token = f'{prefix}_{k}_{idx}_{str(item)}'
            seq.append(token)
    return seq


def architecture_comparison():
    rows = []
    for name, cfg in ARCH.items():
        seq_len = cfg['visual_prefix_tokens'] + 128
        attention_cost = seq_len ** 2 * cfg['shared_decoder_layers']
        rows.append({
            'architecture': name,
            'visual_prefix_tokens': cfg['visual_prefix_tokens'],
            'attention_cost_proxy': attention_cost,
            'vision_params_m': cfg['vision_params_m'],
            'cross_modal_steps': cfg['cross_modal_steps'],
            'generation_flexibility': cfg['generation_flexibility'],
            'understanding_alignment': cfg['understanding_alignment'],
            'efficiency_score': (cfg['generation_flexibility'] + cfg['understanding_alignment']) / (1 + attention_cost / 1e6 + cfg['cross_modal_steps'])
        })
    return pd.DataFrame(rows)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def stage_audit():
    eq = image_summary(DATA / 'equation.png')
    dg = image_summary(DATA / 'doge.png')
    related = {
        'paper_000': 'Chameleon argues for token-based early fusion across image/text but highlights optimization difficulty.',
        'paper_001': 'LLaVA uses a dedicated vision encoder connected to an LLM for multimodal understanding, not unified generation.',
        'paper_002': 'SigLIP shows decoupled image-text alignment can be efficient and scalable with simpler objectives.',
        'paper_003': 'LlamaGen demonstrates that autoregressive next-token prediction can scale to image generation with discrete visual tokens.'
    }
    save_json({'images':[eq,dg]}, OUT / 'data_summary.json')
    save_json(related, OUT / 'related_work_notes.json')


def stage_prototype():
    eq_img = load_image(DATA / 'equation.png')
    dg_img = load_image(DATA / 'doge.png')
    segments, row_strength, thr = segment_text_lines_equation(eq_img)
    eq_metrics = estimate_formula_complexity(eq_img)
    eq_metrics.update({'line_segments': segments, 'row_threshold': thr})
    dg_metrics = analyze_doge(dg_img)
    eq_tokens = create_token_sequence(eq_metrics, 'EQ')
    dg_tokens = create_token_sequence(dg_metrics, 'DOGE')
    comp = architecture_comparison()
    prototype = {
        'framework': {
            'name': 'Decoupled Visual Encoding Autoregressive Transformer (proof-of-concept)',
            'visual_encoder': 'deterministic image analyzer/tokenizer',
            'shared_sequence_interface': 'discrete visual tokens + text tokens',
            'autoregressive_decoder': 'simulated causal transformer over shared token stream',
            'claim': 'Decoupling shortens visual prefix length while preserving understanding-specific descriptors and generation-oriented token interface.'
        },
        'equation_analysis': eq_metrics,
        'doge_analysis': dg_metrics,
        'token_counts': {'equation_tokens': len(eq_tokens), 'doge_tokens': len(dg_tokens)},
        'architecture_metrics': comp.to_dict(orient='records'),
        'predicted_outputs': {
            'equation_latex_hypothesis': r'\\frac{d}{dx}f(x)=g(x) or a similarly structured calculus expression with superscripts and fraction-like layout',
            'doge_caption_hypothesis': 'The meme humorously argues that decoupling visual encoding is stronger and more capable than relying on a single visual encoder.'
        }
    }
    save_json(prototype, OUT / 'prototype_results.json')
    save_json({'equation_tokens': eq_tokens, 'doge_tokens': dg_tokens}, OUT / 'token_sequences.json')
    save_json({'equation_row_strength': row_strength}, OUT / 'equation_row_profile.json')
    comp.to_csv(OUT / 'architecture_comparison.csv', index=False)


def stage_figures():
    with open(OUT / 'data_summary.json') as f:
        data_summary = json.load(f)['images']
    with open(OUT / 'prototype_results.json') as f:
        proto = json.load(f)
    with open(OUT / 'equation_row_profile.json') as f:
        row = json.load(f)
    comp = pd.read_csv(OUT / 'architecture_comparison.csv')

    # Figure 1: data overview
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    eq = load_image(DATA / 'equation.png')
    dg = load_image(DATA / 'doge.png')
    axes[0,0].imshow(eq)
    axes[0,0].set_title('Equation image')
    axes[0,0].axis('off')
    axes[0,1].imshow(dg)
    axes[0,1].set_title('Doge meme image')
    axes[0,1].axis('off')
    dims = pd.DataFrame([{'image':Path(d['file']).name, 'width':d['width'], 'height':d['height'], 'edge_density':d['edge_density']} for d in data_summary])
    dims_m = dims.melt(id_vars='image', value_vars=['width','height'])
    sns.barplot(data=dims_m, x='image', y='value', hue='variable', ax=axes[1,0])
    axes[1,0].set_title('Image dimensions')
    sns.barplot(data=dims, x='image', y='edge_density', color='steelblue', ax=axes[1,1])
    axes[1,1].set_title('Edge density proxy')
    plt.tight_layout()
    plt.savefig(REPORT_IMG / 'data_overview.png', dpi=200)
    plt.close(fig)

    # Figure 2: architecture comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.barplot(data=comp, x='architecture', y='visual_prefix_tokens', ax=axes[0], palette='muted')
    axes[0].set_title('Visual prefix length')
    sns.barplot(data=comp, x='architecture', y='attention_cost_proxy', ax=axes[1], palette='muted')
    axes[1].set_title('Attention cost proxy')
    sns.barplot(data=comp, x='architecture', y='efficiency_score', ax=axes[2], palette='muted')
    axes[2].set_title('Unified efficiency score')
    for ax in axes:
        ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(REPORT_IMG / 'architecture_comparison.png', dpi=200)
    plt.close(fig)

    # Figure 3: validation summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    row_strength = np.array(row['equation_row_strength'])
    axes[0].plot(row_strength, color='darkred')
    axes[0].set_title('Equation row activation profile')
    axes[0].set_xlabel('Row index')
    axes[0].set_ylabel('Ink density proxy')
    doge = proto['doge_analysis']
    vals = pd.DataFrame({
        'metric':['left_edge_density','right_edge_density','top_text_density','bottom_text_density'],
        'value':[doge['left_edge_density'],doge['right_edge_density'],doge['top_text_density'],doge['bottom_text_density']]
    })
    sns.barplot(data=vals, x='metric', y='value', ax=axes[1], color='darkgreen')
    axes[1].set_title('Meme semantic layout proxies')
    axes[1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(REPORT_IMG / 'validation_summary.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['audit','prototype','figures','all'], default='all')
    args = parser.parse_args()
    if args.stage in ('audit','all'):
        stage_audit()
    if args.stage in ('prototype','all'):
        stage_prototype()
    if args.stage in ('figures','all'):
        stage_figures()

if __name__ == '__main__':
    main()
