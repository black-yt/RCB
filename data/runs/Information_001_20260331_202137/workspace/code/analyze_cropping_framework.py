"""Analysis of task-guided cropping framework demo images.

This script:
1. Loads demo images used in experiment 1.
2. Computes basic image statistics (size, aspect ratio).
3. Generates visualizations to support the report.

All outputs are saved under ../outputs and ../report/images.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path('.').resolve()
DATA_DIR = ROOT / 'data' / 'demo_imgs'
OUTPUT_DIR = ROOT / 'outputs'
FIG_DIR = ROOT / 'report' / 'images'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_images():
    images = {}
    for name in ['demo1.png', 'demo2.png', 'method_case.png']:
        path = DATA_DIR / name
        img = Image.open(path).convert('RGB')
        images[name] = img
    return images


def summarize_images(images):
    rows = []
    for name, img in images.items():
        w, h = img.size
        rows.append((name, w, h, w / h))
    import csv
    out_csv = OUTPUT_DIR / 'image_summary.csv'
    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'width', 'height', 'aspect_ratio'])
        writer.writerows(rows)
    return rows


def plot_image_sizes(summary):
    names = [r[0] for r in summary]
    widths = [r[1] for r in summary]
    heights = [r[2] for r in summary]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, widths, width, label='Width')
    plt.bar(x + width/2, heights, width, label='Height')
    plt.xticks(x, names, rotation=20)
    plt.ylabel('Pixels')
    plt.title('Image resolutions of demo figures')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'image_resolutions.png', dpi=300)
    plt.close()


def show_composite(images):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'demo_images_overview.png', dpi=300)
    plt.close()


def main():
    images = load_images()
    summary = summarize_images(images)
    plot_image_sizes(summary)
    show_composite(images)


if __name__ == '__main__':
    main()
