#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "demo_imgs"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"

GLOBAL_ENCODER_SIZE = 336
ROI_SCALES = (0.18, 0.25, 0.33)
EPS = 1e-8


@dataclass
class RoiResult:
    image_name: str
    roi_scale: float
    x0: int
    y0: int
    x1: int
    y1: int
    width: int
    height: int
    detail_density_original: float
    detail_density_global_view: float
    detail_density_crop_view: float
    detail_recovery_gain: float
    variance_original: float
    variance_global_view: float
    variance_crop_view: float
    variance_recovery_gain: float
    high_freq_original: float
    high_freq_global_view: float
    high_freq_crop_view: float
    high_freq_recovery_gain: float
    ssim_global_vs_original: float
    ssim_crop_vs_original: float
    pixel_fraction: float
    encoder_pixel_gain: float


@dataclass
class ImageSummary:
    image_name: str
    width: int
    height: int
    megapixels: float
    best_roi_scale: float
    best_detail_recovery_gain: float
    best_high_freq_recovery_gain: float
    best_encoder_pixel_gain: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]


def resize_np(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    resized = pil.resize(size, RESAMPLE_BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def down_up(image: np.ndarray, side: int = GLOBAL_ENCODER_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    small = resize_np(image, (side, side))
    return resize_np(small, (w, h))


def box_blur(gray: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return gray.copy()
    pad = radius
    k = 2 * radius + 1
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    h, w = gray.shape
    total = (
        integral[k:k + h, k:k + w]
        - integral[:h, k:k + w]
        - integral[k:k + h, :w]
        + integral[:h, :w]
    )
    return total / (k * k)


def local_variance_map(gray: np.ndarray, radius: int = 7) -> np.ndarray:
    mean = box_blur(gray, radius)
    mean_sq = box_blur(gray * gray, radius)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return var


def edge_map(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def high_freq_energy(gray: np.ndarray) -> float:
    smooth = box_blur(gray, radius=2)
    return float(np.mean(np.abs(gray - smooth)))


def detail_density(gray: np.ndarray) -> float:
    return float(np.mean(edge_map(gray)))


def variance_score(gray: np.ndarray) -> float:
    return float(np.var(gray))


def ssim_like(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float(numerator / (denominator + EPS))


def integral_image(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(0).cumsum(1)


def rect_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def best_roi(gray: np.ndarray, scale: float) -> Tuple[int, int, int, int]:
    h, w = gray.shape
    roi_w = max(32, int(round(w * scale)))
    roi_h = max(32, int(round(h * scale)))
    score = local_variance_map(gray, radius=7) + 0.35 * edge_map(gray)
    integ = integral_image(score)

    step_x = max(8, roi_w // 10)
    step_y = max(8, roi_h // 10)
    best = None
    best_score = -1.0

    xs = list(range(0, max(1, w - roi_w + 1), step_x))
    ys = list(range(0, max(1, h - roi_h + 1), step_y))
    if xs[-1] != w - roi_w:
        xs.append(w - roi_w)
    if ys[-1] != h - roi_h:
        ys.append(h - roi_h)

    for y0 in ys:
        y1 = y0 + roi_h
        for x0 in xs:
            x1 = x0 + roi_w
            value = rect_sum(integ, x0, y0, x1, y1) / (roi_w * roi_h)
            center_bias = 1.0 - 0.15 * (
                abs((x0 + x1) / 2 - w / 2) / max(w / 2, 1)
                + abs((y0 + y1) / 2 - h / 2) / max(h / 2, 1)
            )
            value *= center_bias
            if value > best_score:
                best_score = value
                best = (x0, y0, x1, y1)

    assert best is not None
    return best


def crop(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def crop_view_in_original_resolution(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    region = crop(image, box)
    zoomed = resize_np(region, (w, h))
    return zoomed


def encoder_pixel_gain(image_shape: Tuple[int, int], box: Tuple[int, int, int, int], side: int = GLOBAL_ENCODER_SIZE) -> float:
    h, w = image_shape
    x0, y0, x1, y1 = box
    roi_pixels = max((x1 - x0) * (y1 - y0), 1)
    full_pixels = h * w
    return full_pixels / roi_pixels


def analyze_image(path: Path) -> Tuple[ImageSummary, List[RoiResult], Dict[str, object]]:
    image = load_rgb(path)
    gray = rgb_to_gray(image)
    global_view = down_up(image)
    global_gray = rgb_to_gray(global_view)
    h, w = gray.shape

    roi_results: List[RoiResult] = []
    boxes_for_figure: List[Tuple[float, Tuple[int, int, int, int]]] = []

    for scale in ROI_SCALES:
        box = best_roi(gray, scale)
        boxes_for_figure.append((scale, box))

        crop_original = crop(image, box)
        crop_global = crop(global_view, box)

        crop_view_original = resize_np(crop_original, (w, h))
        crop_view_global = resize_np(crop_global, (w, h))

        crop_view_gray_original = rgb_to_gray(crop_view_original)
        crop_view_gray_global = rgb_to_gray(crop_view_global)

        dd_orig = detail_density(crop_view_gray_original)
        dd_global = detail_density(crop_view_gray_global)
        var_orig = variance_score(crop_view_gray_original)
        var_global = variance_score(crop_view_gray_global)
        hf_orig = high_freq_energy(crop_view_gray_original)
        hf_global = high_freq_energy(crop_view_gray_global)

        pixel_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / (w * h)
        enc_gain = encoder_pixel_gain((h, w), box)

        roi_results.append(
            RoiResult(
                image_name=path.name,
                roi_scale=scale,
                x0=box[0],
                y0=box[1],
                x1=box[2],
                y1=box[3],
                width=box[2] - box[0],
                height=box[3] - box[1],
                detail_density_original=dd_orig,
                detail_density_global_view=dd_global,
                detail_density_crop_view=dd_orig,
                detail_recovery_gain=dd_orig / (dd_global + EPS),
                variance_original=var_orig,
                variance_global_view=var_global,
                variance_crop_view=var_orig,
                variance_recovery_gain=var_orig / (var_global + EPS),
                high_freq_original=hf_orig,
                high_freq_global_view=hf_global,
                high_freq_crop_view=hf_orig,
                high_freq_recovery_gain=hf_orig / (hf_global + EPS),
                ssim_global_vs_original=ssim_like(crop_view_gray_original, crop_view_gray_global),
                ssim_crop_vs_original=1.0,
                pixel_fraction=pixel_fraction,
                encoder_pixel_gain=enc_gain,
            )
        )

    best = max(roi_results, key=lambda r: (r.detail_recovery_gain + r.high_freq_recovery_gain) / 2)
    summary = ImageSummary(
        image_name=path.name,
        width=w,
        height=h,
        megapixels=(w * h) / 1e6,
        best_roi_scale=best.roi_scale,
        best_detail_recovery_gain=best.detail_recovery_gain,
        best_high_freq_recovery_gain=best.high_freq_recovery_gain,
        best_encoder_pixel_gain=best.encoder_pixel_gain,
    )

    payload = {
        "image_name": path.name,
        "shape": {"width": w, "height": h},
        "boxes_for_figure": [
            {"roi_scale": scale, "box": [int(v) for v in box]} for scale, box in boxes_for_figure
        ],
    }
    return summary, roi_results, payload


def draw_roi_overview(path: Path, figure_meta: Dict[str, object]) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = [(255, 80, 80), (80, 220, 120), (70, 140, 255)]
    for color, item in zip(colors, figure_meta["boxes_for_figure"]):
        x0, y0, x1, y1 = item["box"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=6)
        draw.text((x0 + 8, max(0, y0 - 24)), f"scale={item['roi_scale']:.2f}", fill=color)
    out = FIG_DIR / f"{path.stem}_roi_overview.png"
    image.save(out)


def draw_comparison_panel(path: Path, best_row: RoiResult) -> None:
    image = load_rgb(path)
    global_view = down_up(image)
    box = (best_row.x0, best_row.y0, best_row.x1, best_row.y1)
    crop_original = crop(image, box)
    crop_global = crop(global_view, box)
    zoom_original = resize_np(crop_original, (image.shape[1], image.shape[0]))
    zoom_global = resize_np(crop_global, (image.shape[1], image.shape[0]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original image")
    axes[0, 1].imshow(global_view)
    axes[0, 1].set_title(f"Global {GLOBAL_ENCODER_SIZE}x{GLOBAL_ENCODER_SIZE} simulation")
    axes[1, 0].imshow(zoom_global)
    axes[1, 0].set_title("ROI seen through global downsample")
    axes[1, 1].imshow(zoom_original)
    axes[1, 1].set_title("Task-guided ROI crop zoom")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        f"{path.name}: best ROI scale={best_row.roi_scale:.2f}, detail gain={best_row.detail_recovery_gain:.2f}x",
        fontsize=13,
    )
    fig.savefig(FIG_DIR / f"{path.stem}_comparison_panel.png", dpi=180)
    plt.close(fig)


def draw_dataset_overview(image_paths: List[Path], summaries: List[ImageSummary]) -> None:
    fig, axes = plt.subplots(1, len(image_paths), figsize=(6 * len(image_paths), 4), constrained_layout=True)
    if len(image_paths) == 1:
        axes = [axes]
    for ax, path, summary in zip(axes, image_paths, summaries):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(
            f"{path.name}\n{summary.width}x{summary.height} | {summary.megapixels:.2f} MP\nBest ROI scale={summary.best_roi_scale:.2f}",
            fontsize=11,
        )
        ax.axis("off")
    fig.suptitle("Dataset overview: demo images used for the crop-vs-global analysis", fontsize=14)
    fig.savefig(FIG_DIR / "dataset_overview.png", dpi=180)
    plt.close(fig)


def draw_metric_bars(roi_rows: List[RoiResult]) -> None:
    names = [f"{row.image_name}\nscale={row.roi_scale:.2f}" for row in roi_rows]
    detail = [row.detail_recovery_gain for row in roi_rows]
    hf = [row.high_freq_recovery_gain for row in roi_rows]
    enc = [row.encoder_pixel_gain for row in roi_rows]

    x = np.arange(len(roi_rows))
    width = 0.26
    fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax1.bar(x - width, detail, width=width, label="Detail density recovery", color="#4C78A8")
    ax1.bar(x, hf, width=width, label="High-frequency recovery", color="#F58518")
    ax1.bar(x + width, enc, width=width, label="Encoder pixel gain", color="#54A24B")
    ax1.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax1.set_ylabel("Gain relative to global view")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_title("Benefit of task-guided crops over a fixed-resolution global encoder view")
    ax1.legend()
    fig.savefig(FIG_DIR / "metric_comparison.png", dpi=180)
    plt.close(fig)


def draw_scale_tradeoff(roi_rows: List[RoiResult]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for image_name in sorted({r.image_name for r in roi_rows}):
        subset = sorted([r for r in roi_rows if r.image_name == image_name], key=lambda r: r.roi_scale)
        ax.plot(
            [r.roi_scale for r in subset],
            [r.detail_recovery_gain for r in subset],
            marker="o",
            linewidth=2,
            label=image_name,
        )
    ax.set_xlabel("ROI scale (fraction of full image)")
    ax.set_ylabel("Detail recovery gain")
    ax.set_title("Trade-off between crop size and recovered fine detail")
    ax.legend()
    fig.savefig(FIG_DIR / "scale_tradeoff.png", dpi=180)
    plt.close(fig)


def write_csv(rows: List[RoiResult], path: Path) -> None:
    headers = list(asdict(rows[0]).keys())
    lines = [",".join(headers)]
    for row in rows:
        vals = []
        for h in headers:
            v = getattr(row, h)
            if isinstance(v, float):
                vals.append(f"{v:.8f}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary(summaries: List[ImageSummary], roi_rows: List[RoiResult]) -> None:
    best_mean_detail = float(np.mean([r.detail_recovery_gain for r in roi_rows]))
    best_mean_hf = float(np.mean([r.high_freq_recovery_gain for r in roi_rows]))
    payload = {
        "global_encoder_size": GLOBAL_ENCODER_SIZE,
        "roi_scales_tested": list(ROI_SCALES),
        "image_summaries": [asdict(s) for s in summaries],
        "aggregate": {
            "mean_detail_recovery_gain": best_mean_detail,
            "mean_high_frequency_recovery_gain": best_mean_hf,
            "max_detail_recovery_gain": float(max(r.detail_recovery_gain for r in roi_rows)),
            "min_detail_recovery_gain": float(min(r.detail_recovery_gain for r in roi_rows)),
        },
        "interpretation": (
            "Values above 1 indicate that the crop-specific view preserves more local detail than the "
            "single global fixed-resolution view. Encoder pixel gain estimates how many more encoder pixels "
            "are effectively allocated to the selected region when it is processed as a dedicated crop."
        ),
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_manifest() -> None:
    manifest = {
        "input_images": sorted(str(p.relative_to(ROOT)) for p in DATA_DIR.glob("*") if p.is_file()),
        "output_files": sorted(
            [str(p.relative_to(ROOT)) for p in OUTPUT_DIR.rglob("*") if p.is_file()]
            + [str(p.relative_to(ROOT)) for p in FIG_DIR.rglob("*") if p.is_file()]
        )
    }
    (OUTPUT_DIR / "output_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    image_paths = sorted([p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    if not image_paths:
        raise FileNotFoundError("No input images found under data/demo_imgs")

    summaries: List[ImageSummary] = []
    roi_rows: List[RoiResult] = []
    figure_meta: Dict[str, Dict[str, object]] = {}

    for path in image_paths:
        summary, rows, meta = analyze_image(path)
        summaries.append(summary)
        roi_rows.extend(rows)
        figure_meta[path.name] = meta

    write_csv(roi_rows, OUTPUT_DIR / "roi_metrics.csv")
    (OUTPUT_DIR / "image_summaries.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8"
    )
    save_summary(summaries, roi_rows)

    draw_dataset_overview(image_paths, summaries)
    for path in image_paths:
        draw_roi_overview(path, figure_meta[path.name])
        best_row = max([r for r in roi_rows if r.image_name == path.name], key=lambda r: r.detail_recovery_gain)
        draw_comparison_panel(path, best_row)

    draw_metric_bars(roi_rows)
    draw_scale_tradeoff(roi_rows)
    save_manifest()


if __name__ == "__main__":
    main()
