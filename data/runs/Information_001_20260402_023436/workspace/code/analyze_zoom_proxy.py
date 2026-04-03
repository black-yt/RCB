#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs"
REPORT_IMG_DIR = WORKSPACE / "report" / "images"
MPL_DIR = OUTPUT_DIR / "mplconfig"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_IMG_DIR.mkdir(exist_ok=True)
MPL_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import cv2
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from skimage.filters import sobel
from skimage.filters.rank import entropy
from skimage.morphology import disk


DATA_DIR = WORKSPACE / "data" / "demo_imgs"
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SEED = 7
RNG = np.random.default_rng(SEED)
DEFAULT_ENCODER_RES = 336
DEFAULT_NUM_CROPS = 4
RANDOM_TRIALS = 8
EVAL_HOTSPOTS = 6
CROP_FRACTIONS = (0.12, 0.16, 0.2)
ENCODER_RESOLUTIONS = (224, 336, 448)
BUDGETS = (0, 1, 2, 3, 4, 5, 6)

sns.set_theme(style="whitegrid", context="talk")


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 0.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(image)
    return np.array(pil.resize((size, size), Image.Resampling.BICUBIC))


def resize_to_shape(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(image)
    return np.array(pil.resize((width, height), Image.Resampling.BICUBIC))


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= arr.min()
    denom = arr.max() - arr.min()
    if denom <= 1e-8:
        return np.zeros_like(arr)
    return arr / denom


def build_detail_map(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    h, w = gray.shape
    max_side = max(h, w)
    scale = 1.0 if max_side <= 1024 else 1024.0 / max_side
    small_w = max(64, int(round(w * scale)))
    small_h = max(64, int(round(h * scale)))
    gray_small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    ent_small = entropy(gray_small, disk(3)).astype(np.float32)
    grad_small = sobel(gray_small.astype(np.float32) / 255.0).astype(np.float32)
    mean = cv2.GaussianBlur(gray_small.astype(np.float32), (0, 0), sigmaX=5)
    sq_mean = cv2.GaussianBlur((gray_small.astype(np.float32) ** 2), (0, 0), sigmaX=5)
    contrast_small = np.sqrt(np.clip(sq_mean - mean**2, 0.0, None))

    detail_small = (
        0.45 * normalize_map(ent_small)
        + 0.4 * normalize_map(grad_small)
        + 0.15 * normalize_map(contrast_small)
    )
    detail = cv2.resize(detail_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return normalize_map(detail)


def integral_image(arr: np.ndarray) -> np.ndarray:
    return cv2.integral(arr.astype(np.float64))


def window_mean(ii: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    total = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]
    area = max((x2 - x1) * (y2 - y1), 1)
    return float(total / area)


def box_iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def non_overlapping_topk(candidates: list[Box], k: int, iou_threshold: float = 0.2) -> list[Box]:
    selected: list[Box] = []
    for box in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(box_iou(box, chosen) <= iou_threshold for chosen in selected):
            selected.append(box)
        if len(selected) >= k:
            break
    return selected


def generate_candidates(detail_map: np.ndarray) -> list[Box]:
    h, w = detail_map.shape
    ii = integral_image(detail_map)
    candidates: list[Box] = []
    for frac in CROP_FRACTIONS:
        size = max(48, int(round(min(h, w) * frac)))
        stride = max(24, size // 3)
        for y1 in range(0, max(h - size + 1, 1), stride):
            for x1 in range(0, max(w - size + 1, 1), stride):
                x2 = min(w, x1 + size)
                y2 = min(h, y1 + size)
                score = window_mean(ii, x1, y1, x2, y2)
                candidates.append(Box(x1, y1, x2, y2, score))
        if (h - size) % stride != 0 or (w - size) % stride != 0:
            x1 = max(0, w - size)
            y1 = max(0, h - size)
            candidates.append(Box(x1, y1, w, h, window_mean(ii, x1, y1, w, h)))
    return candidates


def guided_boxes(detail_map: np.ndarray, k: int) -> list[Box]:
    if k <= 0:
        return []
    return non_overlapping_topk(generate_candidates(detail_map), k)


def evaluation_hotspots(detail_map: np.ndarray, k: int = EVAL_HOTSPOTS) -> list[Box]:
    h, w = detail_map.shape
    ii = integral_image(detail_map)
    candidates: list[Box] = []
    for frac in (0.08, 0.1, 0.12):
        size = max(40, int(round(min(h, w) * frac)))
        stride = max(20, size // 2)
        for y1 in range(0, max(h - size + 1, 1), stride):
            for x1 in range(0, max(w - size + 1, 1), stride):
                x2 = min(w, x1 + size)
                y2 = min(h, y1 + size)
                candidates.append(Box(x1, y1, x2, y2, window_mean(ii, x1, y1, x2, y2)))
    return non_overlapping_topk(candidates, k, iou_threshold=0.15)


def lattice_boxes(shape: tuple[int, int], template_boxes: list[Box]) -> list[Box]:
    if not template_boxes:
        return []
    h, w = shape
    k = len(template_boxes)
    rows = max(1, int(math.floor(math.sqrt(k))))
    cols = int(math.ceil(k / rows))
    xs = np.linspace(0.15, 0.85, cols)
    ys = np.linspace(0.15, 0.85, rows)
    centers = [(cx, cy) for cy in ys for cx in xs][:k]

    boxes = []
    for (cx, cy), template in zip(centers, template_boxes):
        bw, bh = template.width, template.height
        center_x = int(round(cx * w))
        center_y = int(round(cy * h))
        x1 = int(np.clip(center_x - bw // 2, 0, max(w - bw, 0)))
        y1 = int(np.clip(center_y - bh // 2, 0, max(h - bh, 0)))
        boxes.append(Box(x1, y1, x1 + bw, y1 + bh, 0.0))
    return boxes


def random_boxes(shape: tuple[int, int], template_boxes: list[Box], rng: np.random.Generator) -> list[Box]:
    if not template_boxes:
        return []
    h, w = shape
    boxes: list[Box] = []
    for template in template_boxes:
        bw, bh = template.width, template.height
        best: Box | None = None
        for _ in range(40):
            x1 = int(rng.integers(0, max(w - bw, 1)))
            y1 = int(rng.integers(0, max(h - bh, 1)))
            proposal = Box(x1, y1, x1 + bw, y1 + bh, 0.0)
            if all(box_iou(proposal, existing) < 0.15 for existing in boxes):
                best = proposal
                break
            if best is None:
                best = proposal
        boxes.append(best if best is not None else proposal)
    return boxes


def reconstruct_with_crops(image: np.ndarray, boxes: list[Box], encoder_res: int) -> np.ndarray:
    h, w = image.shape[:2]
    reconstructed = resize_to_shape(resize_square(image, encoder_res), w, h)
    for box in boxes:
        patch = image[box.y1 : box.y2, box.x1 : box.x2]
        patch_rec = resize_to_shape(resize_square(patch, encoder_res), box.width, box.height)
        reconstructed[box.y1 : box.y2, box.x1 : box.x2] = patch_rec
    return reconstructed


def patch_detail_score(patch: np.ndarray) -> float:
    gray = to_gray(patch).astype(np.float32) / 255.0
    grad_energy = float(np.mean(np.abs(sobel(gray))))
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(fft)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    threshold = 0.18 * min(h, w)
    high_freq = mag[radius >= threshold]
    hf_energy = float(np.mean(high_freq))
    return grad_energy + 0.02 * hf_energy


def hotspot_recovery(original: np.ndarray, reconstructed: np.ndarray, hotspots: list[Box]) -> float:
    ratios = []
    for box in hotspots:
        orig_patch = original[box.y1 : box.y2, box.x1 : box.x2]
        rec_patch = reconstructed[box.y1 : box.y2, box.x1 : box.x2]
        baseline = patch_detail_score(orig_patch)
        if baseline <= 1e-8:
            continue
        ratios.append(min(1.0, patch_detail_score(rec_patch) / baseline))
    return float(np.mean(ratios)) if ratios else float("nan")


def weighted_detail_fidelity(original: np.ndarray, reconstructed: np.ndarray, detail_map: np.ndarray) -> float:
    orig = original.astype(np.float32)
    rec = reconstructed.astype(np.float32)
    weights = detail_map[..., None] + 1e-6
    mae = np.sum(weights * np.abs(orig - rec)) / np.sum(weights)
    return float(1.0 - mae / 255.0)


def mean_crop_fraction(boxes: list[Box], shape: tuple[int, int]) -> float:
    if not boxes:
        return 0.0
    h, w = shape
    return float(np.mean([b.area / (h * w) for b in boxes]))


def draw_boxes(ax: plt.Axes, image: np.ndarray, boxes: list[Box], title: str) -> None:
    ax.imshow(image)
    for idx, box in enumerate(boxes, start=1):
        rect = plt.Rectangle(
            (box.x1, box.y1),
            box.width,
            box.height,
            fill=False,
            linewidth=2,
            edgecolor="#ff6b35",
        )
        ax.add_patch(rect)
        ax.text(
            box.x1 + 4,
            box.y1 + 18,
            str(idx),
            color="white",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ff6b35", edgecolor="none"),
        )
    ax.set_title(title)
    ax.axis("off")


def save_input_overview(images: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(16, 5))
    if len(images) == 1:
        axes = [axes]
    for ax, (name, image) in zip(axes, images.items()):
        h, w = image.shape[:2]
        ax.imshow(image)
        ax.set_title(f"{name}\n{w}x{h}")
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "data_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_overlay_figure(images: dict[str, np.ndarray], selected: dict[str, list[Box]]) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(17, 5.5))
    if len(images) == 1:
        axes = [axes]
    for ax, (name, image) in zip(axes, images.items()):
        draw_boxes(ax, image, selected[name], f"{name}: guided crops")
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "guided_crop_overlays.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_reconstruction(images: dict[str, np.ndarray], guided_selection: dict[str, list[Box]]) -> None:
    fig, axes = plt.subplots(len(images), 3, figsize=(12, 4 * len(images)))
    if len(images) == 1:
        axes = np.expand_dims(axes, 0)
    for row, (name, image) in enumerate(images.items()):
        global_rec = reconstruct_with_crops(image, [], DEFAULT_ENCODER_RES)
        guided_rec = reconstruct_with_crops(image, guided_selection[name], DEFAULT_ENCODER_RES)
        titles = [f"{name}: original", "global-only", "guided zoom"]
        for col, panel in enumerate((image, global_rec, guided_rec)):
            axes[row, col].imshow(panel)
            axes[row, col].set_title(titles[col])
            axes[row, col].axis("off")
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "qualitative_reconstructions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(summary: pd.DataFrame) -> None:
    metrics = summary.melt(
        id_vars=["method"],
        value_vars=["hotspot_recovery", "weighted_detail_fidelity"],
        var_name="metric",
        value_name="value",
    )
    metric_names = {
        "hotspot_recovery": "Hotspot recovery",
        "weighted_detail_fidelity": "Weighted detail fidelity",
    }
    metrics["metric"] = metrics["metric"].map(metric_names)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, metric_names.values()):
        subset = metrics[metrics["metric"] == metric]
        sns.barplot(data=subset, x="method", y="value", palette="crest", ax=ax)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "method_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_budget_sensitivity(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(
        data=df,
        x="num_crops",
        y="hotspot_recovery",
        hue="method",
        marker="o",
        ax=axes[0],
    )
    sns.lineplot(
        data=df,
        x="num_crops",
        y="weighted_detail_fidelity",
        hue="method",
        marker="o",
        ax=axes[1],
    )
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    axes[0].set_title("Crop budget vs hotspot recovery")
    axes[1].set_title("Crop budget vs weighted fidelity")
    axes[0].legend(title="")
    axes[1].legend(title="")
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "budget_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_resolution_sensitivity(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(
        data=df,
        x="encoder_resolution",
        y="hotspot_recovery",
        hue="method",
        marker="o",
        ax=axes[0],
    )
    sns.lineplot(
        data=df,
        x="encoder_resolution",
        y="weighted_detail_fidelity",
        hue="method",
        marker="o",
        ax=axes[1],
    )
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    axes[0].set_title("Encoder resolution sensitivity")
    axes[1].set_title("Weighted fidelity by encoder size")
    axes[0].legend(title="")
    axes[1].legend(title="")
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "resolution_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_images() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[Box]], dict[str, np.ndarray]]:
    image_paths = sorted(DATA_DIR.glob("*.png"))
    images = {path.stem: load_image(path) for path in image_paths}
    detail_maps = {name: build_detail_map(image) for name, image in images.items()}
    guided_selection = {name: guided_boxes(detail_maps[name], DEFAULT_NUM_CROPS) for name in images}
    hotspots = {name: evaluation_hotspots(detail_maps[name]) for name in images}

    records = []
    per_image_records = []
    budget_records = []
    resolution_records = []

    for name, image in images.items():
        shape = image.shape[:2]
        detail_map = detail_maps[name]
        template_boxes = guided_selection[name]
        methods = {
            "global_only": [],
            "uniform_crops": lattice_boxes(shape, template_boxes),
            "guided_crops": template_boxes,
        }

        random_metrics = []
        for trial in range(RANDOM_TRIALS):
            trial_boxes = random_boxes(shape, template_boxes, np.random.default_rng(SEED + trial))
            reconstructed = reconstruct_with_crops(image, trial_boxes, DEFAULT_ENCODER_RES)
            random_metrics.append(
                {
                    "hotspot_recovery": hotspot_recovery(image, reconstructed, hotspots[name]),
                    "weighted_detail_fidelity": weighted_detail_fidelity(image, reconstructed, detail_map),
                }
            )
        methods["random_crops"] = None

        for method, boxes in methods.items():
            if method == "random_crops":
                hotspot_metric = float(np.mean([m["hotspot_recovery"] for m in random_metrics]))
                fidelity_metric = float(np.mean([m["weighted_detail_fidelity"] for m in random_metrics]))
                crop_fraction = mean_crop_fraction(template_boxes, shape)
            else:
                reconstructed = reconstruct_with_crops(image, boxes, DEFAULT_ENCODER_RES)
                hotspot_metric = hotspot_recovery(image, reconstructed, hotspots[name])
                fidelity_metric = weighted_detail_fidelity(image, reconstructed, detail_map)
                crop_fraction = mean_crop_fraction(boxes, shape)

            record = {
                "image": name,
                "method": method,
                "encoder_resolution": DEFAULT_ENCODER_RES,
                "num_crops": 0 if method == "global_only" else len(template_boxes),
                "hotspot_recovery": hotspot_metric,
                "weighted_detail_fidelity": fidelity_metric,
                "mean_crop_fraction": crop_fraction,
            }
            records.append(record)
            per_image_records.append(record)

        for budget in BUDGETS:
            template_for_budget = guided_boxes(detail_map, budget)
            lattice_for_budget = lattice_boxes(shape, template_for_budget)
            methods_budget = {
                "guided_crops": template_for_budget,
                "uniform_crops": lattice_for_budget,
            }
            for method, boxes in methods_budget.items():
                reconstructed = reconstruct_with_crops(image, boxes, DEFAULT_ENCODER_RES)
                budget_records.append(
                    {
                        "image": name,
                        "method": method,
                        "num_crops": budget,
                        "hotspot_recovery": hotspot_recovery(image, reconstructed, hotspots[name]),
                        "weighted_detail_fidelity": weighted_detail_fidelity(image, reconstructed, detail_map),
                    }
                )
            random_budget_metrics = []
            for trial in range(RANDOM_TRIALS):
                boxes = random_boxes(shape, template_for_budget, np.random.default_rng(1000 + budget * 100 + trial))
                reconstructed = reconstruct_with_crops(image, boxes, DEFAULT_ENCODER_RES)
                random_budget_metrics.append(
                    (
                        hotspot_recovery(image, reconstructed, hotspots[name]),
                        weighted_detail_fidelity(image, reconstructed, detail_map),
                    )
                )
            budget_records.append(
                {
                    "image": name,
                    "method": "random_crops",
                    "num_crops": budget,
                    "hotspot_recovery": float(np.mean([m[0] for m in random_budget_metrics])) if random_budget_metrics else hotspot_recovery(image, reconstruct_with_crops(image, [], DEFAULT_ENCODER_RES), hotspots[name]),
                    "weighted_detail_fidelity": float(np.mean([m[1] for m in random_budget_metrics])) if random_budget_metrics else weighted_detail_fidelity(image, reconstruct_with_crops(image, [], DEFAULT_ENCODER_RES), detail_map),
                }
            )

        for encoder_resolution in ENCODER_RESOLUTIONS:
            methods_resolution = {
                "global_only": [],
                "uniform_crops": lattice_boxes(shape, template_boxes),
                "guided_crops": template_boxes,
            }
            for method, boxes in methods_resolution.items():
                reconstructed = reconstruct_with_crops(image, boxes, encoder_resolution)
                resolution_records.append(
                    {
                        "image": name,
                        "method": method,
                        "encoder_resolution": encoder_resolution,
                        "hotspot_recovery": hotspot_recovery(image, reconstructed, hotspots[name]),
                        "weighted_detail_fidelity": weighted_detail_fidelity(image, reconstructed, detail_map),
                    }
                )

    main_df = pd.DataFrame(records)
    budget_df = pd.DataFrame(budget_records)
    resolution_df = pd.DataFrame(resolution_records)

    main_df.to_csv(OUTPUT_DIR / "main_metrics.csv", index=False)
    budget_df.to_csv(OUTPUT_DIR / "budget_sensitivity.csv", index=False)
    resolution_df.to_csv(OUTPUT_DIR / "resolution_sensitivity.csv", index=False)

    metadata = {
        name: {
            "guided_boxes": [box.as_list() for box in guided_selection[name]],
            "hotspots": [box.as_list() for box in hotspots[name]],
        }
        for name in images
    }
    (OUTPUT_DIR / "selection_metadata.json").write_text(json.dumps(metadata, indent=2))
    return main_df, budget_df, resolution_df, guided_selection, images


def main() -> None:
    main_df, budget_df, resolution_df, guided_selection, images = analyze_images()

    method_summary = (
        main_df.groupby("method", as_index=False)[["hotspot_recovery", "weighted_detail_fidelity", "mean_crop_fraction"]]
        .mean()
        .sort_values("hotspot_recovery", ascending=False)
    )
    budget_summary = (
        budget_df.groupby(["method", "num_crops"], as_index=False)[["hotspot_recovery", "weighted_detail_fidelity"]].mean()
    )
    resolution_summary = (
        resolution_df.groupby(["method", "encoder_resolution"], as_index=False)[["hotspot_recovery", "weighted_detail_fidelity"]]
        .mean()
    )

    method_summary.to_csv(OUTPUT_DIR / "method_summary.csv", index=False)
    budget_summary.to_csv(OUTPUT_DIR / "budget_summary.csv", index=False)
    resolution_summary.to_csv(OUTPUT_DIR / "resolution_summary.csv", index=False)

    save_input_overview(images)
    save_overlay_figure(images, guided_selection)
    save_qualitative_reconstruction(images, guided_selection)
    plot_method_comparison(method_summary)
    plot_budget_sensitivity(budget_summary)
    plot_resolution_sensitivity(resolution_summary)

    print("Saved:")
    for path in [
        OUTPUT_DIR / "main_metrics.csv",
        OUTPUT_DIR / "budget_summary.csv",
        OUTPUT_DIR / "resolution_summary.csv",
        REPORT_IMG_DIR / "data_overview.png",
        REPORT_IMG_DIR / "guided_crop_overlays.png",
        REPORT_IMG_DIR / "qualitative_reconstructions.png",
        REPORT_IMG_DIR / "method_comparison.png",
        REPORT_IMG_DIR / "budget_sensitivity.png",
        REPORT_IMG_DIR / "resolution_sensitivity.png",
    ]:
        print(path.relative_to(WORKSPACE))


if __name__ == "__main__":
    main()
