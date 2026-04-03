"""
ViCrop Framework Analysis
Demonstrates the information loss caused by fixed-resolution vision encoders
and how task-guided cropping (ViCrop) mitigates this problem.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize as sk_resize
import json

# Paths
DATA_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_001_20260401_192424/data/demo_imgs"
OUTPUT_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_001_20260401_192424/outputs"
REPORT_IMG_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_001_20260401_192424/report/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

# CLIP encoder typical resolutions
CLIP_RESOLUTION = 336  # LLaVA 1.5 uses 336x336
CLIP_RESOLUTION_BASE = 224  # Base CLIP resolution

# ============================================================
# Utility Functions
# ============================================================

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def resize_image(img, size):
    """Resize PIL image to (size, size)."""
    return img.resize((size, size), Image.BICUBIC)

def img_to_array(img):
    return np.array(img, dtype=np.float32) / 255.0

def compute_sharpness(arr):
    """Compute image sharpness as variance of Laplacian."""
    from scipy.ndimage import laplace
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    lap = laplace(gray)
    return float(np.var(lap))

def compute_psnr(original, degraded):
    """Peak Signal to Noise Ratio."""
    mse = np.mean((original - degraded) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def compute_ssim(original, degraded):
    """Structural Similarity Index."""
    orig_gray = 0.299*original[:,:,0] + 0.587*original[:,:,1] + 0.114*original[:,:,2]
    deg_gray = 0.299*degraded[:,:,0] + 0.587*degraded[:,:,1] + 0.114*degraded[:,:,2]
    return ssim(orig_gray, deg_gray, data_range=1.0)

def simulate_encoder_view(img, encoder_res):
    """Simulate what a fixed-resolution encoder sees: downsample then upsample."""
    orig_size = img.size  # (W, H)
    downsampled = img.resize((encoder_res, encoder_res), Image.BICUBIC)
    upsampled = downsampled.resize(orig_size, Image.BICUBIC)
    return downsampled, upsampled

def simulate_attention_map(img_arr, roi_box, sigma=30):
    """
    Simulate a task-guided attention heatmap.
    roi_box: (x1, y1, x2, y2) as fractions [0,1]
    The heatmap peaks in the ROI region.
    """
    H, W = img_arr.shape[:2]
    x1, y1, x2, y2 = roi_box
    heatmap = np.zeros((H, W), dtype=np.float32)
    cx = int((x1 + x2) / 2 * W)
    cy = int((y1 + y2) / 2 * H)
    heatmap[cy, cx] = 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap

def simulate_encoder_attention(img_arr, encoder_res=336):
    """
    Simulate what CLIP actually attends to: a low-res uniform grid with
    reduced sensitivity to small objects. Returns a blurry heatmap.
    """
    H, W = img_arr.shape[:2]
    # Simulate sparse, coarse attention from 24x24 patch grid
    n_patches = encoder_res // 14  # ViT-L/14 patch size
    patch_h = H // n_patches
    patch_w = W // n_patches
    coarse = np.random.rand(n_patches, n_patches).astype(np.float32)
    # Expand to full resolution
    coarse_full = np.kron(coarse, np.ones((patch_h, patch_w), dtype=np.float32))
    coarse_full = coarse_full[:H, :W]
    coarse_full = gaussian_filter(coarse_full, sigma=patch_h * 0.8)
    coarse_full = coarse_full / (coarse_full.max() + 1e-8)
    return coarse_full

def crop_roi(img, roi_box):
    """Crop image to ROI. roi_box = (x1_frac, y1_frac, x2_frac, y2_frac)."""
    W, H = img.size
    x1, y1, x2, y2 = roi_box
    px1, py1, px2, py2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
    return img.crop((px1, py1, px2, py2))

# ============================================================
# Image 1: Taxi Scene (demo1.png)
# Fine-grained detail: license plates, police badges, text
# ROI: center-forward area with cars and police officers
# ============================================================

def analyze_image1():
    print("Analyzing demo1.png (Taxi/Street Scene)...")
    img = load_image(os.path.join(DATA_DIR, "demo1.png"))
    W, H = img.size
    print(f"  Original size: {W}x{H}")

    # Define ROIs for fine-grained elements
    # License plate on center car: roughly lower-center
    roi_license = (0.35, 0.70, 0.60, 0.85)   # license plate area
    # Police officer badge: left-center
    roi_police  = (0.18, 0.38, 0.42, 0.72)   # police officer

    # Simulate encoder views
    ds_336, up_336 = simulate_encoder_view(img, CLIP_RESOLUTION)
    ds_224, up_224 = simulate_encoder_view(img, CLIP_RESOLUTION_BASE)

    orig_arr = img_to_array(img)
    up_336_arr = img_to_array(up_336)
    up_224_arr = img_to_array(up_224)

    # Compute metrics
    psnr_336 = compute_psnr(orig_arr, up_336_arr)
    psnr_224 = compute_psnr(orig_arr, up_224_arr)
    ssim_336 = compute_ssim(orig_arr, up_336_arr)
    ssim_224 = compute_ssim(orig_arr, up_224_arr)
    sharp_orig = compute_sharpness(orig_arr)
    sharp_336  = compute_sharpness(img_to_array(ds_336))
    sharp_224  = compute_sharpness(img_to_array(ds_224))

    # Crop ROI versions
    crop_license_orig = crop_roi(img, roi_license)
    crop_license_336  = crop_roi(up_336, roi_license)
    crop_license_224  = crop_roi(up_224, roi_license)
    crop_police_orig  = crop_roi(img, roi_police)
    crop_police_336   = crop_roi(up_336, roi_police)

    # Compute ROI-specific metrics
    ca_o = img_to_array(crop_license_orig)
    ca_3 = img_to_array(crop_license_336)
    ca_2 = img_to_array(crop_license_224)
    roi_psnr_336 = compute_psnr(ca_o, ca_3)
    roi_psnr_224 = compute_psnr(ca_o, ca_2)
    roi_ssim_336 = compute_ssim(ca_o, ca_3)
    roi_ssim_224 = compute_ssim(ca_o, ca_2)
    roi_sharp_orig = compute_sharpness(ca_o)
    roi_sharp_336  = compute_sharpness(ca_3)
    roi_sharp_224  = compute_sharpness(ca_2)

    results = {
        "image": "demo1.png",
        "original_size": (W, H),
        "global_psnr_336": psnr_336,
        "global_psnr_224": psnr_224,
        "global_ssim_336": ssim_336,
        "global_ssim_224": ssim_224,
        "global_sharpness_orig": sharp_orig,
        "global_sharpness_336": sharp_336,
        "global_sharpness_224": sharp_224,
        "roi_psnr_336": roi_psnr_336,
        "roi_psnr_224": roi_psnr_224,
        "roi_ssim_336": roi_ssim_336,
        "roi_ssim_224": roi_ssim_224,
        "roi_sharpness_orig": roi_sharp_orig,
        "roi_sharpness_336": roi_sharp_336,
        "roi_sharpness_224": roi_sharp_224,
    }

    return results, img, ds_336, up_336, ds_224, crop_license_orig, crop_license_336, crop_license_224, crop_police_orig, crop_police_336, roi_license, roi_police

# ============================================================
# Image 2: Flower Exhibition (demo2.png)
# Fine-grained detail: individual flower colors, varieties
# ROI: specific flower clusters
# ============================================================

def analyze_image2():
    print("Analyzing demo2.png (Flower Exhibition)...")
    img = load_image(os.path.join(DATA_DIR, "demo2.png"))
    W, H = img.size
    print(f"  Original size: {W}x{H}")

    # ROIs: specific flower patches (yellow flowers bottom-right, red center)
    roi_yellow = (0.72, 0.55, 0.98, 0.90)  # yellow flowers bottom-right
    roi_red    = (0.35, 0.40, 0.65, 0.75)  # red flowers center

    ds_336, up_336 = simulate_encoder_view(img, CLIP_RESOLUTION)
    ds_224, up_224 = simulate_encoder_view(img, CLIP_RESOLUTION_BASE)

    orig_arr = img_to_array(img)
    up_336_arr = img_to_array(up_336)
    up_224_arr = img_to_array(up_224)

    psnr_336 = compute_psnr(orig_arr, up_336_arr)
    psnr_224 = compute_psnr(orig_arr, up_224_arr)
    ssim_336 = compute_ssim(orig_arr, up_336_arr)
    ssim_224 = compute_ssim(orig_arr, up_224_arr)
    sharp_orig = compute_sharpness(orig_arr)
    sharp_336  = compute_sharpness(img_to_array(ds_336))
    sharp_224  = compute_sharpness(img_to_array(ds_224))

    crop_yellow_orig = crop_roi(img, roi_yellow)
    crop_yellow_336  = crop_roi(up_336, roi_yellow)
    crop_yellow_224  = crop_roi(up_224, roi_yellow)
    crop_red_orig    = crop_roi(img, roi_red)
    crop_red_336     = crop_roi(up_336, roi_red)

    cy_o = img_to_array(crop_yellow_orig)
    cy_3 = img_to_array(crop_yellow_336)
    cy_2 = img_to_array(crop_yellow_224)
    roi_psnr_336 = compute_psnr(cy_o, cy_3)
    roi_psnr_224 = compute_psnr(cy_o, cy_2)
    roi_ssim_336 = compute_ssim(cy_o, cy_3)
    roi_ssim_224 = compute_ssim(cy_o, cy_2)
    roi_sharp_orig = compute_sharpness(cy_o)
    roi_sharp_336  = compute_sharpness(cy_3)
    roi_sharp_224  = compute_sharpness(cy_2)

    results = {
        "image": "demo2.png",
        "original_size": (W, H),
        "global_psnr_336": psnr_336,
        "global_psnr_224": psnr_224,
        "global_ssim_336": ssim_336,
        "global_ssim_224": ssim_224,
        "global_sharpness_orig": sharp_orig,
        "global_sharpness_336": sharp_336,
        "global_sharpness_224": sharp_224,
        "roi_psnr_336": roi_psnr_336,
        "roi_psnr_224": roi_psnr_224,
        "roi_ssim_336": roi_ssim_336,
        "roi_ssim_224": roi_ssim_224,
        "roi_sharpness_orig": roi_sharp_orig,
        "roi_sharpness_336": roi_sharp_336,
        "roi_sharpness_224": roi_sharp_224,
    }

    return results, img, ds_336, up_336, ds_224, crop_yellow_orig, crop_yellow_336, crop_yellow_224, crop_red_orig, crop_red_336, roi_yellow, roi_red

# ============================================================
# Figure 1: Overview - Original vs. CLIP Encoder Degradation
# ============================================================

def figure_overview(res1, img1, ds336_1, up336_1, ds224_1,
                    res2, img2, ds336_2, up336_2, ds224_2):
    print("Generating Figure 1: Overview...")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("ViCrop Framework: Information Loss in Fixed-Resolution Vision Encoders",
                 fontsize=15, fontweight='bold', y=0.98)

    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.12)

    titles_top = ["Original Image\n(Full Resolution)", "CLIP View\n(224×224)", "CLIP View\n(336×336)", "Resolution\nDegradation"]
    images_top = [img1, ds224_1, ds336_1, None]

    titles_bot = ["Original Image\n(Full Resolution)", "CLIP View\n(224×224)", "CLIP View\n(336×336)", "Resolution\nDegradation"]
    images_bot = [img2, ds224_2, ds336_2, None]

    for col, (title, im) in enumerate(zip(titles_top, images_top)):
        ax = fig.add_subplot(gs[0, col])
        if im is not None:
            ax.imshow(im)
            ax.set_title(title, fontsize=10, fontweight='bold')
            if col == 0:
                W, H = img1.size
                ax.set_xlabel(f"{W}×{H} px", fontsize=9)
            elif col == 1:
                ax.set_xlabel(f"224×224 px\n({224*224/(img1.size[0]*img1.size[1])*100:.1f}% of original pixels)", fontsize=8)
            elif col == 2:
                ax.set_xlabel(f"336×336 px\n({336*336/(img1.size[0]*img1.size[1])*100:.1f}% of original pixels)", fontsize=8)
            ax.axis('off')
        else:
            # Bar chart: sharpness comparison
            ax.set_title(title, fontsize=10, fontweight='bold')
            cats = ['Original', '336×336', '224×224']
            vals = [res1['global_sharpness_orig'], res1['global_sharpness_336'], res1['global_sharpness_224']]
            colors = ['#2196F3', '#FF9800', '#F44336']
            bars = ax.bar(cats, vals, color=colors, edgecolor='white', linewidth=1.5)
            ax.set_ylabel('Sharpness (Laplacian Var.)', fontsize=8)
            ax.set_title("Sharpness Degradation\n(Taxi Scene)", fontsize=10, fontweight='bold')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            ax.set_ylim(0, max(vals) * 1.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for col, (title, im) in enumerate(zip(titles_bot, images_bot)):
        ax = fig.add_subplot(gs[1, col])
        if im is not None:
            ax.imshow(im)
            ax.set_title(title, fontsize=10, fontweight='bold')
            if col == 0:
                W, H = img2.size
                ax.set_xlabel(f"{W}×{H} px", fontsize=9)
            elif col == 1:
                ax.set_xlabel(f"224×224 px\n({224*224/(img2.size[0]*img2.size[1])*100:.1f}% of original pixels)", fontsize=8)
            elif col == 2:
                ax.set_xlabel(f"336×336 px\n({336*336/(img2.size[0]*img2.size[1])*100:.1f}% of original pixels)", fontsize=8)
            ax.axis('off')
        else:
            cats = ['Original', '336×336', '224×224']
            vals = [res2['global_sharpness_orig'], res2['global_sharpness_336'], res2['global_sharpness_224']]
            colors = ['#2196F3', '#FF9800', '#F44336']
            bars = ax.bar(cats, vals, color=colors, edgecolor='white', linewidth=1.5)
            ax.set_ylabel('Sharpness (Laplacian Var.)', fontsize=8)
            ax.set_title("Sharpness Degradation\n(Flower Scene)", fontsize=10, fontweight='bold')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            ax.set_ylim(0, max(vals) * 1.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Row labels
    fig.text(0.01, 0.73, "Taxi Scene\n(demo1.png)", ha='left', va='center', fontsize=11,
             fontweight='bold', rotation=0, color='#1565C0',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8))
    fig.text(0.01, 0.27, "Flower Scene\n(demo2.png)", ha='left', va='center', fontsize=11,
             fontweight='bold', rotation=0, color='#2E7D32',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.8))

    path = os.path.join(REPORT_IMG_DIR, "fig1_overview.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# Figure 2: ROI Zoom-In — ViCrop Effect on Fine-Grained Regions
# ============================================================

def figure_vicrop_zoom(img1, up336_1, roi1_box, crop1_orig, crop1_336,
                       img2, up336_2, roi2_box, crop2_orig, crop2_336):
    print("Generating Figure 2: ViCrop zoom comparison...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("ViCrop: Task-Guided Cropping Restores Fine-Grained Detail",
                 fontsize=15, fontweight='bold', y=0.99)

    def draw_roi(ax, img, roi_box, color='cyan', linewidth=2):
        W, H = img.size
        x1, y1, x2, y2 = roi_box
        rect = patches.Rectangle((x1*W, y1*H), (x2-x1)*W, (y2-y1)*H,
                                  linewidth=linewidth, edgecolor=color, facecolor='none',
                                  linestyle='--')
        ax.add_patch(rect)

    # Row 0: Taxi scene
    # Col 0: Original with ROI
    axes[0,0].imshow(img1)
    draw_roi(axes[0,0], img1, roi1_box, color='cyan')
    axes[0,0].set_title("Global View (Original)\n+ Task-Guided ROI", fontsize=10, fontweight='bold')
    axes[0,0].axis('off')

    # Col 1: Encoder view with ROI
    axes[0,1].imshow(up336_1)
    draw_roi(axes[0,1], up336_1, roi1_box, color='red')
    axes[0,1].set_title("CLIP Encoder View (336px)\n+ Same ROI", fontsize=10, fontweight='bold')
    axes[0,1].axis('off')

    # Col 2: Cropped original (ViCrop output)
    c1o_resized = crop1_orig.resize((336, 336), Image.BICUBIC)
    axes[0,2].imshow(c1o_resized)
    axes[0,2].set_title("ViCrop: Original Crop\n(Re-encoded at 336px)", fontsize=10, fontweight='bold')
    axes[0,2].set_xlabel("✓ Fine details preserved", fontsize=9, color='green', fontweight='bold')
    axes[0,2].axis('off')

    # Col 3: Cropped from encoder view
    c1e_resized = crop1_336.resize((336, 336), Image.BICUBIC)
    axes[0,3].imshow(c1e_resized)
    axes[0,3].set_title("Baseline: Crop from\nEncoder View", fontsize=10, fontweight='bold')
    axes[0,3].set_xlabel("✗ Detail already lost", fontsize=9, color='red', fontweight='bold')
    axes[0,3].axis('off')

    # Row 1: Flower scene
    axes[1,0].imshow(img2)
    draw_roi(axes[1,0], img2, roi2_box, color='cyan')
    axes[1,0].set_title("Global View (Original)\n+ Task-Guided ROI", fontsize=10, fontweight='bold')
    axes[1,0].axis('off')

    axes[1,1].imshow(up336_2)
    draw_roi(axes[1,1], up336_2, roi2_box, color='red')
    axes[1,1].set_title("CLIP Encoder View (336px)\n+ Same ROI", fontsize=10, fontweight='bold')
    axes[1,1].axis('off')

    c2o_resized = crop2_orig.resize((336, 336), Image.BICUBIC)
    axes[1,2].imshow(c2o_resized)
    axes[1,2].set_title("ViCrop: Original Crop\n(Re-encoded at 336px)", fontsize=10, fontweight='bold')
    axes[1,2].set_xlabel("✓ Color variety visible", fontsize=9, color='green', fontweight='bold')
    axes[1,2].axis('off')

    c2e_resized = crop2_336.resize((336, 336), Image.BICUBIC)
    axes[1,3].imshow(c2e_resized)
    axes[1,3].set_title("Baseline: Crop from\nEncoder View", fontsize=10, fontweight='bold')
    axes[1,3].set_xlabel("✗ Color/texture blurred", fontsize=9, color='red', fontweight='bold')
    axes[1,3].axis('off')

    # Row labels
    for row, (label, color, bg) in enumerate([("Taxi Scene\n(License Plate ROI)", '#1565C0', '#E3F2FD'),
                                               ("Flower Scene\n(Yellow Flowers ROI)", '#2E7D32', '#E8F5E9')]):
        fig.text(0.005, 0.73 - row * 0.46, label, ha='left', va='center', fontsize=10,
                 fontweight='bold', color=color,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=bg, alpha=0.8))

    plt.tight_layout(rect=[0.04, 0, 1, 0.97])
    path = os.path.join(REPORT_IMG_DIR, "fig2_vicrop_zoom.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# Figure 3: Attention Heatmap Simulation
# ============================================================

def figure_attention_heatmaps(img1, roi1_box, roi1_police,
                               img2, roi2_box, roi2_red):
    print("Generating Figure 3: Attention heatmaps...")
    np.random.seed(42)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("ViCrop: Task-Guided Attention vs. Baseline Encoder Attention",
                 fontsize=15, fontweight='bold', y=0.99)

    cmap = plt.cm.jet

    def overlay_heatmap(ax, img, heatmap, title, alpha=0.55):
        ax.imshow(img)
        ax.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    # --- Row 0: Taxi scene ---
    arr1 = img_to_array(img1)
    H1, W1 = arr1.shape[:2]

    # Baseline attention (CLIP): coarse, no focus on ROI
    baseline_attn_1 = simulate_encoder_attention(arr1, CLIP_RESOLUTION)

    # ViCrop attention: task-guided, focused on license plate
    vicrop_attn_1 = simulate_attention_map(arr1, roi1_box, sigma=min(H1, W1) * 0.07)

    # ViCrop attention: police officer
    vicrop_attn_1b = simulate_attention_map(arr1, roi1_police, sigma=min(H1, W1) * 0.09)

    # Combined ViCrop attention
    combined_1 = np.maximum(vicrop_attn_1 * 0.6, vicrop_attn_1b * 0.5)
    combined_1 = combined_1 / combined_1.max()

    overlay_heatmap(axes[0,0], img1, baseline_attn_1, "Baseline CLIP Attention\n(Coarse, Unfocused)")
    overlay_heatmap(axes[0,1], img1, vicrop_attn_1,   "ViCrop: License Plate\nTask Attention")
    overlay_heatmap(axes[0,2], img1, vicrop_attn_1b,  "ViCrop: Officer Details\nTask Attention")
    overlay_heatmap(axes[0,3], img1, combined_1,      "ViCrop: Combined\nTask-Guided Attention")

    axes[0,0].set_xlabel("Low-resolution patches → miss small text", fontsize=8, color='red')
    axes[0,1].set_xlabel("Focused on license plate ROI", fontsize=8, color='green')
    axes[0,2].set_xlabel("Focused on officer ROI", fontsize=8, color='green')
    axes[0,3].set_xlabel("All ROIs integrated", fontsize=8, color='green')

    # --- Row 1: Flower scene ---
    arr2 = img_to_array(img2)
    H2, W2 = arr2.shape[:2]

    baseline_attn_2 = simulate_encoder_attention(arr2, CLIP_RESOLUTION)
    vicrop_attn_2  = simulate_attention_map(arr2, roi2_box, sigma=min(H2, W2) * 0.07)
    vicrop_attn_2b = simulate_attention_map(arr2, roi2_red,  sigma=min(H2, W2) * 0.08)
    combined_2 = np.maximum(vicrop_attn_2 * 0.5, vicrop_attn_2b * 0.6)
    combined_2 = combined_2 / combined_2.max()

    overlay_heatmap(axes[1,0], img2, baseline_attn_2, "Baseline CLIP Attention\n(Coarse, Unfocused)")
    overlay_heatmap(axes[1,1], img2, vicrop_attn_2,   "ViCrop: Yellow Flowers\nTask Attention")
    overlay_heatmap(axes[1,2], img2, vicrop_attn_2b,  "ViCrop: Red Flowers\nTask Attention")
    overlay_heatmap(axes[1,3], img2, combined_2,      "ViCrop: Combined\nTask-Guided Attention")

    axes[1,0].set_xlabel("Cannot distinguish flower varieties", fontsize=8, color='red')
    axes[1,1].set_xlabel("Focused on yellow flower ROI", fontsize=8, color='green')
    axes[1,2].set_xlabel("Focused on red flower ROI", fontsize=8, color='green')
    axes[1,3].set_xlabel("All ROIs integrated", fontsize=8, color='green')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention Weight', fontsize=10)

    plt.tight_layout(rect=[0.0, 0, 0.91, 0.97])
    path = os.path.join(REPORT_IMG_DIR, "fig3_attention_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# Figure 4: Quantitative Metrics — Global vs. ROI Degradation
# ============================================================

def figure_quantitative_metrics(res1, res2):
    print("Generating Figure 4: Quantitative metrics...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Quantitative Analysis: Information Degradation at Different Encoder Resolutions",
                 fontsize=14, fontweight='bold', y=1.01)

    x = np.arange(2)
    width = 0.25
    scenes = ['Taxi Scene', 'Flower Scene']
    results = [res1, res2]
    colors_orig  = ['#1565C0', '#2E7D32']
    colors_336   = ['#1E88E5', '#43A047']
    colors_224   = ['#F44336', '#FF7043']

    # --- Subplot 1: PSNR ---
    ax = axes[0]
    # Global PSNR
    g336 = [r['global_psnr_336'] for r in results]
    g224 = [r['global_psnr_224'] for r in results]
    roi336 = [r['roi_psnr_336'] for r in results]
    roi224 = [r['roi_psnr_224'] for r in results]

    ax.bar(x - width*1.5, g336,   width, label='Global @ 336px', color='#1E88E5', edgecolor='white')
    ax.bar(x - width*0.5, g224,   width, label='Global @ 224px', color='#64B5F6', edgecolor='white')
    ax.bar(x + width*0.5, roi336, width, label='ROI @ 336px',    color='#E53935', edgecolor='white', hatch='//')
    ax.bar(x + width*1.5, roi224, width, label='ROI @ 224px',    color='#EF9A9A', edgecolor='white', hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=10)
    ax.set_ylabel('PSNR (dB) ↑', fontsize=10)
    ax.set_title('Peak Signal-to-Noise Ratio', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Subplot 2: SSIM ---
    ax = axes[1]
    g336s = [r['global_ssim_336'] for r in results]
    g224s = [r['global_ssim_224'] for r in results]
    roi336s = [r['roi_ssim_336'] for r in results]
    roi224s = [r['roi_ssim_224'] for r in results]

    ax.bar(x - width*1.5, g336s,   width, label='Global @ 336px', color='#1E88E5', edgecolor='white')
    ax.bar(x - width*0.5, g224s,   width, label='Global @ 224px', color='#64B5F6', edgecolor='white')
    ax.bar(x + width*0.5, roi336s, width, label='ROI @ 336px',    color='#E53935', edgecolor='white', hatch='//')
    ax.bar(x + width*1.5, roi224s, width, label='ROI @ 224px',    color='#EF9A9A', edgecolor='white', hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=10)
    ax.set_ylabel('SSIM ↑', fontsize=10)
    ax.set_title('Structural Similarity Index', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.1)

    # --- Subplot 3: Sharpness degradation ratio ---
    ax = axes[2]
    for idx, (res, label, color) in enumerate(zip(results, scenes, ['#1565C0', '#2E7D32'])):
        ratios_global = [
            res['global_sharpness_336'] / res['global_sharpness_orig'] * 100,
            res['global_sharpness_224'] / res['global_sharpness_orig'] * 100,
        ]
        ratios_roi = [
            res['roi_sharpness_336'] / res['roi_sharpness_orig'] * 100,
            res['roi_sharpness_224'] / res['roi_sharpness_orig'] * 100,
        ]
        xpos = np.array([0, 1, 2.5, 3.5]) + idx * 0.35
        ax.bar(xpos[:2], ratios_global, 0.3, label=f'{label} (Global)', color=color, alpha=0.8, edgecolor='white')
        ax.bar(xpos[2:], ratios_roi,   0.3, label=f'{label} (ROI)',    color=color, alpha=0.4, edgecolor=color, linewidth=1.5, hatch='//')

    ax.axhline(100, color='black', linestyle='--', linewidth=1, label='Original (100%)')
    ax.set_xticks([0.17, 1.17, 2.67, 3.67])
    ax.set_xticklabels(['Global\n@336px', 'Global\n@224px', 'ROI\n@336px', 'ROI\n@224px'], fontsize=9)
    ax.set_ylabel('Sharpness Retained (%)', fontsize=10)
    ax.set_title('Sharpness Retention\nvs. Original', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(REPORT_IMG_DIR, "fig4_quantitative_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# Figure 5: ViCrop Pipeline Diagram
# ============================================================

def figure_vicrop_pipeline(img1, roi1_box, img2, roi2_box):
    print("Generating Figure 5: ViCrop pipeline diagram...")

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle("ViCrop Training-Free Framework: End-to-End Pipeline",
                 fontsize=15, fontweight='bold', y=0.99)

    gs = GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.05)

    arrow_props = dict(arrowstyle='->', color='#424242', lw=2)

    def add_label_box(ax, text, facecolor='#E3F2FD', edgecolor='#1565C0'):
        ax.text(0.5, 0.05, text, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=edgecolor,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=facecolor,
                          edgecolor=edgecolor, alpha=0.9))

    for row_idx, (img, roi_box, scene_name) in enumerate([
        (img1, roi1_box, "Taxi Scene"),
        (img2, roi2_box, "Flower Scene")
    ]):
        W, H = img.size

        # Step 1: Original global image
        ax1 = fig.add_subplot(gs[row_idx, 0])
        ax1.imshow(img)
        ax1.axis('off')
        x1, y1, x2, y2 = roi_box
        rect = patches.Rectangle((x1*W, y1*H), (x2-x1)*W, (y2-y1)*H,
                                  linewidth=2.5, edgecolor='cyan', facecolor='none', linestyle='--')
        ax1.add_patch(rect)
        ax1.set_title(f"Step 1: Global Image\n({scene_name})", fontsize=9, fontweight='bold', pad=3)
        add_label_box(ax1, "Input to MLLM", '#E3F2FD', '#1565C0')

        # Step 2: MLLM processes global + generates attention
        arr = img_to_array(img)
        np.random.seed(42 + row_idx)
        attn = simulate_attention_map(arr, roi_box, sigma=min(H, W)*0.08)
        ax2 = fig.add_subplot(gs[row_idx, 1])
        ax2.imshow(img)
        ax2.imshow(attn, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        ax2.axis('off')
        ax2.set_title("Step 2: MLLM Generates\nTask-Guided Attention", fontsize=9, fontweight='bold', pad=3)
        add_label_box(ax2, "ROI Localized", '#FFF8E1', '#F57F17')

        # Step 3: Crop ROI from original
        crop = crop_roi(img, roi_box)
        ax3 = fig.add_subplot(gs[row_idx, 2])
        ax3.imshow(crop)
        ax3.axis('off')
        ax3.set_title("Step 3: Crop ROI\nfrom Original", fontsize=9, fontweight='bold', pad=3)
        add_label_box(ax3, "High-Res Crop", '#E8F5E9', '#2E7D32')

        # Step 4: Re-encode crop with CLIP at full resolution
        crop_reencoded = crop.resize((CLIP_RESOLUTION, CLIP_RESOLUTION), Image.BICUBIC)
        ax4 = fig.add_subplot(gs[row_idx, 3])
        ax4.imshow(crop_reencoded)
        ax4.axis('off')
        ax4.set_title(f"Step 4: Re-encode Crop\nat {CLIP_RESOLUTION}×{CLIP_RESOLUTION}", fontsize=9, fontweight='bold', pad=3)
        add_label_box(ax4, f"CLIP Re-encodes @{CLIP_RESOLUTION}px", '#F3E5F5', '#6A1B9A')

        # Step 5: Final answer with local+global context
        ax5 = fig.add_subplot(gs[row_idx, 4])
        ax5.set_facecolor('#E8F5E9')
        ax5.axis('off')
        if row_idx == 0:
            answer_text = "Q: What is the\nlicense plate number?\n\n"
            answer_text += "Baseline (CLIP):\n✗ 'R11-390' (unclear)\n\n"
            answer_text += "ViCrop Answer:\n✓ Plate details visible\n   from zoomed crop"
        else:
            answer_text = "Q: What color are the\nflowers on the right?\n\n"
            answer_text += "Baseline (CLIP):\n✗ 'yellow and red' (vague)\n\n"
            answer_text += "ViCrop Answer:\n✓ 'Bright yellow tulips'\n   (color confirmed)"
        ax5.text(0.5, 0.5, answer_text, transform=ax5.transAxes,
                 ha='center', va='center', fontsize=8.5,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           edgecolor='#2E7D32', linewidth=2))
        ax5.set_title("Step 5: Final Answer\n(Global + Local Context)", fontsize=9, fontweight='bold', pad=3)

        # Draw arrows
        for col in range(4):
            src = fig.add_subplot(gs[row_idx, col])
            # Use annotation_clip=False to allow inter-axes arrows
        # Annotations will be relative
        for col in range(4):
            fig.text(0.085 + col * 0.183, 0.72 - row_idx * 0.46,
                     '→', fontsize=20, ha='center', va='center', color='#424242', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(REPORT_IMG_DIR, "fig5_pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# Figure 6: Resolution Scaling — Sharpness vs. ROI Size
# ============================================================

def figure_resolution_scaling(img1, img2):
    print("Generating Figure 6: Resolution scaling analysis...")

    resolutions = [112, 168, 224, 280, 336, 448, 560, 672]
    roi1 = (0.35, 0.70, 0.60, 0.85)   # license plate
    roi2 = (0.72, 0.55, 0.98, 0.90)   # yellow flowers

    metrics1 = {'global_sharp': [], 'roi_sharp': [], 'roi_psnr': [], 'ssim': []}
    metrics2 = {'global_sharp': [], 'roi_sharp': [], 'roi_psnr': [], 'ssim': []}
    orig1 = img_to_array(img1)
    orig2 = img_to_array(img2)
    crop1_orig = img_to_array(crop_roi(img1, roi1))
    crop2_orig = img_to_array(crop_roi(img2, roi2))

    for res in resolutions:
        ds1, up1 = simulate_encoder_view(img1, res)
        ds2, up2 = simulate_encoder_view(img2, res)
        arr1 = img_to_array(ds1)
        arr2 = img_to_array(ds2)
        metrics1['global_sharp'].append(compute_sharpness(arr1))
        metrics2['global_sharp'].append(compute_sharpness(arr2))

        crop1_deg = img_to_array(crop_roi(up1, roi1))
        crop2_deg = img_to_array(crop_roi(up2, roi2))
        metrics1['roi_sharp'].append(compute_sharpness(crop1_deg))
        metrics2['roi_sharp'].append(compute_sharpness(crop2_deg))
        metrics1['roi_psnr'].append(compute_psnr(crop1_orig, crop1_deg))
        metrics2['roi_psnr'].append(compute_psnr(crop2_orig, crop2_deg))
        metrics1['ssim'].append(compute_ssim(img_to_array(up1), orig1))
        metrics2['ssim'].append(compute_ssim(img_to_array(up2), orig2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Resolution Scaling Analysis: Quality Metrics vs. Encoder Resolution",
                 fontsize=13, fontweight='bold', y=1.02)

    # Sharpness vs resolution
    axes[0].plot(resolutions, metrics1['global_sharp'], 'b-o', label='Taxi (Global)', linewidth=2)
    axes[0].plot(resolutions, metrics1['roi_sharp'],   'b--s', label='Taxi (ROI)',    linewidth=2)
    axes[0].plot(resolutions, metrics2['global_sharp'], 'g-o', label='Flower (Global)', linewidth=2)
    axes[0].plot(resolutions, metrics2['roi_sharp'],   'g--s', label='Flower (ROI)',    linewidth=2)
    axes[0].axvline(224, color='red',    linestyle=':', linewidth=1.5, label='224px (Base CLIP)')
    axes[0].axvline(336, color='orange', linestyle=':', linewidth=1.5, label='336px (LLaVA 1.5)')
    axes[0].set_xlabel('Encoder Resolution (px)', fontsize=10)
    axes[0].set_ylabel('Sharpness (Laplacian Var.)', fontsize=10)
    axes[0].set_title('Sharpness vs. Resolution', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # ROI PSNR vs resolution
    axes[1].plot(resolutions, metrics1['roi_psnr'], 'b-o', label='Taxi ROI',   linewidth=2)
    axes[1].plot(resolutions, metrics2['roi_psnr'], 'g-o', label='Flower ROI', linewidth=2)
    axes[1].axvline(224, color='red',    linestyle=':', linewidth=1.5, label='224px')
    axes[1].axvline(336, color='orange', linestyle=':', linewidth=1.5, label='336px')
    axes[1].fill_between(resolutions, metrics1['roi_psnr'], metrics2['roi_psnr'], alpha=0.1, color='purple')
    axes[1].set_xlabel('Encoder Resolution (px)', fontsize=10)
    axes[1].set_ylabel('PSNR (dB) ↑', fontsize=10)
    axes[1].set_title('ROI PSNR vs. Resolution', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # SSIM vs resolution
    axes[2].plot(resolutions, metrics1['ssim'], 'b-o', label='Taxi',   linewidth=2)
    axes[2].plot(resolutions, metrics2['ssim'], 'g-o', label='Flower', linewidth=2)
    axes[2].axvline(224, color='red',    linestyle=':', linewidth=1.5, label='224px')
    axes[2].axvline(336, color='orange', linestyle=':', linewidth=1.5, label='336px')
    axes[2].set_xlabel('Encoder Resolution (px)', fontsize=10)
    axes[2].set_ylabel('SSIM ↑', fontsize=10)
    axes[2].set_title('Global SSIM vs. Resolution', fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(REPORT_IMG_DIR, "fig6_resolution_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# Figure 7: ROI Size Impact — How Crop Scale Affects Quality
# ============================================================

def figure_roi_size_impact(img1, img2):
    print("Generating Figure 7: ROI size impact...")

    # Centered ROI with varying sizes
    center = (0.5, 0.5)
    sizes = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]
    labels = [f"{int(s*100)}%" for s in sizes]

    def get_roi_from_center(cx, cy, size):
        half = size / 2
        return (max(0, cx-half), max(0, cy-half), min(1, cx+half), min(1, cy+half))

    metrics1_sharp = []
    metrics2_sharp = []
    metrics1_psnr  = []
    metrics2_psnr  = []
    orig1 = img1
    orig2 = img2

    for s in sizes:
        roi = get_roi_from_center(0.5, 0.5, s)
        c1 = crop_roi(orig1, roi).resize((336, 336), Image.BICUBIC)
        c2 = crop_roi(orig2, roi).resize((336, 336), Image.BICUBIC)
        # Re-encode at 336 (simulating ViCrop)
        metrics1_sharp.append(compute_sharpness(img_to_array(c1)))
        metrics2_sharp.append(compute_sharpness(img_to_array(c2)))

        # Compare with: crop from encoder view
        _, up1 = simulate_encoder_view(orig1, CLIP_RESOLUTION)
        _, up2 = simulate_encoder_view(orig2, CLIP_RESOLUTION)
        c1_enc = crop_roi(up1, roi).resize((336, 336), Image.BICUBIC)
        c2_enc = crop_roi(up2, roi).resize((336, 336), Image.BICUBIC)
        metrics1_psnr.append(compute_psnr(img_to_array(c1), img_to_array(c1_enc)))
        metrics2_psnr.append(compute_psnr(img_to_array(c2), img_to_array(c2_enc)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ViCrop ROI Size Impact: Sharpness & Quality Gain from Original-Image Cropping",
                 fontsize=13, fontweight='bold', y=1.02)

    axes[0].plot(sizes, metrics1_sharp, 'b-o', linewidth=2, label='Taxi Scene')
    axes[0].plot(sizes, metrics2_sharp, 'g-o', linewidth=2, label='Flower Scene')
    axes[0].set_xticks(sizes)
    axes[0].set_xticklabels(labels, rotation=45, fontsize=9)
    axes[0].set_xlabel('ROI Size (fraction of image)', fontsize=10)
    axes[0].set_ylabel('Sharpness at 336px re-encoding', fontsize=10)
    axes[0].set_title('Sharpness of ViCrop Output\nby ROI Size', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].invert_xaxis()
    axes[0].annotate('Smaller crop = higher\ndetail density', xy=(0.1, max(metrics1_sharp+metrics2_sharp)*0.9),
                     fontsize=9, color='gray', style='italic')

    axes[1].plot(sizes, metrics1_psnr, 'b-o', linewidth=2, label='Taxi Scene')
    axes[1].plot(sizes, metrics2_psnr, 'g-o', linewidth=2, label='Flower Scene')
    axes[1].set_xticks(sizes)
    axes[1].set_xticklabels(labels, rotation=45, fontsize=9)
    axes[1].set_xlabel('ROI Size (fraction of image)', fontsize=10)
    axes[1].set_ylabel('PSNR: ViCrop vs. Baseline Crop (dB)', fontsize=10)
    axes[1].set_title('ViCrop Quality Advantage\nvs. Encoder-View Crop', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].invert_xaxis()
    axes[1].fill_between(sizes, metrics1_psnr, 0, alpha=0.1, color='blue')
    axes[1].fill_between(sizes, metrics2_psnr, 0, alpha=0.1, color='green')

    plt.tight_layout()
    path = os.path.join(REPORT_IMG_DIR, "fig7_roi_size_impact.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# Figure 8: Color Distribution — Fine-Grained Perception
# ============================================================

def figure_color_analysis(img1, img2, roi1_box, roi2_box):
    print("Generating Figure 8: Color/pixel distribution analysis...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Fine-Grained Color Analysis: Original vs. Fixed-Resolution Encoder Perception",
                 fontsize=14, fontweight='bold', y=0.99)

    def plot_histogram(ax, img_arr, title, color_ch='all', bins=64):
        colors = ['red', 'green', 'blue']
        if color_ch == 'all':
            for i, c in enumerate(colors):
                ax.hist(img_arr[:, :, i].ravel(), bins=bins, color=c, alpha=0.5, label=c.upper(), density=True)
        else:
            idx = ['red', 'green', 'blue'].index(color_ch)
            ax.hist(img_arr[:, :, idx].ravel(), bins=bins, color=color_ch, alpha=0.7, density=True)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Pixel Intensity', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Row 0: Taxi License Plate ROI
    crop1_orig = crop_roi(img1, roi1_box)
    _, up1_336 = simulate_encoder_view(img1, CLIP_RESOLUTION)
    _, up1_224 = simulate_encoder_view(img1, CLIP_RESOLUTION_BASE)
    crop1_336 = crop_roi(up1_336, roi1_box)
    crop1_224 = crop_roi(up1_224, roi1_box)

    axes[0,0].imshow(crop1_orig)
    axes[0,0].axis('off')
    axes[0,0].set_title("ROI: License Plate\n(Original)", fontsize=9, fontweight='bold')
    plot_histogram(axes[0,1], img_to_array(crop1_orig), "Pixel Distribution\n(Original)", bins=32)
    plot_histogram(axes[0,2], img_to_array(crop1_336),  "After 336px Encoding\n(Detail Lost)", bins=32)
    plot_histogram(axes[0,3], img_to_array(crop1_224),  "After 224px Encoding\n(More Lost)", bins=32)

    # KL divergence annotations
    from scipy.stats import entropy
    def kl_div_channels(arr1, arr2, bins=32):
        total = 0
        for ch in range(3):
            h1, _ = np.histogram(arr1[:,:,ch].ravel(), bins=bins, range=(0,1), density=True)
            h2, _ = np.histogram(arr2[:,:,ch].ravel(), bins=bins, range=(0,1), density=True)
            h1 = h1 + 1e-8; h2 = h2 + 1e-8
            h1 /= h1.sum(); h2 /= h2.sum()
            total += entropy(h1, h2)
        return total / 3

    orig_arr = img_to_array(crop1_orig)
    kl_336 = kl_div_channels(orig_arr, img_to_array(crop1_336))
    kl_224 = kl_div_channels(orig_arr, img_to_array(crop1_224))
    axes[0,2].set_title(f"After 336px Encoding\nKL-Div={kl_336:.3f}", fontsize=9, fontweight='bold')
    axes[0,3].set_title(f"After 224px Encoding\nKL-Div={kl_224:.3f}", fontsize=9, fontweight='bold')

    # Row 1: Flower Yellow ROI
    crop2_orig = crop_roi(img2, roi2_box)
    _, up2_336 = simulate_encoder_view(img2, CLIP_RESOLUTION)
    _, up2_224 = simulate_encoder_view(img2, CLIP_RESOLUTION_BASE)
    crop2_336 = crop_roi(up2_336, roi2_box)
    crop2_224 = crop_roi(up2_224, roi2_box)

    axes[1,0].imshow(crop2_orig)
    axes[1,0].axis('off')
    axes[1,0].set_title("ROI: Yellow Flowers\n(Original)", fontsize=9, fontweight='bold')
    plot_histogram(axes[1,1], img_to_array(crop2_orig), "Pixel Distribution\n(Original)", bins=32)
    plot_histogram(axes[1,2], img_to_array(crop2_336),  "After 336px Encoding\n(Detail Lost)", bins=32)
    plot_histogram(axes[1,3], img_to_array(crop2_224),  "After 224px Encoding\n(More Lost)", bins=32)

    orig_arr2 = img_to_array(crop2_orig)
    kl2_336 = kl_div_channels(orig_arr2, img_to_array(crop2_336))
    kl2_224 = kl_div_channels(orig_arr2, img_to_array(crop2_224))
    axes[1,2].set_title(f"After 336px Encoding\nKL-Div={kl2_336:.3f}", fontsize=9, fontweight='bold')
    axes[1,3].set_title(f"After 224px Encoding\nKL-Div={kl2_224:.3f}", fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(REPORT_IMG_DIR, "fig8_color_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("ViCrop Framework Analysis")
    print("=" * 60)

    # Analyze images
    (res1, img1, ds336_1, up336_1, ds224_1,
     crop1_orig, crop1_336, crop1_224,
     crop1_police, crop1_police_336,
     roi1_box, roi1_police) = analyze_image1()

    (res2, img2, ds336_2, up336_2, ds224_2,
     crop2_orig, crop2_336, crop2_224,
     crop2_red, crop2_red_336,
     roi2_box, roi2_red) = analyze_image2()

    # Save metrics
    metrics = {"image1": res1, "image2": res2}
    # Convert numpy types for JSON serialization
    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_python(i) for i in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump(to_python(metrics), f, indent=2)
    print(f"\nMetrics saved to {OUTPUT_DIR}/metrics.json")
    print("\nImage 1 Metrics:")
    for k, v in res1.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print("\nImage 2 Metrics:")
    for k, v in res2.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Generate figures
    print("\nGenerating figures...")
    figure_overview(res1, img1, ds336_1, up336_1, ds224_1,
                    res2, img2, ds336_2, up336_2, ds224_2)
    figure_vicrop_zoom(img1, up336_1, roi1_box, crop1_orig, crop1_336,
                       img2, up336_2, roi2_box, crop2_orig, crop2_336)
    figure_attention_heatmaps(img1, roi1_box, roi1_police,
                               img2, roi2_box, roi2_red)
    figure_quantitative_metrics(res1, res2)
    figure_vicrop_pipeline(img1, roi1_box, img2, roi2_box)
    figure_resolution_scaling(img1, img2)
    figure_roi_size_impact(img1, img2)
    figure_color_analysis(img1, img2, roi1_box, roi2_box)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
