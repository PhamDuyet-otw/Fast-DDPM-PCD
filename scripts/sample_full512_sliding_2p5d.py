#!/usr/bin/env python3
"""
Full-512 Sliding-Window Inference for Fast-DDPM 2.5D CT Denoising.

Denoises full 512x512 CT slices using sliding-window patch sampling with
a model trained on 256x256 crops. Outputs comparison PNGs and metrics CSV.

Usage (Colab/server):
    python scripts/sample_full512_sliding_2p5d.py \
        --config configs/ldfd_v3_2p5d_5090_full.yml \
        --manifest /content/FastDDPM_Data/02_manifests/test_v3.csv \
        --ckpt /path/to/best_psnr.pth \
        --output_dir /content/full512_results \
        --max_samples 20
"""

import os
import sys
import csv
import math
import time
import random
import argparse
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import project modules
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.diffusion import Model
from models.ema import EMAHelper
from functions.denoising import sg_generalized_steps


# ============================================================================
# Helpers
# ============================================================================

def to_ns(x):
    """Recursively convert dicts to SimpleNamespace for attribute access."""
    if isinstance(x, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [to_ns(v) for v in x]
    return x


def set_seed(seed: int):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(model, ckpt_path, config, device):
    """
    Load checkpoint into model with flexible handling:
    - Supports list-format and dict-format checkpoints
    - Strips ``module.`` prefix from state_dict keys
    - Applies EMA weights if available and config.model.ema is True
    """
    print(f"[CKPT] Loading: {ckpt_path}")
    states = torch.load(ckpt_path, map_location="cpu")

    # --- Extract model state_dict ---
    if isinstance(states, (list, tuple)):
        state_dict = states[0]
    elif isinstance(states, dict):
        state_dict = states.get("model", states)
    else:
        raise ValueError(f"Unknown checkpoint format at {ckpt_path}")

    # Strip "module." prefix
    clean = {}
    for k, v in state_dict.items():
        clean[k.replace("module.", "", 1) if k.startswith("module.") else k] = v

    model.load_state_dict(clean, strict=True)

    # --- Print checkpoint metadata ---
    if isinstance(states, (list, tuple)):
        if len(states) >= 4:
            print(f"[CKPT] Epoch: {states[2]}, Step: {states[3]}")
        if len(states) >= 7:
            print(f"[CKPT] Best PSNR: {states[5]:.6f}, Best SSIM: {states[6]:.6f}")

    # --- Apply EMA weights ---
    ema_applied = False
    if getattr(config.model, "ema", False) and isinstance(states, (list, tuple)):
        ema_state = None
        if len(states) >= 5 and isinstance(states[4], dict):
            ema_state = states[4]

        if ema_state is not None:
            print("[CKPT] Applying EMA weights")
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(ema_state)
            ema_helper.ema(model)
            ema_applied = True

    if not ema_applied:
        print("[CKPT] No EMA applied (not available or disabled)")

    return model


# ---------------------------------------------------------------------------
# Manifest loading with multi-schema support
# ---------------------------------------------------------------------------

_COL_PATIENT_VARIANTS = ["patient_id", "Patient_ID", "PatientID", "PATIENT_ID"]
_COL_SLICE_VARIANTS = ["slice_idx", "slice_id", "Slice_ID", "SliceIdx", "SLICE_IDX"]
_COL_LD_VARIANTS = ["LD_Path", "low_path", "low_npy_path", "ld_path", "LD_path"]
_COL_FD_VARIANTS = ["FD_Path", "full_path", "full_npy_path", "fd_path", "FD_path"]


def _find_col(fieldnames, variants):
    """Find the first matching column name from a list of variants."""
    for v in variants:
        if v in fieldnames:
            return v
    # Case-insensitive fallback
    lower_map = {f.lower(): f for f in fieldnames}
    for v in variants:
        if v.lower() in lower_map:
            return lower_map[v.lower()]
    return None


def _resolve_path(path_str, manifest_dir, data_root=None):
    """
    Robustly resolve a file path from a manifest entry.

    Strategy:
    1. Absolute path if it exists
    2. Relative to manifest directory
    3. Relative to manifest parent directory
    4. Relative to data_root (from config) if provided
    5. Try joining data_root with the suffix after common markers
    """
    path_str = str(path_str).strip().strip('"').strip("'").replace("\\", "/")

    if not path_str:
        return None

    # 1. Absolute path exists
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return os.path.normpath(path_str)

    # Identify useful path suffixes
    basename = os.path.basename(path_str)
    suffixes = [path_str]
    for marker in ["01_processed_npy_512/", "02_processed_npy_256/",
                    "LDCT_DATASET_V2/", "FastDDPM_Data/"]:
        if marker in path_str:
            suffixes.append(path_str.split(marker, 1)[1])
            suffixes.append(marker + path_str.split(marker, 1)[1])

    # Build candidate roots
    manifest_parent = os.path.dirname(manifest_dir)
    roots = []
    if data_root:
        roots.append(str(data_root).replace("\\", "/"))
    roots.append(manifest_dir)
    roots.append(manifest_parent)

    env_root = os.environ.get("LDCT_DATASET_ROOT", "")
    if env_root:
        roots.append(env_root.replace("\\", "/"))

    roots.extend([
        "/content/FastDDPM_Data",
        "/content/drive/MyDrive/Thesis_Mus",
    ])
    # Deduplicate while preserving order
    seen = set()
    unique_roots = []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            unique_roots.append(r)

    # 2-5. Try combinations
    for root in unique_roots:
        for suffix in suffixes:
            candidate = os.path.normpath(os.path.join(root, suffix))
            if os.path.exists(candidate):
                return candidate
        # Also try just the basename under root + patient subfolder
        candidate = os.path.normpath(os.path.join(root, basename))
        if os.path.exists(candidate):
            return candidate

    return None


def load_manifest_samples(manifest_path, data_root=None):
    """
    Load samples from a V3 manifest CSV.

    Returns list of dicts with keys:
        patient_id, slice_idx, ld_path, fd_path
    """
    manifest_path = os.path.normpath(manifest_path)
    manifest_dir = os.path.dirname(manifest_path)

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples = []
    with open(manifest_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]

        col_patient = _find_col(fieldnames, _COL_PATIENT_VARIANTS)
        col_slice = _find_col(fieldnames, _COL_SLICE_VARIANTS)
        col_ld = _find_col(fieldnames, _COL_LD_VARIANTS)
        col_fd = _find_col(fieldnames, _COL_FD_VARIANTS)

        if col_ld is None:
            raise ValueError(
                f"Cannot find LD path column in manifest.\n"
                f"Available columns: {fieldnames}\n"
                f"Expected one of: {_COL_LD_VARIANTS}"
            )

        print(f"[MANIFEST] Columns detected: patient={col_patient}, "
              f"slice={col_slice}, ld={col_ld}, fd={col_fd}")

        for row_idx, row in enumerate(reader):
            patient_id = str(row.get(col_patient, "")).strip() if col_patient else ""
            slice_val = str(row.get(col_slice, "")).strip() if col_slice else ""

            ld_raw = str(row.get(col_ld, "")).strip()
            fd_raw = str(row.get(col_fd, "")).strip() if col_fd else ""

            # Resolve paths
            ld_path = _resolve_path(ld_raw, manifest_dir, data_root)
            fd_path = _resolve_path(fd_raw, manifest_dir, data_root) if fd_raw else None

            if ld_path is None:
                print(f"  [WARN] Row {row_idx}: Cannot resolve LD path: {ld_raw}")
                continue

            # Infer patient_id from filename if not present
            if not patient_id:
                stem = Path(ld_path).stem
                patient_id = stem.split("_")[0] if "_" in stem else stem

            # Parse slice index
            slice_idx = -1
            if slice_val:
                try:
                    slice_idx = int(slice_val)
                except ValueError:
                    pass
            if slice_idx < 0:
                # Try to parse from filename
                import re
                stem = Path(ld_path).stem
                m = re.search(r"(\d+)", stem.split("_")[-1] if "_" in stem else stem)
                if m:
                    slice_idx = int(m.group(1))

            samples.append({
                "patient_id": patient_id,
                "slice_idx": slice_idx,
                "ld_path": ld_path,
                "fd_path": fd_path,
            })

    print(f"[MANIFEST] Loaded {len(samples)} rows from {manifest_path}")
    return samples


def build_25d_triplets(samples):
    """
    Group samples by patient_id, sort by slice_idx, and build valid 2.5D
    triplets (center slices with both z-1 and z+1 neighbors available).

    Returns list of dicts:
        patient_id, slice_idx,
        ld_paths: [ld_z-1, ld_z, ld_z+1],
        fd_path: fd_z (or None)
    """
    # Group by patient
    groups = defaultdict(list)
    for s in samples:
        groups[s["patient_id"]].append(s)

    # Sort each group by slice_idx
    for pid in groups:
        groups[pid].sort(key=lambda x: x["slice_idx"])

    triplets = []
    for pid, slices in groups.items():
        n = len(slices)
        for center_pos in range(n):
            if center_pos == 0 or center_pos == n - 1:
                continue  # Skip boundary slices

            prev_s = slices[center_pos - 1]
            curr_s = slices[center_pos]
            next_s = slices[center_pos + 1]

            # Verify consecutive slice indices
            # (they should be neighbors, but we accept any ordering)
            triplets.append({
                "patient_id": pid,
                "slice_idx": curr_s["slice_idx"],
                "ld_paths": [prev_s["ld_path"], curr_s["ld_path"], next_s["ld_path"]],
                "fd_path": curr_s["fd_path"],
            })

    print(f"[TRIPLET] Built {len(triplets)} valid 2.5D triplets "
          f"from {len(samples)} rows, {len(groups)} patients")
    return triplets


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def load_hu_npy_512(path, hu_min=-1000, hu_max=400):
    """
    Load a raw HU NPY file and preprocess identically to training:
    - Clip to [hu_min, hu_max]
    - Normalize to [-1, 1]
    - Return as float32 numpy array (H, W)

    Does NOT resize or crop.
    """
    arr = np.load(path).astype(np.float32)

    # Handle 3D arrays with singleton channel dim
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape} at {path}")

    # HU window + normalize to [-1, 1]
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)   # [0, 1]
    arr = arr * 2.0 - 1.0                       # [-1, 1]
    return arr


# ---------------------------------------------------------------------------
# Patch coordinate generation
# ---------------------------------------------------------------------------

def generate_patch_coords(img_h, img_w, patch_size=256, overlap=128):
    """
    Generate sliding-window patch coordinates for any image size >= patch_size.

    Returns list of (y, x) top-left corners. Guarantees:
    - Always includes (0, 0)
    - Always includes (H - patch_size, W - patch_size)
    - No duplicate coordinates
    - Covers the full image with given overlap
    """
    stride = patch_size - overlap
    assert stride > 0, f"Overlap ({overlap}) must be < patch_size ({patch_size})"
    assert img_h >= patch_size, f"Image height ({img_h}) < patch_size ({patch_size})"
    assert img_w >= patch_size, f"Image width ({img_w}) < patch_size ({patch_size})"

    def _coords_1d(length, p_size, s):
        coords = list(range(0, length - p_size + 1, s))
        # Always include the last valid position
        last = length - p_size
        if last not in coords:
            coords.append(last)
        return sorted(set(coords))

    ys = _coords_1d(img_h, patch_size, stride)
    xs = _coords_1d(img_w, patch_size, stride)

    positions = [(y, x) for y in ys for x in xs]
    return positions


# ---------------------------------------------------------------------------
# Fast-DDPM sampling for a single patch
# ---------------------------------------------------------------------------

def build_diffusion_seq(scheduler_type, timesteps, num_diffusion_timesteps=1000):
    """
    Build the timestep sequence for Fast-DDPM sampling.
    Returns a list of ints.
    """
    if scheduler_type == "uniform":
        skip = num_diffusion_timesteps // timesteps
        seq = list(range(-1, num_diffusion_timesteps, skip))
        seq[0] = 0
    elif scheduler_type == "non-uniform":
        if timesteps == 10:
            seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]
        else:
            num_1 = int(timesteps * 0.4)
            num_2 = int(timesteps * 0.6)
            stage_1 = np.linspace(0, 699, num_1 + 1)[:-1]
            stage_2 = np.linspace(699, 999, num_2)
            stage_1 = np.ceil(stage_1).astype(int)
            stage_2 = np.ceil(stage_2).astype(int)
            seq = list(np.concatenate((stage_1, stage_2)))
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    return seq


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """Build beta schedule (copied from runners/diffusion.py)."""
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                        num_diffusion_timesteps, dtype=np.float64) ** 2
        )
    elif beta_schedule == "alpha_cosine":
        s = 0.008
        ts = np.arange(0, num_diffusion_timesteps + 1, dtype=np.float64) / num_diffusion_timesteps
        alphas_cumprod = np.cos((ts + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=None, a_max=0.999)
    else:
        raise NotImplementedError(f"Beta schedule '{beta_schedule}' not implemented")
    return betas


def denoise_patch(model, condition_patch, seq, betas, device, eta=0.0):
    """
    Run Fast-DDPM sampling on a single 256x256 patch.

    Args:
        model: UNet model (raw, not DataParallel)
        condition_patch: [1, 3, 256, 256] tensor (LD triplet patch)
        seq: timestep sequence (list of ints)
        betas: beta schedule tensor on device
        device: torch device
        eta: DDIM eta

    Returns:
        denoised_patch: [1, 1, 256, 256] tensor
    """
    condition_patch = condition_patch.to(device)

    # Start from Gaussian noise
    x_noise = torch.randn(1, 1, 256, 256, device=device)

    # sg_generalized_steps expects: model(cat([xt, x_img], dim=1), t)
    # where xt = [B,1,H,W] (noisy target) and x_img = [B,3,H,W] (condition)
    xs, _ = sg_generalized_steps(
        x_noise, condition_patch, seq, model, betas, eta=eta
    )
    # xs is a list of states; last element is the final denoised output
    denoised = xs[-1]  # [1, 1, 256, 256]
    return denoised


# ---------------------------------------------------------------------------
# Full-512 sliding-window inference
# ---------------------------------------------------------------------------

def _make_gaussian_weight_map(patch_size: int) -> np.ndarray:
    """Build a 2D Gaussian weight map for sliding-window blending.

    The weight is highest at the patch center and tapers smoothly toward
    the edges (~3-sigma at the boundary). This reduces seam artifacts at
    patch boundaries because the model's predictions are most reliable
    in the center of each patch (where it has full receptive field context).

    Args:
        patch_size: Size of the (square) patch.

    Returns:
        Gaussian weight map of shape (patch_size, patch_size), values in (0, 1].
    """
    sigma = patch_size / 6.0  # ~3-sigma at the patch boundary
    center = patch_size // 2
    yy, xx = np.mgrid[0:patch_size, 0:patch_size]
    d2 = (yy - center) ** 2 + (xx - center) ** 2
    weight = np.exp(-d2 / (2.0 * sigma ** 2))
    return weight.astype(np.float64)


def infer_full_512(model, ld_triplet, seq, betas, device, patch_size=256,
                   overlap=128, eta=0.0):
    """
    Denoise a full 512x512 CT slice using sliding-window patch inference.

    Args:
        model: UNet model in eval mode
        ld_triplet: numpy array [3, H, W] — the 2.5D LD condition
        seq: diffusion timestep sequence
        betas: beta tensor on device
        device: torch device
        patch_size: patch size (256)
        overlap: overlap pixels (128)
        eta: DDIM eta

    Returns:
        denoised_512: numpy array [H, W] in [-1, 1]
    """
    _, img_h, img_w = ld_triplet.shape
    positions = generate_patch_coords(img_h, img_w, patch_size, overlap)

    output_sum = np.zeros((img_h, img_w), dtype=np.float64)
    weight_map = np.zeros((img_h, img_w), dtype=np.float64)
    gaussian_weight = _make_gaussian_weight_map(patch_size)

    ld_tensor = torch.from_numpy(ld_triplet).float()  # [3, H, W]

    print(f"    Patches: {len(positions)} positions, "
          f"patch={patch_size}, overlap={overlap}, stride={patch_size - overlap}")

    for patch_idx, (y, x) in enumerate(positions):
        # Extract condition patch
        cond_patch = ld_tensor[:, y:y + patch_size, x:x + patch_size]  # [3, 256, 256]
        cond_patch = cond_patch.unsqueeze(0)  # [1, 3, 256, 256]

        # Debug print for first patch
        if patch_idx == 0:
            print(f"    Patch[0] condition shape: {list(cond_patch.shape)}")
            # Peek at what the model will receive
            dummy_noise = torch.randn(1, 1, patch_size, patch_size)
            model_input_shape = list(torch.cat([dummy_noise, cond_patch], dim=1).shape)
            print(f"    Patch[0] model input shape: {model_input_shape}")

        # Denoise
        denoised = denoise_patch(model, cond_patch, seq, betas, device, eta=eta)
        denoised_np = denoised.squeeze().cpu().numpy()  # [256, 256]

        # Accumulate (Gaussian-weighted blending to reduce seam artifacts)
        output_sum[y:y + patch_size, x:x + patch_size] += denoised_np * gaussian_weight
        weight_map[y:y + patch_size, x:x + patch_size] += gaussian_weight

    # Average overlapping regions
    denoised_512 = output_sum / np.maximum(weight_map, 1e-8)
    denoised_512 = np.clip(denoised_512, -1.0, 1.0).astype(np.float32)

    print(f"    Stitched output shape: {denoised_512.shape}")
    return denoised_512


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def to_uint8(arr_minus1_1):
    """Convert [-1, 1] float array to [0, 255] uint8."""
    arr_01 = np.clip((arr_minus1_1 + 1.0) / 2.0, 0.0, 1.0)
    return (arr_01 * 255.0).round().astype(np.uint8)


def save_grayscale_png(arr_minus1_1, path):
    """Save a [-1,1] float array as a grayscale PNG."""
    from PIL import Image
    img = Image.fromarray(to_uint8(arr_minus1_1), mode="L")
    img.save(path)


def save_comparison_png(ld_center, denoised, fd=None, path="comparison.png"):
    """
    Save a horizontal comparison panel:
    - 2 panels if FD unavailable: [Low-dose | Denoised]
    - 3 panels if FD available: [Low-dose | Denoised | Full-dose]

    All inputs in [-1, 1].
    """
    from PIL import Image, ImageDraw, ImageFont

    panels = [
        ("Low-dose (LD)", to_uint8(ld_center)),
        ("Denoised", to_uint8(denoised)),
    ]
    if fd is not None:
        panels.append(("Full-dose (FD)", to_uint8(fd)))

    h, w = panels[0][1].shape
    n_panels = len(panels)
    pad = 4
    label_h = 24

    canvas_w = n_panels * w + (n_panels - 1) * pad
    canvas_h = h + label_h
    canvas = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(canvas)

    # Try to use a font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()

    for i, (label, arr) in enumerate(panels):
        x_off = i * (w + pad)
        panel_img = Image.fromarray(arr, mode="L")
        canvas.paste(panel_img, (x_off, label_h))
        # Draw label centered above panel
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
        except AttributeError:
            tw = len(label) * 8
        tx = x_off + (w - tw) // 2
        draw.text((tx, 2), label, fill=255, font=font)

    canvas.save(path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_minus1_1, target_minus1_1):
    """
    Compute PSNR and SSIM on [0, 1] scale.

    Args:
        pred_minus1_1: predicted array in [-1, 1]
        target_minus1_1: ground truth array in [-1, 1]

    Returns:
        psnr, ssim (float)
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    pred_01 = np.clip((pred_minus1_1 + 1.0) / 2.0, 0.0, 1.0)
    target_01 = np.clip((target_minus1_1 + 1.0) / 2.0, 0.0, 1.0)

    psnr = psnr_fn(target_01, pred_01, data_range=1.0)
    ssim = ssim_fn(target_01, pred_01, data_range=1.0)
    return float(psnr), float(ssim)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-512 sliding-window inference for Fast-DDPM 2.5D CT denoising"
    )
    parser.add_argument("--config", type=str,
                        default="configs/ldfd_v3_2p5d_5090_full.yml",
                        help="YAML config path")
    parser.add_argument("--manifest", type=str, default=None,
                        help="V3 CSV manifest path (overrides config)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint path (ckpt.pth / best_psnr.pth / etc.)")
    parser.add_argument("--output_dir", type=str, default="./full512_results",
                        help="Output directory")
    parser.add_argument("--max_samples", type=int, default=20,
                        help="Maximum number of samples to process")
    parser.add_argument("--overlap", type=int, default=128,
                        help="Patch overlap in pixels")
    parser.add_argument("--scheduler_type", type=str, default="uniform",
                        choices=["uniform", "non-uniform"],
                        help="Fast-DDPM scheduler type")
    parser.add_argument("--timesteps", type=int, default=10,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta parameter (0=deterministic)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda / cpu)")
    parser.add_argument("--save_npy", action="store_true", default=True,
                        help="Save denoised_512.npy (default: True)")
    parser.add_argument("--no_save_npy", action="store_true", default=False,
                        help="Disable saving denoised_512.npy")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_save_npy:
        args.save_npy = False

    # --- Seed ---
    set_seed(args.seed)
    print(f"[SEED] {args.seed}")

    # --- Device ---
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[DEVICE] {device}")

    # --- Load config ---
    with open(args.config, "r") as f:
        config = to_ns(yaml.safe_load(f))
    print(f"[CONFIG] {args.config}")

    # --- Determine manifest path ---
    manifest_path = args.manifest
    if manifest_path is None:
        # Try config paths in order of preference
        for attr in ["test_dataroot", "test_manifest", "val_dataroot", "val_manifest"]:
            candidate = getattr(config.data, attr, None)
            if candidate and os.path.isfile(str(candidate)):
                manifest_path = str(candidate)
                break
    if manifest_path is None:
        raise FileNotFoundError(
            "No manifest specified via --manifest and none found in config. "
            "Please provide --manifest /path/to/test_v3.csv"
        )
    print(f"[MANIFEST] {manifest_path}")

    # --- Data root for path resolution ---
    data_root = getattr(config.data, "data_root", None)

    # --- HU windowing params ---
    hu_min = getattr(config.data, "hu_window_min", -1000)
    hu_max = getattr(config.data, "hu_window_max", 400)
    print(f"[HU] Window: [{hu_min}, {hu_max}] → [-1, 1]")

    # --- Load manifest & build triplets ---
    raw_samples = load_manifest_samples(manifest_path, data_root=data_root)
    triplets = build_25d_triplets(raw_samples)

    if len(triplets) == 0:
        print("[ERROR] No valid 2.5D triplets found. Check manifest and data paths.")
        sys.exit(1)

    # Limit samples
    if args.max_samples > 0:
        random.seed(args.seed)
        random.shuffle(triplets) 
        triplets = triplets[:args.max_samples]
    print(f"[SAMPLES] Processing {len(triplets)} triplets")

    # --- Load model ---
    model = Model(config).to(device)
    model = load_checkpoint(model, args.ckpt, config, device)
    model.eval()
    print(f"[MODEL] in_channels={config.model.in_channels}, "
          f"out_ch={config.model.out_ch}, image_size={config.data.image_size}")

    # --- Build diffusion schedule ---
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)

    seq = build_diffusion_seq(
        args.scheduler_type, args.timesteps,
        config.diffusion.num_diffusion_timesteps
    )
    print(f"[SCHEDULE] {args.scheduler_type}, {args.timesteps} steps → seq={seq}")

    # --- Output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Metrics storage ---
    metrics_rows = []

    # --- Process each triplet ---
    print("\n" + "=" * 70)
    print("Starting Full-512 Sliding-Window Inference")
    print("=" * 70)

    total_start = time.time()

    for sample_idx, triplet in enumerate(triplets):
        pid = triplet["patient_id"]
        sidx = triplet["slice_idx"]
        sample_name = f"{pid}_{sidx}"
        sample_dir = os.path.join(args.output_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        print(f"\n--- Sample {sample_idx + 1}/{len(triplets)}: "
              f"patient={pid}, slice={sidx} ---")

        t_start = time.time()

        # 1. Load 2.5D LD triplet at full 512x512
        ld_slices = []
        for i, ld_path in enumerate(triplet["ld_paths"]):
            arr = load_hu_npy_512(ld_path, hu_min=hu_min, hu_max=hu_max)
            ld_slices.append(arr)

        ld_triplet = np.stack(ld_slices, axis=0)  # [3, H, W]
        print(f"    LD triplet shape: {list(ld_triplet.shape)}")

        # Center LD slice for baseline metrics and visualization
        ld_center = ld_triplet[1]  # [H, W] — the z=0 offset, i.e., the center

        # 2. Load FD ground truth if available
        fd_512 = None
        if triplet["fd_path"] and os.path.exists(triplet["fd_path"]):
            fd_512 = load_hu_npy_512(triplet["fd_path"], hu_min=hu_min, hu_max=hu_max)
            print(f"    FD shape: {list(fd_512.shape)}")

        # 3. Run sliding-window inference
        with torch.no_grad():
            denoised_512 = infer_full_512(
                model, ld_triplet, seq, betas, device,
                patch_size=config.data.image_size,
                overlap=args.overlap,
                eta=args.eta,
            )

        t_elapsed = time.time() - t_start

        # 4. Save outputs
        save_grayscale_png(ld_center, os.path.join(sample_dir, "low_center_512.png"))
        save_grayscale_png(denoised_512, os.path.join(sample_dir, "denoised_512.png"))

        if fd_512 is not None:
            save_grayscale_png(fd_512, os.path.join(sample_dir, "full_dose_512.png"))

        save_comparison_png(
            ld_center, denoised_512, fd_512,
            os.path.join(sample_dir, "comparison_512.png")
        )

        if args.save_npy:
            np.save(os.path.join(sample_dir, "denoised_512.npy"), denoised_512)

        # 5. Compute metrics
        metric_row = {
            "patient_id": pid,
            "slice_idx": sidx,
            "ld_psnr": "",
            "ld_ssim": "",
            "denoised_psnr": "",
            "denoised_ssim": "",
        }

        if fd_512 is not None:
            # LD baseline: center LD[z] vs FD[z]
            ld_psnr, ld_ssim = compute_metrics(ld_center, fd_512)
            # Denoised vs FD
            dn_psnr, dn_ssim = compute_metrics(denoised_512, fd_512)

            metric_row["ld_psnr"] = f"{ld_psnr:.4f}"
            metric_row["ld_ssim"] = f"{ld_ssim:.4f}"
            metric_row["denoised_psnr"] = f"{dn_psnr:.4f}"
            metric_row["denoised_ssim"] = f"{dn_ssim:.4f}"

            print(f"    LD baseline  → PSNR: {ld_psnr:.4f}, SSIM: {ld_ssim:.4f}")
            print(f"    Denoised     → PSNR: {dn_psnr:.4f}, SSIM: {dn_ssim:.4f}")
            print(f"    Improvement  → ΔPSNR: {dn_psnr - ld_psnr:+.4f}, "
                  f"ΔSSIM: {dn_ssim - ld_ssim:+.4f}")
        else:
            print("    [No FD available — metrics skipped]")

        metrics_rows.append(metric_row)
        print(f"    Time: {t_elapsed:.1f}s")
        print(f"    Output: {sample_dir}")

    # --- Save metrics CSV ---
    total_elapsed = time.time() - total_start
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["patient_id", "slice_idx", "ld_psnr", "ld_ssim",
                           "denoised_psnr", "denoised_ssim"]
        )
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\n[CSV] Metrics saved to: {csv_path}")

    # --- Summary statistics ---
    ld_psnrs = [float(r["ld_psnr"]) for r in metrics_rows if r["ld_psnr"]]
    ld_ssims = [float(r["ld_ssim"]) for r in metrics_rows if r["ld_ssim"]]
    dn_psnrs = [float(r["denoised_psnr"]) for r in metrics_rows if r["denoised_psnr"]]
    dn_ssims = [float(r["denoised_ssim"]) for r in metrics_rows if r["denoised_ssim"]]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Samples processed: {len(triplets)}")
    print(f"Total time: {total_elapsed:.1f}s "
          f"({total_elapsed / max(len(triplets), 1):.1f}s/sample)")

    if dn_psnrs:
        print(f"\nLD Baseline:  PSNR = {np.mean(ld_psnrs):.4f} ± {np.std(ld_psnrs):.4f}, "
              f"SSIM = {np.mean(ld_ssims):.4f} ± {np.std(ld_ssims):.4f}")
        print(f"Denoised:     PSNR = {np.mean(dn_psnrs):.4f} ± {np.std(dn_psnrs):.4f}, "
              f"SSIM = {np.mean(dn_ssims):.4f} ± {np.std(dn_ssims):.4f}")
        delta_p = np.mean(dn_psnrs) - np.mean(ld_psnrs)
        delta_s = np.mean(dn_ssims) - np.mean(ld_ssims)
        print(f"Improvement:  ΔPSNR = {delta_p:+.4f}, ΔSSIM = {delta_s:+.4f}")
    else:
        print("\n[No FD ground truth available — no aggregate metrics]")

    print(f"\nOutputs saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
