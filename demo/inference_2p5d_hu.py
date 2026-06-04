"""
HU-NPY / DICOM Clinical-style 2.5D Mode — Inference Module

This module implements the Dataset V3 raw-HU NPY / 2.5D Fast-DDPM pipeline.

Pipeline:
    - Input: raw-HU NPY files (512×512 float32) or DICOM series
    - HU windowing: clip to [-1000, 400]
    - Normalize: x = 2 * (clip(HU) - (-1000)) / (400 - (-1000)) - 1  →  [-1, 1]
    - 2.5D condition: stack LD[z-1], LD[z], LD[z+1]  →  [3, H, W]
    - Model type: "sg" (single-guide)
    - in_channels = 4  (1 noisy target x_t + 3 LD condition channels)
    - out_ch = 1
    - Concatenation order: [x_t, condition]  (inside sg_generalized_steps)
    - 512×512 inference via sliding-window patches of 256×256
    - Target: FD[z] (center slice only)

WARNING: Do NOT use the PNG/original checkpoint for this mode.
WARNING: This is a research prototype, not for clinical diagnosis.
"""

import os
import sys
import csv
import re
import time
import math
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.diffusion import Model
from models.ema import EMAHelper
from functions.denoising import sg_generalized_steps, sg_ddpm_steps


# ============================================================================
# Config loading
# ============================================================================

def _to_ns(x):
    """Recursively convert dicts to SimpleNamespace for attribute access."""
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    return x


def load_config_2p5d(config_path: str) -> SimpleNamespace:
    """
    Load a YAML config file for the 2.5D HU-NPY model.

    Expected config: configs/ldfd_v3_2p5d_5090_full.yml
    Expected: model.in_channels = 4

    Args:
        config_path: path to .yml config file.

    Returns:
        SimpleNamespace config object.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"2.5D config not found: {config_path}\n"
            "Please provide the correct config path for the 2.5D HU-NPY model."
        )

    with open(config_path, "r") as f:
        config = _to_ns(yaml.safe_load(f))

    in_ch = getattr(config.model, "in_channels", None)
    if in_ch is not None and in_ch != 4:
        print(
            f"[2.5D_WARN] Config has in_channels={in_ch}, expected 4 for 2.5D mode.\n"
            f"  (1 noisy target + 3 LD condition channels)\n"
            f"  If this is intentional, you can ignore this warning."
        )

    return config


# ============================================================================
# Model loading
# ============================================================================

def load_model_2p5d(
    config: SimpleNamespace,
    ckpt_path: str,
    device: str = "cuda",
) -> tuple:
    """
    Build and load the 2.5D HU-NPY Fast-DDPM model.

    Args:
        config: parsed YAML config.
        ckpt_path: path to checkpoint file.
        device: "cuda" or "cpu".

    Returns:
        (model, betas) tuple.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"2.5D checkpoint not found: {ckpt_path}\n"
            "Please provide the correct checkpoint path for the 2.5D model."
        )

    if device == "cuda" and not torch.cuda.is_available():
        print("[2.5D] CUDA not available, falling back to CPU.")
        device = "cpu"
    device_obj = torch.device(device)

    # Build model
    model = Model(config)
    print(f"[2.5D] Model built: in_channels={config.model.in_channels}, "
          f"out_ch={config.model.out_ch}, image_size={config.data.image_size}")

    # Load checkpoint
    print(f"[2.5D] Loading checkpoint: {ckpt_path}")
    states = torch.load(ckpt_path, map_location="cpu")

    if isinstance(states, (list, tuple)):
        state_dict = states[0]
        if len(states) >= 4:
            print(f"[2.5D] Checkpoint epoch={states[2]}, step={states[3]}")
        if len(states) >= 7:
            print(f"[2.5D] Best PSNR={states[5]:.6f}, Best SSIM={states[6]:.6f}")
    elif isinstance(states, dict):
        state_dict = states.get("model", states)
    else:
        raise ValueError(f"Unknown checkpoint format at {ckpt_path}")

    # Strip "module." prefix
    clean = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "", 1) if k.startswith("module.") else k
        clean[key] = v
    model.load_state_dict(clean, strict=True)

    # Apply EMA
    ema_applied = False
    if getattr(config.model, "ema", False) and isinstance(states, (list, tuple)):
        ema_state = None
        if len(states) >= 5 and isinstance(states[4], dict):
            ema_state = states[4]

        if ema_state is not None:
            print("[2.5D] Applying EMA weights")
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(ema_state)
            ema_helper.ema(model)
            ema_applied = True

    if not ema_applied:
        print("[2.5D] No EMA applied (not available or disabled)")

    model = model.to(device_obj)
    model.eval()

    # Build beta schedule
    betas = _get_beta_schedule(
        config.diffusion.beta_schedule,
        config.diffusion.beta_start,
        config.diffusion.beta_end,
        config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device_obj)

    print(f"[2.5D] Model loaded successfully on {device}")
    return model, betas


# ============================================================================
# Beta schedule and diffusion sequence
# ============================================================================

def _get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """Build beta schedule."""
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


def build_diffusion_seq(scheduler_type, timesteps, num_diffusion_timesteps=1000):
    """Build the timestep sequence for Fast-DDPM sampling."""
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


# ============================================================================
# HU NPY preprocessing
# ============================================================================

def load_hu_npy(path: str, hu_min: float = -1000, hu_max: float = 400) -> np.ndarray:
    """
    Load a raw-HU NPY file and normalize to [-1, 1].

    Processing (matches Dataset V3 training):
        1. Load float32 array
        2. Clip to [hu_min, hu_max]
        3. Normalize: x = 2 * (clip(HU) - hu_min) / (hu_max - hu_min) - 1

    Args:
        path: path to .npy file containing raw Hounsfield Unit values.
        hu_min: lower HU window bound (default -1000).
        hu_max: upper HU window bound (default 400).

    Returns:
        float32 numpy array [H, W] in [-1, 1].
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NPY file not found: {path}")

    arr = np.load(path).astype(np.float32)

    # Handle singleton channel dimension
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


def build_2p5d_condition(
    ld_paths: list,
    hu_min: float = -1000,
    hu_max: float = 400,
) -> np.ndarray:
    """
    Build the 2.5D condition tensor from 3 consecutive LD slice paths.

    Args:
        ld_paths: [ld_z-1_path, ld_z_path, ld_z+1_path]
        hu_min: HU window min.
        hu_max: HU window max.

    Returns:
        float32 numpy array [3, H, W] in [-1, 1].
    """
    assert len(ld_paths) == 3, f"Expected 3 LD paths, got {len(ld_paths)}"

    slices = []
    for p in ld_paths:
        slices.append(load_hu_npy(p, hu_min, hu_max))

    return np.stack(slices, axis=0)  # [3, H, W]


def build_2p5d_condition_from_arrays(
    ld_arrays: list,
) -> np.ndarray:
    """
    Build the 2.5D condition tensor from 3 pre-normalized LD arrays.

    Args:
        ld_arrays: list of 3 numpy arrays [H, W] in [-1, 1].

    Returns:
        float32 numpy array [3, H, W] in [-1, 1].
    """
    assert len(ld_arrays) == 3, f"Expected 3 arrays, got {len(ld_arrays)}"
    return np.stack(ld_arrays, axis=0).astype(np.float32)


# ============================================================================
# Manifest loading
# ============================================================================

_COL_PATIENT_VARIANTS = ["patient_id", "Patient_ID", "PatientID", "PATIENT_ID"]
_COL_SLICE_VARIANTS = ["slice_idx", "slice_id", "Slice_ID", "SliceIdx", "SLICE_IDX"]
_COL_LD_VARIANTS = ["LD_Path", "low_path", "low_npy_path", "ld_path", "LD_path"]
_COL_FD_VARIANTS = ["FD_Path", "full_path", "full_npy_path", "fd_path", "FD_path"]


def _find_col(fieldnames, variants):
    """Find the first matching column name from variants."""
    for v in variants:
        if v in fieldnames:
            return v
    lower_map = {f.lower(): f for f in fieldnames}
    for v in variants:
        if v.lower() in lower_map:
            return lower_map[v.lower()]
    return None


def _resolve_path(path_str, manifest_dir, data_root=None):
    """Resolve a file path from a manifest entry."""
    path_str = str(path_str).strip().strip('"').strip("'").replace("\\", "/")

    if not path_str:
        return None

    # Absolute path exists
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return os.path.normpath(path_str)

    basename = os.path.basename(path_str)
    suffixes = [path_str]
    for marker in ["01_processed_npy_512/", "02_processed_npy_256/",
                    "LDCT_DATASET_V2/", "FastDDPM_Data/"]:
        if marker in path_str:
            suffixes.append(path_str.split(marker, 1)[1])
            suffixes.append(marker + path_str.split(marker, 1)[1])

    manifest_parent = os.path.dirname(manifest_dir)
    roots = []
    if data_root:
        roots.append(str(data_root).replace("\\", "/"))
    roots.append(manifest_dir)
    roots.append(manifest_parent)

    env_root = os.environ.get("LDCT_DATASET_ROOT", "")
    if env_root:
        roots.append(env_root.replace("\\", "/"))

    seen = set()
    unique_roots = []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            unique_roots.append(r)

    for root in unique_roots:
        for suffix in suffixes:
            candidate = os.path.normpath(os.path.join(root, suffix))
            if os.path.exists(candidate):
                return candidate
        candidate = os.path.normpath(os.path.join(root, basename))
        if os.path.exists(candidate):
            return candidate

    return None


def load_manifest_samples(manifest_path: str, data_root: str = None) -> list:
    """
    Load samples from a V3 manifest CSV.

    Returns list of dicts: {patient_id, slice_idx, ld_path, fd_path}
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
                f"Cannot find LD path column.\n"
                f"Available: {fieldnames}\nExpected one of: {_COL_LD_VARIANTS}"
            )

        print(f"[MANIFEST] Columns: patient={col_patient}, "
              f"slice={col_slice}, ld={col_ld}, fd={col_fd}")

        for row_idx, row in enumerate(reader):
            patient_id = str(row.get(col_patient, "")).strip() if col_patient else ""
            slice_val = str(row.get(col_slice, "")).strip() if col_slice else ""

            ld_raw = str(row.get(col_ld, "")).strip()
            fd_raw = str(row.get(col_fd, "")).strip() if col_fd else ""

            ld_path = _resolve_path(ld_raw, manifest_dir, data_root)
            fd_path = _resolve_path(fd_raw, manifest_dir, data_root) if fd_raw else None

            if ld_path is None:
                continue  # Skip unresolvable paths (will be common without data)

            if not patient_id:
                stem = Path(ld_path).stem
                patient_id = stem.split("_")[0] if "_" in stem else stem

            slice_idx = -1
            if slice_val:
                try:
                    slice_idx = int(slice_val)
                except ValueError:
                    pass
            if slice_idx < 0:
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


def build_25d_triplets(samples: list) -> list:
    """
    Group samples by patient, sort by slice_idx, build valid 2.5D triplets.

    Returns list of dicts:
        {patient_id, slice_idx, ld_paths: [z-1, z, z+1], fd_path}
    """
    groups = defaultdict(list)
    for s in samples:
        groups[s["patient_id"]].append(s)

    for pid in groups:
        groups[pid].sort(key=lambda x: x["slice_idx"])

    triplets = []
    for pid, slices in groups.items():
        n = len(slices)
        for center_pos in range(1, n - 1):  # Skip boundary slices
            prev_s = slices[center_pos - 1]
            curr_s = slices[center_pos]
            next_s = slices[center_pos + 1]

            triplets.append({
                "patient_id": pid,
                "slice_idx": curr_s["slice_idx"],
                "ld_paths": [prev_s["ld_path"], curr_s["ld_path"], next_s["ld_path"]],
                "fd_path": curr_s["fd_path"],
                "display_name": f"{pid}_slice{curr_s['slice_idx']}",
            })

    print(f"[TRIPLET] Built {len(triplets)} valid 2.5D triplets "
          f"from {len(samples)} rows, {len(groups)} patients")
    return triplets


# ============================================================================
# Sliding-window inference
# ============================================================================

def generate_patch_coords(img_h, img_w, patch_size=256, overlap=128):
    """Generate sliding-window patch coordinates."""
    stride = patch_size - overlap
    assert stride > 0
    assert img_h >= patch_size and img_w >= patch_size

    def _coords_1d(length, p_size, s):
        coords = list(range(0, length - p_size + 1, s))
        last = length - p_size
        if last not in coords:
            coords.append(last)
        return sorted(set(coords))

    ys = _coords_1d(img_h, patch_size, stride)
    xs = _coords_1d(img_w, patch_size, stride)
    return [(y, x) for y in ys for x in xs]


def denoise_patch(model, condition_patch, seq, betas, device, eta=0.0):
    """
    Run Fast-DDPM sampling on a single 256×256 patch.

    sg_generalized_steps internally does:
        model(cat([xt, x_img], dim=1), t)

    For 2.5D mode:
        xt = [1, 1, 256, 256] (noisy target)
        x_img = condition_patch = [1, 3, 256, 256] (LD triplet)
        model input = cat → [1, 4, 256, 256]

    This is the CORRECT concatenation order: [x_t, condition].
    Do NOT reverse it.
    """
    condition_patch = condition_patch.to(device)
    x_noise = torch.randn(1, 1, 256, 256, device=device)

    xs, _ = sg_generalized_steps(
        x_noise, condition_patch, seq, model, betas, eta=eta
    )
    denoised = xs[-1]  # [1, 1, 256, 256]
    return denoised


def infer_full_512_sliding(
    model,
    ld_triplet: np.ndarray,
    seq: list,
    betas: torch.Tensor,
    device: str,
    patch_size: int = 256,
    overlap: int = 128,
    eta: float = 0.0,
    progress_callback=None,
) -> np.ndarray:
    """
    Denoise a full 512×512 CT slice using sliding-window patch inference.

    Args:
        model: UNet model in eval mode.
        ld_triplet: [3, H, W] numpy array — the 2.5D LD condition in [-1, 1].
        seq: diffusion timestep sequence.
        betas: beta tensor on device.
        device: device string.
        patch_size: patch size (256).
        overlap: overlap pixels (128).
        eta: DDIM eta.
        progress_callback: optional callable(current, total) for UI progress.

    Returns:
        denoised_512: numpy array [H, W] in [-1, 1].
    """
    device_obj = torch.device(device)
    _, img_h, img_w = ld_triplet.shape
    positions = generate_patch_coords(img_h, img_w, patch_size, overlap)

    output_sum = np.zeros((img_h, img_w), dtype=np.float64)
    weight_map = np.zeros((img_h, img_w), dtype=np.float64)

    ld_tensor = torch.from_numpy(ld_triplet).float()

    print(f"    Patches: {len(positions)} positions, "
          f"patch={patch_size}, overlap={overlap}")

    with torch.no_grad():
        for patch_idx, (y, x) in enumerate(positions):
            cond_patch = ld_tensor[:, y:y + patch_size, x:x + patch_size]
            cond_patch = cond_patch.unsqueeze(0)  # [1, 3, 256, 256]

            denoised = denoise_patch(
                model, cond_patch, seq, betas, device_obj, eta=eta
            )
            denoised_np = denoised.squeeze().cpu().numpy()

            output_sum[y:y + patch_size, x:x + patch_size] += denoised_np
            weight_map[y:y + patch_size, x:x + patch_size] += 1.0

            if progress_callback:
                progress_callback(patch_idx + 1, len(positions))

    denoised_512 = output_sum / np.maximum(weight_map, 1e-8)
    denoised_512 = np.clip(denoised_512, -1.0, 1.0).astype(np.float32)
    return denoised_512


def run_inference_2p5d(
    model,
    betas: torch.Tensor,
    ld_triplet: np.ndarray,
    device: str = "cuda",
    timesteps: int = 10,
    scheduler_type: str = "uniform",
    overlap: int = 128,
    eta: float = 0.0,
    progress_callback=None,
) -> tuple:
    """
    Run 2.5D HU-NPY inference on a single sample.

    Args:
        model: loaded 2.5D UNet in eval mode.
        betas: beta schedule tensor on device.
        ld_triplet: [3, H, W] numpy array in [-1, 1].
        device: "cuda" or "cpu".
        timesteps: number of sampling steps.
        scheduler_type: "uniform" or "non-uniform".
        overlap: sliding-window overlap pixels.
        eta: DDIM eta.
        progress_callback: optional progress callback.

    Returns:
        (denoised_array, elapsed_time) where:
            denoised_array: [H, W] numpy in [-1, 1]
            elapsed_time: float seconds
    """
    num_diff_steps = betas.shape[0]
    seq = build_diffusion_seq(scheduler_type, timesteps, num_diff_steps)

    _, h, w = ld_triplet.shape
    patch_size = 256  # Must match model's expected resolution

    t_start = time.time()

    if h <= patch_size and w <= patch_size:
        # Small enough for direct inference (no sliding window needed)
        cond = torch.from_numpy(ld_triplet).float().unsqueeze(0)  # [1, 3, H, W]
        with torch.no_grad():
            denoised_tensor = denoise_patch(
                model, cond, seq, betas, torch.device(device), eta=eta
            )
        denoised = denoised_tensor.squeeze().cpu().numpy()
        denoised = np.clip(denoised, -1.0, 1.0).astype(np.float32)
    else:
        # Sliding-window inference for larger images
        denoised = infer_full_512_sliding(
            model, ld_triplet, seq, betas, device,
            patch_size=patch_size, overlap=overlap, eta=eta,
            progress_callback=progress_callback,
        )

    elapsed = time.time() - t_start
    return denoised, elapsed


# ============================================================================
# Post-processing
# ============================================================================

def postprocess_to_numpy_01(arr_minus1_1: np.ndarray) -> np.ndarray:
    """Convert [-1, 1] array to [0, 1] for display and metrics."""
    return np.clip((arr_minus1_1 + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics_2p5d(
    denoised_minus1_1: np.ndarray,
    fd_minus1_1: np.ndarray,
    ld_center_minus1_1: np.ndarray = None,
) -> dict:
    """
    Compute PSNR and SSIM for 2.5D mode.

    All inputs in [-1, 1]. Metrics computed on [0, 1] scale.

    Args:
        denoised_minus1_1: denoised image [H, W] in [-1, 1].
        fd_minus1_1: full-dose ground truth [H, W] in [-1, 1].
        ld_center_minus1_1: optional LD center slice [H, W] in [-1, 1] for baseline.

    Returns:
        dict with: denoised_psnr, denoised_ssim,
                   and optionally ld_psnr, ld_ssim, delta_psnr, delta_ssim
    """
    dn_01 = postprocess_to_numpy_01(denoised_minus1_1)
    fd_01 = postprocess_to_numpy_01(fd_minus1_1)

    result = {}
    result["denoised_psnr"] = float(psnr_fn(fd_01, dn_01, data_range=1.0))
    result["denoised_ssim"] = float(ssim_fn(fd_01, dn_01, data_range=1.0))

    if ld_center_minus1_1 is not None:
        ld_01 = postprocess_to_numpy_01(ld_center_minus1_1)
        result["ld_psnr"] = float(psnr_fn(fd_01, ld_01, data_range=1.0))
        result["ld_ssim"] = float(ssim_fn(fd_01, ld_01, data_range=1.0))
        result["delta_psnr"] = result["denoised_psnr"] - result["ld_psnr"]
        result["delta_ssim"] = result["denoised_ssim"] - result["ld_ssim"]

    return result
