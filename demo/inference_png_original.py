"""
PNG Original Fast-DDPM Mode — Inference Module

This module implements the ORIGINAL-style Fast-DDPM pipeline for CT denoising.
It uses the same model architecture and preprocessing as the original Fast-DDPM repo.

Pipeline:
    - Input: grayscale PNG (LDCT)
    - Resize to config.data.image_size (default 256)
    - Normalize to [-1, 1]  (rescaled mode: X = 2*X - 1)
    - Model type: "sg" (single-guide)
    - in_channels = 2  (1 noisy target x_t + 1 LD condition)
    - out_ch = 1
    - Concatenation order: [x_t, LD]  (inside sg_generalized_steps)
    - Output: denoised tensor → convert back to [0, 1] → display as PNG

WARNING: This mode does NOT use 2.5D neighboring slices.
WARNING: This mode does NOT use Dataset V3 raw-HU NPY logic.
WARNING: Do NOT use the 2.5D checkpoint for this mode.
"""

import os
import sys
import re
import time
import math
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import existing modules
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.diffusion import Model
from models.ema import EMAHelper
from functions.denoising import sg_generalized_steps, sg_ddpm_steps


# ============================================================================
# PNG Sample Folder Scanner
# ============================================================================

# Supported naming patterns for LD/FD pair detection:
#   {patient}_{slice}_ld.png  /  {patient}_{slice}_fd.png
#   {patient}_{slice}_LD.png  /  {patient}_{slice}_FD.png
# Also matches filenames with "low" / "full" variants.

_LD_SUFFIX_RE = re.compile(r"^(.+?)_(ld|low)\.png$", re.IGNORECASE)
_FD_SUFFIX_RE = re.compile(r"^(.+?)_(fd|full)\.png$", re.IGNORECASE)


def scan_png_sample_folder(folder_path: str) -> list:
    """
    Recursively scan a PNG sample folder and detect LD/FD pairs.

    Expected naming convention (case-insensitive):
        {patient_id}_{slice_idx}_ld.png  ↔  {patient_id}_{slice_idx}_fd.png

    The scanner searches all subdirectories recursively.

    Args:
        folder_path: root directory to scan.

    Returns:
        list of dicts sorted by (patient_id, slice_idx):
        [{
            "display_name": str,      # e.g. "C002_slice0"
            "patient_id": str,
            "slice_idx": int,
            "ld_path": str,           # absolute path to LD PNG
            "fd_path": str or None,   # absolute path to FD PNG (None if not found)
        }, ...]
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"PNG sample folder not found: {folder_path}")

    # Collect all PNG files recursively
    ld_files = {}   # key → path (key = stem without _ld suffix)
    fd_files = {}   # key → path

    for root, _dirs, files in os.walk(folder_path):
        for fname in files:
            if not fname.lower().endswith(".png"):
                continue
            fpath = os.path.join(root, fname)

            ld_match = _LD_SUFFIX_RE.match(fname)
            if ld_match:
                key = ld_match.group(1).lower()
                ld_files[key] = fpath
                continue

            fd_match = _FD_SUFFIX_RE.match(fname)
            if fd_match:
                key = fd_match.group(1).lower()
                fd_files[key] = fpath

    if not ld_files:
        print(f"[PNG_SCAN] No LD PNG files found in: {folder_path}")
        print(f"  Expected naming: *_ld.png or *_low.png")
        return []

    # Build paired samples
    samples = []
    for key, ld_path in ld_files.items():
        fd_path = fd_files.get(key, None)

        # Parse patient_id and slice_idx from key
        # key format: "c002_0" or "c002_100" etc.
        parts = key.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            patient_id = parts[0].upper()
            slice_idx = int(parts[1])
        else:
            patient_id = key.upper()
            slice_idx = 0

        samples.append({
            "display_name": f"{patient_id}_slice{slice_idx}",
            "patient_id": patient_id,
            "slice_idx": slice_idx,
            "ld_path": ld_path,
            "fd_path": fd_path,
        })

    # Sort by patient_id, then slice_idx
    samples.sort(key=lambda s: (s["patient_id"], s["slice_idx"]))

    n_paired = sum(1 for s in samples if s["fd_path"] is not None)
    n_patients = len(set(s["patient_id"] for s in samples))
    print(f"[PNG_SCAN] Found {len(samples)} LD images, {n_paired} with FD pairs, "
          f"{n_patients} patients in: {folder_path}")

    return samples


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


def load_config_png(config_path: str) -> SimpleNamespace:
    """
    Load a YAML config file for PNG/original Fast-DDPM mode.

    Args:
        config_path: absolute or relative path to a .yml config file.
                     Default expected: configs/ldfd_linear.yml

    Returns:
        SimpleNamespace config object with attribute access.

    Raises:
        FileNotFoundError: if config_path does not exist.
        ValueError: if config has unexpected in_channels for PNG mode.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"PNG config not found: {config_path}\n"
            "Please provide the correct config path for the PNG/original model."
        )

    with open(config_path, "r") as f:
        config = _to_ns(yaml.safe_load(f))

    # Validate this is a PNG/original config (in_channels should be 2)
    in_ch = getattr(config.model, "in_channels", None)
    if in_ch is not None and in_ch != 2:
        print(
            f"[PNG_WARN] Config has in_channels={in_ch}, expected 2 for PNG mode.\n"
            f"  If this is intentional (e.g. a custom config), you can ignore this warning.\n"
            f"  If you accidentally loaded a 2.5D config, please use the correct PNG config."
        )

    return config


# ============================================================================
# Model loading
# ============================================================================

def load_model_png(
    config: SimpleNamespace,
    ckpt_path: str,
    device: str = "cuda",
) -> tuple:
    """
    Build and load the PNG/original Fast-DDPM model.

    Args:
        config: parsed YAML config (SimpleNamespace).
        ckpt_path: path to the checkpoint file (.pth).
        device: "cuda" or "cpu".

    Returns:
        (model, betas) tuple where:
            model: loaded UNet in eval mode, on device
            betas: beta schedule tensor on device

    Raises:
        FileNotFoundError: if ckpt_path does not exist.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"PNG checkpoint not found: {ckpt_path}\n"
            "Please provide the correct checkpoint path for the PNG/original model."
        )

    if device == "cuda" and not torch.cuda.is_available():
        print("[PNG] CUDA not available, falling back to CPU.")
        device = "cpu"
    device_obj = torch.device(device)

    # --- Build model ---
    model = Model(config)
    print(f"[PNG] Model built: in_channels={config.model.in_channels}, "
          f"out_ch={config.model.out_ch}, image_size={config.data.image_size}")

    # --- Load checkpoint ---
    print(f"[PNG] Loading checkpoint: {ckpt_path}")
    states = torch.load(ckpt_path, map_location="cpu")

    # Extract state_dict from various checkpoint formats
    if isinstance(states, (list, tuple)):
        state_dict = states[0]
        if len(states) >= 4:
            print(f"[PNG] Checkpoint epoch={states[2]}, step={states[3]}")
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

    # --- Apply EMA if available ---
    ema_applied = False
    if getattr(config.model, "ema", False) and isinstance(states, (list, tuple)):
        ema_state = None
        if len(states) >= 5 and isinstance(states[4], dict):
            ema_state = states[4]

        if ema_state is not None:
            print("[PNG] Applying EMA weights")
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(ema_state)
            ema_helper.ema(model)
            ema_applied = True

    if not ema_applied:
        print("[PNG] No EMA applied (not available or disabled)")

    model = model.to(device_obj)
    model.eval()

    # --- Build beta schedule ---
    betas = _get_beta_schedule(
        config.diffusion.beta_schedule,
        config.diffusion.beta_start,
        config.diffusion.beta_end,
        config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device_obj)

    print(f"[PNG] Model loaded successfully on {device}")
    return model, betas


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_png(image_input, image_size: int = 256) -> torch.Tensor:
    """
    Preprocess a PNG image for the original Fast-DDPM pipeline.

    Matches the original repo's data loading:
        1. Open as grayscale
        2. Resize to image_size × image_size
        3. Convert to float [0, 1]
        4. Apply rescaled transform: X = 2*X - 1 → [-1, 1]

    Args:
        image_input: file path (str), PIL Image, or numpy array.
        image_size: target spatial size (default 256).

    Returns:
        Tensor of shape [1, 1, image_size, image_size] in [-1, 1].
    """
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            # Take first channel or convert
            image_input = image_input[:, :, 0] if image_input.shape[2] >= 1 else image_input
        img = Image.fromarray(image_input.astype(np.uint8), mode="L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    # Resize
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # Convert to tensor [0, 1]
    arr = np.array(img).astype(np.float32) / 255.0

    # Apply rescaled transform: X = 2*X - 1
    arr = arr * 2.0 - 1.0

    # Shape: [1, 1, H, W]
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor


# ============================================================================
# Sampling / Inference
# ============================================================================

def _get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """Build beta schedule (matches runners/diffusion.py)."""
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


def _build_diffusion_seq(scheduler_type, timesteps, num_diffusion_timesteps=1000):
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


def run_inference_png(
    model: torch.nn.Module,
    betas: torch.Tensor,
    ld_tensor: torch.Tensor,
    device: str = "cuda",
    timesteps: int = 10,
    scheduler_type: str = "uniform",
    sample_type: str = "generalized",
    eta: float = 0.0,
) -> tuple:
    """
    Run PNG/original Fast-DDPM inference on a single LDCT image.

    This uses the sg_generalized_steps function which internally does:
        model(cat([x_t, condition], dim=1), t)
    where x_t is [B,1,H,W] and condition is [B,1,H,W], producing [B,2,H,W] input.

    Args:
        model: loaded UNet in eval mode.
        betas: beta schedule tensor on device.
        ld_tensor: preprocessed LDCT tensor [1, 1, H, W] in [-1, 1].
        device: "cuda" or "cpu".
        timesteps: number of diffusion sampling steps (default 10).
        scheduler_type: "uniform" or "non-uniform".
        sample_type: "generalized" (DDIM-like) or "ddpm_noisy".
        eta: DDIM eta parameter (0=deterministic).

    Returns:
        (denoised_tensor, elapsed_time) where:
            denoised_tensor: [1, 1, H, W] in [-1, 1]
            elapsed_time: float seconds
    """
    device_obj = torch.device(device)
    ld_tensor = ld_tensor.to(device_obj)

    _, _, h, w = ld_tensor.shape

    # Build timestep sequence
    num_diff_steps = betas.shape[0]
    seq = _build_diffusion_seq(scheduler_type, timesteps, num_diff_steps)

    # Start from Gaussian noise (same shape as target: 1 channel)
    x_noise = torch.randn(1, 1, h, w, device=device_obj)

    t_start = time.time()

    with torch.no_grad():
        if sample_type == "generalized":
            # sg_generalized_steps does: model(cat([xt, x_img], dim=1), t)
            # xt = [1,1,H,W], x_img = [1,1,H,W] → model input = [1,2,H,W]
            xs, _ = sg_generalized_steps(
                x_noise, ld_tensor, seq, model, betas, eta=eta
            )
            denoised = xs[-1]  # last state
        elif sample_type == "ddpm_noisy":
            skip = num_diff_steps // timesteps
            seq_ddpm = list(range(0, num_diff_steps, skip))
            xs, _ = sg_ddpm_steps(
                x_noise, ld_tensor, seq_ddpm, model, betas
            )
            denoised = xs[-1]
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

    elapsed = time.time() - t_start
    denoised = denoised.cpu()

    return denoised, elapsed


# ============================================================================
# Post-processing
# ============================================================================

def postprocess_to_numpy_01(tensor_minus1_1: torch.Tensor) -> np.ndarray:
    """
    Convert model output [-1, 1] tensor to [0, 1] numpy array.

    Matches inverse_data_transform with rescaled=True:
        X = (X + 1) / 2, clamped to [0, 1]

    Args:
        tensor_minus1_1: tensor [1, 1, H, W] or [H, W] in [-1, 1].

    Returns:
        numpy array [H, W] in [0, 1] float32.
    """
    arr = tensor_minus1_1.squeeze().cpu().float().numpy()
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    return arr.astype(np.float32)


def postprocess_to_uint8(tensor_minus1_1: torch.Tensor) -> np.ndarray:
    """Convert model output [-1,1] tensor to [0,255] uint8 numpy array."""
    arr_01 = postprocess_to_numpy_01(tensor_minus1_1)
    return (arr_01 * 255.0).round().astype(np.uint8)


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics_png(
    denoised_01: np.ndarray,
    fd_01: np.ndarray,
    ld_01: np.ndarray = None,
) -> dict:
    """
    Compute PSNR and SSIM for PNG mode.

    All inputs must be in [0, 1] range.

    Args:
        denoised_01: denoised image [H, W] in [0, 1].
        fd_01: full-dose ground truth [H, W] in [0, 1].
        ld_01: optional low-dose input [H, W] in [0, 1] for baseline metrics.

    Returns:
        dict with keys: denoised_psnr, denoised_ssim,
                        and optionally ld_psnr, ld_ssim, delta_psnr, delta_ssim.
    """
    result = {}

    dn_psnr = float(psnr_fn(fd_01, denoised_01, data_range=1.0))
    dn_ssim = float(ssim_fn(fd_01, denoised_01, data_range=1.0))
    result["denoised_psnr"] = dn_psnr
    result["denoised_ssim"] = dn_ssim

    if ld_01 is not None:
        ld_psnr = float(psnr_fn(fd_01, ld_01, data_range=1.0))
        ld_ssim = float(ssim_fn(fd_01, ld_01, data_range=1.0))
        result["ld_psnr"] = ld_psnr
        result["ld_ssim"] = ld_ssim
        result["delta_psnr"] = dn_psnr - ld_psnr
        result["delta_ssim"] = dn_ssim - ld_ssim

    return result
