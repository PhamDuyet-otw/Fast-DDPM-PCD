#!/usr/bin/env python3
"""
Gradio Thesis Demo App — Low-Dose Lung CT Image Denoising Using Diffusion Models

Two separate tabs, two separate model pipelines:
    Tab 1: PNG Original Fast-DDPM Mode     (in_channels=2, 2D, PNG input)
    Tab 2: HU-NPY / DICOM 2.5D Mode       (in_channels=4, 2.5D, NPY/DICOM input)

Usage:
    python demo/app_gradio.py [--share] [--port 7860] [--device cuda]

    # With default paths pre-filled (optional):
    python demo/app_gradio.py \
        --png_config configs/ldfd_linear.yml \
        --v25d_config configs/ldfd_v3_2p5d_5090_full.yml

Research prototype only. Not for clinical diagnosis.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import demo modules
from demo.inference_png_original import (
    load_config_png,
    load_model_png,
    preprocess_png,
    run_inference_png,
    postprocess_to_numpy_01,
    postprocess_to_uint8,
    compute_metrics_png,
    scan_png_sample_folder,
)
from demo.inference_2p5d_hu import (
    load_config_2p5d,
    load_model_2p5d,
    load_hu_npy,
    build_2p5d_condition,
    run_inference_2p5d,
    load_manifest_samples,
    build_25d_triplets,
    compute_metrics_2p5d,
    postprocess_to_numpy_01 as postprocess_2p5d_01,
)
from demo.image_utils import to_display_image, arr_01_to_display, create_comparison_figure
from demo.dicom_utils import (
    HAS_PYDICOM,
    extract_dicom_zip,
    load_dicom_series,
    dicom_series_to_triplets,
    normalize_hu_to_minus1_1,
)

import gradio as gr


# ============================================================================
# CLI argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast-DDPM Thesis Demo — Gradio App"
    )
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link (for Colab)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Server port (default: 7860)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Default device")
    # Default config paths (optional, editable in UI)
    parser.add_argument("--png_config", type=str,
                        default="configs/ldfd_linear.yml",
                        help="Default PNG mode config path")
    parser.add_argument("--png_ckpt", type=str,
                        default="D:/B3/Thesis/Fast-DDPM/pretrained_models/ckpt_LDFDCT.pth",
                        help="Default PNG mode checkpoint path")
    parser.add_argument("--png_sample_folder", type=str,
                        default="D:/B3/Thesis/Fast-DDPM/Fast-DDPM/data/LD_FD_CT_test",
                        help="Default PNG sample folder path")
    parser.add_argument("--v25d_config", type=str,
                        default="configs/ldfd_v3_2p5d_5090_full.yml",
                        help="Default 2.5D mode config path")
    parser.add_argument("--v25d_ckpt", type=str, default="",
                        help="Default 2.5D mode checkpoint path")
    return parser.parse_args()


# ============================================================================
# Tab 1: PNG Original Fast-DDPM Mode
# ============================================================================

def load_png_model_fn(config_path, ckpt_path, device, state):
    """Load the PNG/original Fast-DDPM model."""
    try:
        if not config_path.strip():
            return state, "❌ Please provide the PNG model config path."
        if not ckpt_path.strip():
            return state, "❌ Please provide the PNG model checkpoint path."

        # Resolve relative paths from project root
        if not os.path.isabs(config_path):
            config_path = os.path.join(_PROJECT_ROOT, config_path)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(_PROJECT_ROOT, ckpt_path)

        config = load_config_png(config_path)
        model, betas = load_model_png(config, ckpt_path, device)

        state = {
            "model": model,
            "betas": betas,
            "config": config,
            "device": device,
            "ckpt_path": ckpt_path,
        }
        return state, (
            f"✅ PNG model loaded successfully!\n"
            f"   Config: {config_path}\n"
            f"   Checkpoint: {ckpt_path}\n"
            f"   Device: {device}\n"
            f"   in_channels: {config.model.in_channels}\n"
            f"   image_size: {config.data.image_size}"
        )
    except Exception as e:
        return state, f"❌ Error loading PNG model:\n{str(e)}\n{traceback.format_exc()}"


def load_png_samples_fn(folder_path, state):
    """Scan a PNG sample folder and detect LD/FD pairs."""
    try:
        if not folder_path.strip():
            return state, [], "❌ Please provide a PNG sample folder path."

        if not os.path.isabs(folder_path):
            folder_path = os.path.join(_PROJECT_ROOT, folder_path)

        samples = scan_png_sample_folder(folder_path)

        if not samples:
            return state, [], (
                f"⚠️ No LD PNG files found in: {folder_path}\n"
                "Expected naming: {{patient}}_{{slice}}_ld.png / _fd.png"
            )

        if state is None:
            state = {}
        state["png_samples"] = samples

        choices = [s["display_name"] for s in samples]
        n_paired = sum(1 for s in samples if s["fd_path"] is not None)
        n_patients = len(set(s["patient_id"] for s in samples))

        return state, choices, (
            f"✅ Found {len(samples)} LD images, {n_paired} with FD pairs\n"
            f"   {n_patients} patients in: {folder_path}\n"
            f"   Select a sample from the dropdown to run inference."
        )
    except Exception as e:
        return state, [], f"❌ Error scanning folder:\n{str(e)}\n{traceback.format_exc()}"


def _run_png_inference_core(ld_input, fd_input, timesteps, scheduler_type,
                            sample_type, state, sample_label=""):
    """
    Core PNG denoising logic shared by manual upload and sample-folder modes.

    ld_input/fd_input: PIL Image, file path str, or None.
    """
    if state is None or "model" not in state:
        return None, None, None, "❌ Please load the PNG model first."
    if ld_input is None:
        return None, None, None, "❌ No LDCT input available."

    model = state["model"]
    betas = state["betas"]
    config = state["config"]
    device = state["device"]
    image_size = config.data.image_size

    # Preprocess LDCT
    ld_tensor = preprocess_png(ld_input, image_size)

    # Run inference
    denoised_tensor, elapsed = run_inference_png(
        model, betas, ld_tensor, device,
        timesteps=int(timesteps),
        scheduler_type=scheduler_type,
        sample_type=sample_type,
    )

    # Post-process
    denoised_01 = postprocess_to_numpy_01(denoised_tensor)
    ld_01 = postprocess_to_numpy_01(ld_tensor)
    denoised_img = arr_01_to_display(denoised_01)
    ld_display = arr_01_to_display(ld_01)

    # Metrics
    metrics_text = ""
    if sample_label:
        metrics_text += f"🔬 Sample: {sample_label}\n"
    metrics_text += f"⏱ Runtime: {elapsed:.2f}s\n"
    metrics_text += f"📐 Timesteps: {timesteps}, Scheduler: {scheduler_type}\n"

    fd_display = None
    if fd_input is not None:
        fd_tensor = preprocess_png(fd_input, image_size)
        fd_01 = postprocess_to_numpy_01(fd_tensor)
        fd_display = arr_01_to_display(fd_01)

        metrics = compute_metrics_png(denoised_01, fd_01, ld_01)
        metrics_text += (
            f"\n📊 Metrics (vs Full-Dose):\n"
            f"   Denoised PSNR: {metrics['denoised_psnr']:.4f} dB\n"
            f"   Denoised SSIM: {metrics['denoised_ssim']:.4f}\n"
        )
        if "ld_psnr" in metrics:
            metrics_text += (
                f"\n📊 LD Baseline:\n"
                f"   LD PSNR: {metrics['ld_psnr']:.4f} dB\n"
                f"   LD SSIM: {metrics['ld_ssim']:.4f}\n"
                f"\n📈 Improvement:\n"
                f"   ΔPSNR: {metrics['delta_psnr']:+.4f} dB\n"
                f"   ΔSSIM: {metrics['delta_ssim']:+.4f}"
            )
    else:
        metrics_text += (
            "\nℹ️ No FDCT ground truth provided.\n"
            "   Upload an FDCT PNG to compute PSNR/SSIM."
        )

    return ld_display, denoised_img, fd_display, metrics_text


def run_png_denoising_fn(ld_image, fd_image, timesteps, scheduler_type,
                         sample_type, state):
    """Run PNG mode denoising from manual image upload."""
    try:
        return _run_png_inference_core(
            ld_image, fd_image, timesteps, scheduler_type,
            sample_type, state
        )
    except Exception as e:
        return None, None, None, f"❌ Error:\n{str(e)}\n{traceback.format_exc()}"


def run_png_from_sample_fn(sample_name, timesteps, scheduler_type,
                           sample_type, state):
    """Run PNG denoising from a selected sample in the folder scanner."""
    try:
        if state is None or "png_samples" not in state:
            return None, None, None, "❌ Please load PNG samples from a folder first."
        if not sample_name:
            return None, None, None, "❌ Please select a sample from the dropdown."

        sample = None
        for s in state["png_samples"]:
            if s["display_name"] == sample_name:
                sample = s
                break
        if sample is None:
            return None, None, None, f"❌ Sample '{sample_name}' not found."

        return _run_png_inference_core(
            sample["ld_path"],
            sample["fd_path"],  # May be None if no FD pair
            timesteps, scheduler_type, sample_type, state,
            sample_label=sample["display_name"],
        )
    except Exception as e:
        return None, None, None, f"❌ Error:\n{str(e)}\n{traceback.format_exc()}"


# ============================================================================
# Tab 2: HU-NPY / DICOM 2.5D Mode
# ============================================================================

def load_2p5d_model_fn(config_path, ckpt_path, device, state):
    """Load the 2.5D HU-NPY model."""
    try:
        if not config_path.strip():
            return state, "❌ Please provide the 2.5D model config path."
        if not ckpt_path.strip():
            return state, "❌ Please provide the 2.5D model checkpoint path."

        if not os.path.isabs(config_path):
            config_path = os.path.join(_PROJECT_ROOT, config_path)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(_PROJECT_ROOT, ckpt_path)

        config = load_config_2p5d(config_path)
        model, betas = load_model_2p5d(config, ckpt_path, device)

        state = {
            "model": model,
            "betas": betas,
            "config": config,
            "device": device,
            "ckpt_path": ckpt_path,
        }
        return state, (
            f"✅ 2.5D model loaded successfully!\n"
            f"   Config: {config_path}\n"
            f"   Checkpoint: {ckpt_path}\n"
            f"   Device: {device}\n"
            f"   in_channels: {config.model.in_channels}\n"
            f"   image_size: {config.data.image_size}"
        )
    except Exception as e:
        return state, f"❌ Error loading 2.5D model:\n{str(e)}\n{traceback.format_exc()}"


def load_manifest_fn(manifest_path, data_root, state):
    """Load manifest and build 2.5D triplets."""
    try:
        if not manifest_path.strip():
            return state, [], "❌ Please provide the manifest CSV path."

        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(_PROJECT_ROOT, manifest_path)
        data_root_abs = None
        if data_root.strip():
            data_root_abs = data_root if os.path.isabs(data_root) else os.path.join(_PROJECT_ROOT, data_root)

        samples = load_manifest_samples(manifest_path, data_root=data_root_abs)
        triplets = build_25d_triplets(samples)

        if len(triplets) == 0:
            return state, [], (
                "⚠️ No valid 2.5D triplets found.\n"
                "Check that the manifest paths point to existing NPY files\n"
                "and that each patient has at least 3 consecutive slices."
            )

        # Store triplets in state
        if state is None:
            state = {}
        state["triplets"] = triplets

        # Build dropdown choices
        choices = [t["display_name"] for t in triplets]

        return state, choices, (
            f"✅ Loaded {len(samples)} samples, built {len(triplets)} valid 2.5D triplets.\n"
            f"   Select a sample from the dropdown to run inference."
        )
    except Exception as e:
        return state, [], f"❌ Error loading manifest:\n{str(e)}\n{traceback.format_exc()}"


def run_2p5d_npy_denoising_fn(
    sample_name, timesteps, scheduler_type, overlap, state
):
    """Run 2.5D NPY denoising on a selected manifest sample."""
    try:
        if state is None or "model" not in state:
            return None, None, None, "❌ Please load the 2.5D model first."
        if "triplets" not in state or not state["triplets"]:
            return None, None, None, "❌ Please load manifest samples first."
        if not sample_name:
            return None, None, None, "❌ Please select a sample from the dropdown."

        model = state["model"]
        betas = state["betas"]
        config = state["config"]
        device = state["device"]

        # Find the selected triplet
        triplet = None
        for t in state["triplets"]:
            if t["display_name"] == sample_name:
                triplet = t
                break
        if triplet is None:
            return None, None, None, f"❌ Sample '{sample_name}' not found."

        # HU parameters from config
        hu_min = getattr(config.data, "hu_window_min", -1000)
        hu_max = getattr(config.data, "hu_window_max", 400)

        # Load 2.5D condition
        ld_triplet_arr = build_2p5d_condition(triplet["ld_paths"], hu_min, hu_max)
        ld_center = ld_triplet_arr[1]  # Center slice for display

        # Run inference
        denoised, elapsed = run_inference_2p5d(
            model, betas, ld_triplet_arr, device,
            timesteps=int(timesteps),
            scheduler_type=scheduler_type,
            overlap=int(overlap),
        )

        # Display images
        ld_display = to_display_image(ld_center)
        dn_display = to_display_image(denoised)

        # Metrics
        metrics_text = (
            f"🔬 Sample: {triplet['display_name']}\n"
            f"⏱ Runtime: {elapsed:.2f}s\n"
            f"📐 Timesteps: {timesteps}, Scheduler: {scheduler_type}, Overlap: {overlap}\n"
        )

        fd_display = None
        if triplet["fd_path"] and os.path.isfile(triplet["fd_path"]):
            fd_arr = load_hu_npy(triplet["fd_path"], hu_min, hu_max)
            fd_display = to_display_image(fd_arr)

            metrics = compute_metrics_2p5d(denoised, fd_arr, ld_center)
            metrics_text += (
                f"\n📊 Metrics (vs Full-Dose):\n"
                f"   Denoised PSNR: {metrics['denoised_psnr']:.4f} dB\n"
                f"   Denoised SSIM: {metrics['denoised_ssim']:.4f}\n"
            )
            if "ld_psnr" in metrics:
                metrics_text += (
                    f"\n📊 LD Baseline:\n"
                    f"   LD PSNR: {metrics['ld_psnr']:.4f} dB\n"
                    f"   LD SSIM: {metrics['ld_ssim']:.4f}\n"
                    f"\n📈 Improvement:\n"
                    f"   ΔPSNR: {metrics['delta_psnr']:+.4f} dB\n"
                    f"   ΔSSIM: {metrics['delta_ssim']:+.4f}"
                )
        else:
            metrics_text += (
                "\nℹ️ No FDCT ground truth available for this sample.\n"
                "   PSNR/SSIM cannot be computed."
            )

        return ld_display, dn_display, fd_display, metrics_text

    except Exception as e:
        return None, None, None, f"❌ Error:\n{str(e)}\n{traceback.format_exc()}"


def run_2p5d_dicom_denoising_fn(
    dicom_zip, slice_index, timesteps, scheduler_type, overlap, state
):
    """Run 2.5D denoising on a DICOM series."""
    try:
        if state is None or "model" not in state:
            return None, None, None, "❌ Please load the 2.5D model first."
        if dicom_zip is None:
            return None, None, None, "❌ Please upload a DICOM series ZIP file."
        if not HAS_PYDICOM:
            return None, None, None, (
                "❌ pydicom is not installed.\n"
                "Install with: pip install pydicom"
            )

        model = state["model"]
        betas = state["betas"]
        config = state["config"]
        device = state["device"]

        hu_min = getattr(config.data, "hu_window_min", -1000)
        hu_max = getattr(config.data, "hu_window_max", 400)

        # Extract DICOM ZIP
        zip_path = dicom_zip.name if hasattr(dicom_zip, "name") else str(dicom_zip)
        dicom_dir = extract_dicom_zip(zip_path)
        dicom_slices = load_dicom_series(dicom_dir)

        # Build triplets
        triplets = dicom_series_to_triplets(dicom_slices, hu_min, hu_max)

        if not triplets:
            return None, None, None, "❌ Not enough slices for 2.5D inference (need ≥ 3)."

        # Select slice to process
        slice_idx = int(slice_index) if slice_index else len(triplets) // 2
        slice_idx = max(0, min(slice_idx, len(triplets) - 1))
        selected = triplets[slice_idx]

        ld_triplet = selected["ld_triplet"]
        ld_center = ld_triplet[1]

        # Run inference
        denoised, elapsed = run_inference_2p5d(
            model, betas, ld_triplet, device,
            timesteps=int(timesteps),
            scheduler_type=scheduler_type,
            overlap=int(overlap),
        )

        ld_display = to_display_image(ld_center)
        dn_display = to_display_image(denoised)

        metrics_text = (
            f"🔬 DICOM slice: {selected['display_name']} "
            f"(index {slice_idx + 1}/{len(triplets)})\n"
            f"⏱ Runtime: {elapsed:.2f}s\n"
            f"📐 Timesteps: {timesteps}, Scheduler: {scheduler_type}\n"
            f"📦 Total DICOM slices: {len(dicom_slices)}\n"
            f"📦 Valid 2.5D triplets: {len(triplets)}\n"
            f"\nℹ️ DICOM mode has no FDCT ground truth.\n"
            f"   PSNR/SSIM cannot be computed."
        )

        # Store DICOM state for potential export
        state["dicom_slices"] = dicom_slices
        state["dicom_triplets"] = triplets

        return ld_display, dn_display, None, metrics_text

    except Exception as e:
        return None, None, None, f"❌ Error:\n{str(e)}\n{traceback.format_exc()}"


# ============================================================================
# Gradio UI
# ============================================================================

def build_app(args):
    """Build the Gradio Blocks app."""

    with gr.Blocks(
        title="Fast-DDPM CT Denoising — Thesis Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    ) as app:

        # --- Header ---
        gr.Markdown(
            """
            # 🫁 Low-Dose Lung CT Image Denoising Using Diffusion Models
            ### Fast-DDPM Thesis Demo — Inference Only

            This demo supports **two separate inference modes** with **two separate model pipelines**:

            | Tab | Mode | Input | Model |
            |-----|------|-------|-------|
            | 1 | PNG Original Fast-DDPM | 2D PNG | in_channels=2 |
            | 2 | HU-NPY / DICOM 2.5D | raw-HU NPY or DICOM | in_channels=4 |

            > ⚠️ **Research prototype only. Not for clinical diagnosis.**
            """
        )

        # ================================================================
        # TAB 1: PNG Original Fast-DDPM Mode
        # ================================================================
        with gr.Tab("🖼️ PNG Original Fast-DDPM Mode"):
            gr.Markdown(
                """
                ### PNG Original Fast-DDPM Mode
                Uses the original-style PNG pipeline with `in_channels=2` (1 LD condition + 1 noisy target).

                > ⚠️ This mode is for **visualization and original-repo-style comparison**, not for clinical DICOM processing.
                > Do NOT use a 2.5D checkpoint here.
                """
            )

            png_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📂 Model Configuration")
                    png_config_input = gr.Textbox(
                        label="PNG Config Path",
                        value=args.png_config,
                        placeholder="configs/ldfd_linear.yml",
                        info="YAML config for the PNG/original model",
                    )
                    png_ckpt_input = gr.Textbox(
                        label="PNG Checkpoint Path",
                        value=args.png_ckpt,
                        placeholder="D:/B3/Thesis/Fast-DDPM/pretrained_models/ckpt_LDFDCT.pth",
                        info="Trained checkpoint for the PNG/original model",
                    )
                    png_device = gr.Dropdown(
                        label="Device",
                        choices=["cuda", "cpu"],
                        value=args.device,
                    )
                    png_load_btn = gr.Button(
                        "🔄 Load PNG Model", variant="primary"
                    )
                    png_load_status = gr.Textbox(
                        label="Model Status",
                        lines=6,
                        interactive=False,
                        value="Model not loaded. Please provide config and checkpoint paths.",
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### 📁 PNG Sample Folder")
                    png_sample_folder_input = gr.Textbox(
                        label="PNG Sample Folder Path",
                        value=args.png_sample_folder,
                        placeholder="D:/B3/Thesis/Fast-DDPM/Fast-DDPM/data/LD_FD_CT_test",
                        info="Folder with LD/FD PNG pairs (scanned recursively)",
                    )
                    png_load_samples_btn = gr.Button(
                        "📂 Load PNG Samples"
                    )
                    png_sample_dropdown = gr.Dropdown(
                        label="Select Sample",
                        choices=[],
                        interactive=True,
                        info="Auto-detected LD/FD pairs from sample folder",
                    )
                    png_sample_status = gr.Textbox(
                        label="Sample Status",
                        lines=3,
                        interactive=False,
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### ⚙️ Sampling Parameters")
                    png_timesteps = gr.Slider(
                        label="Timesteps",
                        minimum=2, maximum=100, value=10, step=1,
                    )
                    png_scheduler = gr.Dropdown(
                        label="Scheduler Type",
                        choices=["uniform", "non-uniform"],
                        value="uniform",
                    )
                    png_sample_type = gr.Dropdown(
                        label="Sample Type",
                        choices=["generalized", "ddpm_noisy"],
                        value="generalized",
                    )
                    with gr.Row():
                        png_run_sample_btn = gr.Button(
                            "▶️ Run on Selected Sample", variant="primary"
                        )

                with gr.Column(scale=1):
                    gr.Markdown("#### 🖼️ Manual Upload (alternative)")
                    png_ld_upload = gr.Image(
                        label="Upload LDCT PNG",
                        type="pil",
                        image_mode="L",
                    )
                    png_fd_upload = gr.Image(
                        label="Upload FDCT PNG (optional, for metrics)",
                        type="pil",
                        image_mode="L",
                    )
                    png_run_upload_btn = gr.Button(
                        "▶️ Run on Uploaded Image", variant="secondary"
                    )

            gr.Markdown("#### 📊 Results")
            with gr.Row():
                png_ld_output = gr.Image(label="LDCT Input", type="pil")
                png_dn_output = gr.Image(label="Denoised Output", type="pil")
                png_fd_output = gr.Image(label="FDCT Ground Truth", type="pil")

            png_metrics_output = gr.Textbox(
                label="Metrics & Runtime",
                lines=12,
                interactive=False,
            )

            # --- Button callbacks ---
            png_load_btn.click(
                fn=load_png_model_fn,
                inputs=[png_config_input, png_ckpt_input, png_device, png_state],
                outputs=[png_state, png_load_status],
            )

            png_load_samples_btn.click(
                fn=load_png_samples_fn,
                inputs=[png_sample_folder_input, png_state],
                outputs=[png_state, png_sample_dropdown, png_sample_status],
            )

            png_run_sample_btn.click(
                fn=run_png_from_sample_fn,
                inputs=[
                    png_sample_dropdown, png_timesteps,
                    png_scheduler, png_sample_type, png_state,
                ],
                outputs=[png_ld_output, png_dn_output, png_fd_output, png_metrics_output],
            )

            png_run_upload_btn.click(
                fn=run_png_denoising_fn,
                inputs=[
                    png_ld_upload, png_fd_upload, png_timesteps,
                    png_scheduler, png_sample_type, png_state,
                ],
                outputs=[png_ld_output, png_dn_output, png_fd_output, png_metrics_output],
            )

        # ================================================================
        # TAB 2: HU-NPY / DICOM Clinical-style 2.5D Mode
        # ================================================================
        with gr.Tab("🩻 HU-NPY / DICOM 2.5D Mode"):
            gr.Markdown(
                """
                ### HU-NPY / DICOM Clinical-style 2.5D Mode
                Uses Dataset V3 raw-HU NPY pipeline with `in_channels=4`
                (3 LD condition channels + 1 noisy target).

                **2.5D condition**: LD[z-1], LD[z], LD[z+1] → target FD[z]

                > ⚠️ This mode uses the Dataset V3 raw-HU / DICOM-style 2.5D pipeline.
                > It is a **research prototype** and is **NOT for clinical diagnosis**.
                > Do NOT use a PNG/original checkpoint here.
                """
            )

            v25d_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📂 Model Configuration")
                    v25d_config_input = gr.Textbox(
                        label="2.5D Config Path",
                        value=args.v25d_config,
                        placeholder="configs/ldfd_v3_2p5d_5090_full.yml",
                        info="YAML config for the 2.5D HU-NPY model",
                    )
                    v25d_ckpt_input = gr.Textbox(
                        label="2.5D Checkpoint Path",
                        value=args.v25d_ckpt,
                        placeholder="path/to/2p5d_model/best_psnr.pth",
                        info="Trained checkpoint for the 2.5D model",
                    )
                    v25d_device = gr.Dropdown(
                        label="Device",
                        choices=["cuda", "cpu"],
                        value=args.device,
                    )
                    v25d_load_btn = gr.Button(
                        "🔄 Load 2.5D Model", variant="primary"
                    )
                    v25d_load_status = gr.Textbox(
                        label="Model Status",
                        lines=6,
                        interactive=False,
                        value="Model not loaded. Please provide config and checkpoint paths.",
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### 📁 Dataset V3 Manifest")
                    v25d_manifest_input = gr.Textbox(
                        label="Manifest CSV Path",
                        placeholder="path/to/test_v3.csv",
                        info="Dataset V3 manifest with LD/FD NPY paths",
                    )
                    v25d_dataroot_input = gr.Textbox(
                        label="Data Root (optional)",
                        placeholder="path/to/FastDDPM_Data",
                        info="Root directory for resolving relative paths in manifest",
                    )
                    v25d_load_manifest_btn = gr.Button(
                        "📂 Load Manifest Samples"
                    )
                    v25d_manifest_status = gr.Textbox(
                        label="Manifest Status",
                        lines=4,
                        interactive=False,
                    )

            # --- NPY Mode ---
            with gr.Accordion("🔬 NPY Sample Inference", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        v25d_sample_dropdown = gr.Dropdown(
                            label="Select Sample",
                            choices=[],
                            interactive=True,
                        )
                        v25d_timesteps = gr.Slider(
                            label="Timesteps",
                            minimum=2, maximum=100, value=10, step=1,
                        )
                        v25d_scheduler = gr.Dropdown(
                            label="Scheduler Type",
                            choices=["uniform", "non-uniform"],
                            value="uniform",
                        )
                        v25d_overlap = gr.Slider(
                            label="Sliding Window Overlap (px)",
                            minimum=0, maximum=224, value=128, step=16,
                        )
                        v25d_run_npy_btn = gr.Button(
                            "▶️ Run 2.5D NPY Denoising", variant="primary"
                        )

                gr.Markdown("#### 📊 Results")
                with gr.Row():
                    v25d_ld_output = gr.Image(label="LDCT (center slice)", type="pil")
                    v25d_dn_output = gr.Image(label="Denoised", type="pil")
                    v25d_fd_output = gr.Image(label="FDCT Ground Truth", type="pil")

                v25d_metrics_output = gr.Textbox(
                    label="Metrics & Runtime",
                    lines=14,
                    interactive=False,
                )

            # --- DICOM Mode ---
            with gr.Accordion("🏥 DICOM Series Inference", open=False):
                gr.Markdown(
                    """
                    Upload a ZIP file containing a DICOM CT series.
                    The app will convert DICOM to HU arrays, build 2.5D triplets, and run inference.

                    > ℹ️ Requires `pydicom`. Install with: `pip install pydicom`
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        v25d_dicom_upload = gr.File(
                            label="Upload DICOM Series ZIP",
                            file_types=[".zip"],
                        )
                        v25d_dicom_slice_idx = gr.Number(
                            label="Center Slice Index (0-based, leave empty for middle)",
                            value=None,
                            precision=0,
                        )
                        v25d_run_dicom_btn = gr.Button(
                            "▶️ Run DICOM 2.5D Denoising", variant="secondary"
                        )

                with gr.Row():
                    v25d_dicom_ld = gr.Image(label="LDCT (center slice)", type="pil")
                    v25d_dicom_dn = gr.Image(label="Denoised", type="pil")
                    v25d_dicom_fd = gr.Image(label="FDCT (N/A for DICOM)", type="pil")

                v25d_dicom_metrics = gr.Textbox(
                    label="DICOM Results",
                    lines=10,
                    interactive=False,
                )

            # --- Button callbacks ---
            v25d_load_btn.click(
                fn=load_2p5d_model_fn,
                inputs=[v25d_config_input, v25d_ckpt_input, v25d_device, v25d_state],
                outputs=[v25d_state, v25d_load_status],
            )

            v25d_load_manifest_btn.click(
                fn=load_manifest_fn,
                inputs=[v25d_manifest_input, v25d_dataroot_input, v25d_state],
                outputs=[v25d_state, v25d_sample_dropdown, v25d_manifest_status],
            )

            v25d_run_npy_btn.click(
                fn=run_2p5d_npy_denoising_fn,
                inputs=[
                    v25d_sample_dropdown, v25d_timesteps,
                    v25d_scheduler, v25d_overlap, v25d_state,
                ],
                outputs=[v25d_ld_output, v25d_dn_output, v25d_fd_output, v25d_metrics_output],
            )

            v25d_run_dicom_btn.click(
                fn=run_2p5d_dicom_denoising_fn,
                inputs=[
                    v25d_dicom_upload, v25d_dicom_slice_idx,
                    v25d_timesteps, v25d_scheduler, v25d_overlap, v25d_state,
                ],
                outputs=[v25d_dicom_ld, v25d_dicom_dn, v25d_dicom_fd, v25d_dicom_metrics],
            )

        # --- Footer ---
        gr.Markdown(
            """
            ---
            **Fast-DDPM CT Denoising** — Final-year Data Science Thesis Demo

            📌 PNG mode uses the original-style Fast-DDPM model (in_channels=2).
            📌 HU-NPY/DICOM mode uses the Dataset V3 2.5D model (in_channels=4).
            📌 These are **two separate pipelines** with **two separate checkpoints**.

            > ⚠️ Research prototype only. Not for clinical diagnosis.
            """
        )

    return app


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Check CUDA availability
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, defaulting to CPU")
        args.device = "cpu"

    print("=" * 60)
    print("Fast-DDPM CT Denoising — Thesis Demo")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"PNG config default: {args.png_config}")
    print(f"2.5D config default: {args.v25d_config}")
    print(f"Share: {args.share}")
    print(f"Port: {args.port}")
    print("=" * 60)

    app = build_app(args)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
