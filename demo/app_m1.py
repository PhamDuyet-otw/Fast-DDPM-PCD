"""
app_m1.py - Gradio demo for M1 (proposed 2.5D Fast-DDPM).
Port 7861 to coexist with pretrained app on 7860.
"""
import argparse
import os
import sys
import yaml
import numpy as np
import torch
import gradio as gr
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEMO_DIR)

from models.diffusion import Model
from functions.denoising import sg_generalized_steps

from preprocess import load_input_as_tensor, denormalize_to_uint8
from sliding_window import sliding_window_denoise


CONFIG_PATH = os.path.join(REPO_ROOT, "configs/proposed_2p5d_heads4_400k.yml")
CKPT_PATH = os.path.join(REPO_ROOT, "demo/checkpoints/m1_400k.pth")

TARGET_SIZE = 512
PATCH_SIZE = 256
STRIDE = 128
TIMESTEPS = 10
SCHEDULER = "uniform"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return dict2namespace(cfg_dict)


def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)


print("=" * 60)
print(" Initializing M1 LDCT Denoising Demo ".center(60, "="))
print("=" * 60)

print(f"[INIT] Device: {DEVICE}")
print(f"[INIT] Config:  {CONFIG_PATH}")
print(f"[INIT] Ckpt:    {CKPT_PATH}")

CONFIG = load_config(CONFIG_PATH)
print(f"[INIT] in_channels={CONFIG.model.in_channels}, "
      f"num_heads={CONFIG.model.num_heads}, "
      f"attn_resolutions={CONFIG.model.attn_resolutions}")

MODEL = Model(CONFIG).to(DEVICE)
MODEL.eval()
n_params = sum(p.numel() for p in MODEL.parameters()) / 1e6
print(f"[INIT] U-Net params: {n_params:.1f}M")

STATES = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
if isinstance(STATES, (list, tuple)) and len(STATES) >= 5:
    EMA_STATE = STATES[4]
    print(f"[INIT] Checkpoint format: list ({len(STATES)} items). "
          f"Loading EMA from index 4. Step={STATES[3]}, Epoch={STATES[2]}")
elif isinstance(STATES, dict):
    EMA_STATE = STATES.get("ema", STATES.get("state_dict", STATES))
    print(f"[INIT] Checkpoint format: dict.")
else:
    raise RuntimeError(f"Unknown checkpoint format: {type(STATES)}")

clean_state = {(k[7:] if k.startswith("module.") else k): v
               for k, v in EMA_STATE.items()}
missing, unexpected = MODEL.load_state_dict(clean_state, strict=False)
if missing:
    print(f"[INIT] [WARN] Missing keys: {len(missing)} (first 3: {missing[:3]})")
if unexpected:
    print(f"[INIT] [WARN] Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
if not missing and not unexpected:
    print(f"[INIT] [OK] EMA weights loaded perfectly")

BETAS = torch.from_numpy(get_beta_schedule(
    CONFIG.diffusion.beta_start,
    CONFIG.diffusion.beta_end,
    CONFIG.diffusion.num_diffusion_timesteps,
)).float().to(DEVICE)
NUM_T = BETAS.shape[0]

if SCHEDULER == "uniform":
    SEQ = np.linspace(0, NUM_T - 1, TIMESTEPS).astype(int).tolist()
else:
    SEQ = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]

print(f"[INIT] Sampling sequence ({SCHEDULER}, {TIMESTEPS} steps): {SEQ}")
print("=" * 60)
print(" Model ready. Launching Gradio interface... ".center(60, "="))
print("=" * 60)


def denoise_m1(filepath, progress=gr.Progress(track_tqdm=False)):
    if filepath is None:
        return None, None, "Please upload a file first."

    try:
        cond_tensor, info = load_input_as_tensor(filepath, target_size=TARGET_SIZE)
    except Exception as e:
        return None, None, f"Error reading file: {e}"

    status = ""

    input_display_arr = cond_tensor[0, 0].numpy()
    input_display = Image.fromarray(denormalize_to_uint8(input_display_arr))

    def progress_cb(idx, total):
        progress(idx / total, desc=f"Denoising patch {idx}/{total}")

    output_tensor, total_patches = sliding_window_denoise(
        input_tensor=cond_tensor,
        model=MODEL,
        betas=BETAS,
        seq=SEQ,
        sg_sample_fn=sg_generalized_steps,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        eta=0.0,
        seed=SEED,
        device=DEVICE,
        progress_callback=progress_cb,
    )

    output_arr = output_tensor[0, 0].cpu().numpy()
    output_display = Image.fromarray(denormalize_to_uint8(output_arr))

    return input_display, output_display, status


def display_input_only(filepath):
    if filepath is None:
        return None, ""
    try:
        cond_tensor, info = load_input_as_tensor(filepath, target_size=TARGET_SIZE)
    except Exception as e:
        return None, f"Error: {e}"
    input_arr = cond_tensor[0, 0].numpy()
    input_display = Image.fromarray(denormalize_to_uint8(input_arr))
    info_text = ""
    return input_display, info_text


with gr.Blocks(title="Low-dose CT Denoising (M1)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Low-dose CT Denoising

        """
    )

    with gr.Row():
        upload_file = gr.File(
            label="Upload (.npy / .dcm / .png)",
            file_types=[".npy", ".dcm", ".dicom", ".png", ".jpg", ".jpeg"],
            type="filepath",
        )

    status_box = gr.Markdown("")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input",
                type="pil",
                interactive=False,
                height=400,
            )
        with gr.Column():
            output_image = gr.Image(
                label="Output",
                type="pil",
                interactive=False,
                height=400,
            )

    with gr.Row():
        run_btn = gr.Button("Run Denoising", variant="primary", size="lg")
        clear_btn = gr.Button("Reset", size="lg")

    upload_file.change(
        fn=display_input_only,
        inputs=upload_file,
        outputs=[input_image, status_box],
    )

    run_btn.click(
        fn=denoise_m1,
        inputs=upload_file,
        outputs=[input_image, output_image, status_box],
    )

    clear_btn.click(
        fn=lambda: (None, None, None, ""),
        outputs=[upload_file, input_image, output_image, status_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
    )
