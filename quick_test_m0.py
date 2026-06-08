"""
Quick inference test for M0 baseline
Loads 1 sample, runs denoising, saves images for visual inspection
"""
import os, sys, yaml, torch, numpy as np
sys.path.insert(0, '.')
from PIL import Image

from datasets.LDFDCT import LDFDCT
from models.diffusion import Model
from models.ema import EMAHelper
from functions.denoising import sg_generalized_steps

# === Config ===
CONFIG_PATH = 'configs/baseline_2d_npy_test_100k.yml'
CHECKPOINT = 'exp/logs/Fast-DDPM_experiments/best_ssim.pth'  # adjust if needed
OUTPUT_DIR = 'test_outputs/m0_verify'

os.makedirs(OUTPUT_DIR, exist_ok=True)

class Config:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

# Load config
cfg = Config(yaml.safe_load(open(CONFIG_PATH)))
print(f"[CONFIG] Loaded {CONFIG_PATH}")

# Load model
model = Model(cfg).cuda()
print(f"[MODEL] Created, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

# Load checkpoint
print(f"[CHECKPOINT] Looking for: {CHECKPOINT}")
if not os.path.exists(CHECKPOINT):
    # Try alternative paths
    for alt in ['exp/logs/Fast-DDPM_experiments/best_ssim.pth',
                'exp/logs/Fast-DDPM_experiments/ckpt.pth',
                'exp/logs/Fast-DDPM_experiments/final_100000.pth']:
        if os.path.exists(alt):
            CHECKPOINT = alt
            print(f"[CHECKPOINT] Found alternative: {alt}")
            break
    else:
        print(f"[ERROR] Checkpoint not found. Available:")
        os.system("find exp/ -name '*.pth' 2>/dev/null")
        sys.exit(1)

states = torch.load(CHECKPOINT, map_location='cuda', weights_only=False)
print(f"[CHECKPOINT] Loaded states (keys: {len(states)})")

# Load model weights (try multiple state formats)
if isinstance(states, list):
    model_state = states[0]
elif isinstance(states, dict):
    model_state = states.get('model', states)
else:
    model_state = states
    
model.load_state_dict(model_state)
print(f"[MODEL] Weights loaded")

# Use EMA if available
if isinstance(states, list) and len(states) >= 5:
    try:
        ema_helper = EMAHelper(mu=cfg.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(states[4])
        ema_helper.ema(model)
        print(f"[EMA] Applied")
    except Exception as e:
        print(f"[EMA] Skip: {e}")

model.eval()

# Load 1 test sample
dataset = LDFDCT(cfg.data.val_dataroot, cfg.data.image_size, split='val', config=cfg)
sample = dataset[0]
ld = sample['LD'].unsqueeze(0).cuda()  # (1, 1, 256, 256)
fd = sample['FD'].unsqueeze(0).cuda()
case_name = sample.get('case_name', 'sample_0')

print(f"\n[SAMPLE] Case: {case_name}")
print(f"  LD shape: {ld.shape}, range: [{ld.min():.3f}, {ld.max():.3f}]")
print(f"  FD shape: {fd.shape}, range: [{fd.min():.3f}, {fd.max():.3f}]")

# Setup diffusion parameters
betas = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end,
                       cfg.diffusion.num_diffusion_timesteps).cuda()

# Run sampling with 10 DDIM steps
T = cfg.diffusion.num_diffusion_timesteps  # 1000
n_steps = 10
skip = T // n_steps
seq = list(range(0, T, skip))[:n_steps]

with torch.no_grad():
    # Start from noise
    x_t = torch.randn_like(fd)
    
    # Sliding-window-style sampling for 2D
    xs, x0_preds = sg_generalized_steps(x_t, ld, seq, model, betas, eta=0.0)
    denoised = xs[-1].cuda()

print(f"\n[OUTPUT] denoised shape: {denoised.shape}, range: [{denoised.min():.3f}, {denoised.max():.3f}]")

# Compute PSNR
def psnr(a, b):
    a = (a.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
    b = (b.clamp(-1, 1) + 1) / 2
    mse = ((a - b) ** 2).mean()
    return (10 * torch.log10(1.0 / (mse + 1e-12))).item()

p_input = psnr(ld, fd)
p_output = psnr(denoised, fd)

print(f"\n[METRICS]")
print(f"  Input LDCT vs FDCT PSNR: {p_input:.4f} dB")
print(f"  Output     vs FDCT PSNR: {p_output:.4f} dB")
print(f"  Improvement:             {p_output - p_input:+.4f} dB")

# Save images
def save_img(t, name):
    img = t.squeeze().cpu().numpy()
    img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(OUTPUT_DIR, name))
    print(f"  Saved: {os.path.join(OUTPUT_DIR, name)}")

print(f"\n[SAVING IMAGES to {OUTPUT_DIR}]")
save_img(ld, '1_input_LDCT.png')
save_img(denoised, '2_output_denoised.png')
save_img(fd, '3_target_FDCT.png')

print(f"\n[DONE] View images in {OUTPUT_DIR}/")
