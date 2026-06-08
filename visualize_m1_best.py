"""
Visualize M1 best checkpoint (ckpt_400000)
Save 5 samples from test set: input / output / target
"""
import os, sys, yaml, torch, numpy as np
sys.path.insert(0, '.')
from PIL import Image

from datasets.LDFDCT import LDFDCT
from models.diffusion import Model
from models.ema import EMAHelper
from functions.denoising import sg_generalized_steps

CONFIG_PATH = 'configs/proposed_2p5d_heads4_400k.yml'
CHECKPOINT = 'exp/logs/Fast-DDPM_experiments/ckpt_400000.pth'  # Best PSNR
OUTPUT_DIR = 'test_outputs/m1_visualize_best'
N_SAMPLES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

class Config:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

cfg = Config(yaml.safe_load(open(CONFIG_PATH)))
print(f"[CONFIG] {CONFIG_PATH}")
print(f"[CHECKPOINT] {CHECKPOINT}")
print(f"[OUTPUT] {OUTPUT_DIR}")

# Load model
model = Model(cfg).cuda()
states = torch.load(CHECKPOINT, map_location='cuda', weights_only=False)
model_state = states[0] if isinstance(states, list) else states
model.load_state_dict(model_state)

# Apply EMA
if isinstance(states, list) and len(states) >= 5:
    ema_helper = EMAHelper(mu=cfg.model.ema_rate)
    ema_helper.register(model)
    ema_helper.load_state_dict(states[4])
    ema_helper.ema(model)
    print("[EMA] Applied")

model.eval()

# Test set
dataset = LDFDCT(cfg.data.test_dataroot, cfg.data.image_size, split='test', config=cfg)
print(f"[DATASET] {len(dataset)} test samples\n")

# Diffusion
betas = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end,
                       cfg.diffusion.num_diffusion_timesteps).cuda()
T = cfg.diffusion.num_diffusion_timesteps
seq = list(range(0, T, T // 10))[:10]

def save_img(t, name):
    img = t.squeeze().cpu().numpy()
    if img.ndim == 3:  # multi-channel, take center
        img = img[img.shape[0] // 2]
    img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    path = os.path.join(OUTPUT_DIR, name)
    Image.fromarray(img).save(path)
    return path

# Process N samples
for i in range(N_SAMPLES):
    sample = dataset[i]
    ld = sample['LD'].unsqueeze(0).cuda()
    fd = sample['FD'].unsqueeze(0).cuda()
    
    with torch.no_grad():
        x_t = torch.randn_like(fd)
        xs, _ = sg_generalized_steps(x_t, ld, seq, model, betas, eta=0.0)
        denoised = xs[-1].cuda()
    
    # Save 3 images per sample
    ld_path = save_img(ld, f'sample_{i:02d}_1_input_LDCT.png')
    out_path = save_img(denoised, f'sample_{i:02d}_2_output_M1.png')
    fd_path = save_img(fd, f'sample_{i:02d}_3_target_FDCT.png')
    
    print(f"Sample {i}: saved 3 images")

print(f"\n[DONE] {N_SAMPLES * 3} images saved in {OUTPUT_DIR}/")
print(f"View in VSCode explorer or run: ls -la {OUTPUT_DIR}/")
