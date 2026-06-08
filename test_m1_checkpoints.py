"""
Test M1 với 3 checkpoints (best_psnr, best_ssim, ckpt_400000)
Trên N samples val, tính average PSNR/SSIM
"""
import os, sys, yaml, torch, numpy as np
sys.path.insert(0, '.')
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

from datasets.LDFDCT import LDFDCT
from models.diffusion import Model
from models.ema import EMAHelper
from functions.denoising import sg_generalized_steps

# === Config ===
CONFIG_PATH = 'configs/proposed_2p5d_heads4_400k.yml'
CHECKPOINTS = [
    ('best_psnr',   'exp/logs/Fast-DDPM_experiments/best_psnr.pth'),
    ('best_ssim',   'exp/logs/Fast-DDPM_experiments/best_ssim.pth'),
    ('ckpt_400000', 'exp/logs/Fast-DDPM_experiments/ckpt_400000.pth'),
]
N_SAMPLES = 10  # Số sample để test
OUTPUT_DIR = 'test_outputs/m1_checkpoints'

os.makedirs(OUTPUT_DIR, exist_ok=True)

class Config:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

# Load config + dataset
cfg = Config(yaml.safe_load(open(CONFIG_PATH)))
print(f"[CONFIG] {CONFIG_PATH}")

dataset = LDFDCT(cfg.data.test_dataroot, cfg.data.image_size, split='test', config=cfg)
print(f"[DATASET] {len(dataset)} samples in test\n")

# Diffusion params
betas = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end,
                       cfg.diffusion.num_diffusion_timesteps).cuda()
T = cfg.diffusion.num_diffusion_timesteps
n_steps = 10
skip = T // n_steps
seq = list(range(0, T, skip))[:n_steps]

# Helper: PSNR
def psnr(a, b):
    a = (a.clamp(-1, 1) + 1) / 2
    b = (b.clamp(-1, 1) + 1) / 2
    mse = ((a - b) ** 2).mean()
    return (10 * torch.log10(4.0 / (mse + 1e-12))).item()

def ssim(a, b):
    a = ((a.clamp(-1, 1) + 1) / 2).squeeze().cpu().numpy()
    b = ((b.clamp(-1, 1) + 1) / 2).squeeze().cpu().numpy()
    return ssim_fn(a, b, data_range=1.0)

# Test mỗi checkpoint
results = {}

for ckpt_name, ckpt_path in CHECKPOINTS:
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] {ckpt_name} không tồn tại: {ckpt_path}")
        continue
    
    print(f"\n{'='*60}")
    print(f"[CHECKPOINT] {ckpt_name}")
    print(f"{'='*60}")
    
    # Load model
    model = Model(cfg).cuda()
    states = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    
    if isinstance(states, list):
        model_state = states[0]
    elif isinstance(states, dict):
        model_state = states.get('model', states)
    else:
        model_state = states
    
    model.load_state_dict(model_state)
    
    # Apply EMA
    if isinstance(states, list) and len(states) >= 5:
        try:
            ema_helper = EMAHelper(mu=cfg.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[4])
            ema_helper.ema(model)
        except Exception as e:
            print(f"  [EMA skip] {e}")
    
    model.eval()
    
    # Test trên N samples
    psnr_inputs, psnr_outputs = [], []
    ssim_inputs, ssim_outputs = [], []
    
    for i in range(N_SAMPLES):
        sample = dataset[i]
        ld = sample['LD'].unsqueeze(0).cuda()  # (1, 3, 256, 256) cho 2.5D
        fd = sample['FD'].unsqueeze(0).cuda()  # (1, 1, 256, 256)
        
        # Get center slice của LD cho PSNR input
        # LD shape: (1, 3, 256, 256) → center is channel index 1
        if ld.shape[1] == 3:
            ld_center = ld[:, 1:2]  # (1, 1, 256, 256)
        else:
            ld_center = ld
        
        with torch.no_grad():
            x_t = torch.randn_like(fd)
            xs, _ = sg_generalized_steps(x_t, ld, seq, model, betas, eta=0.0)
            denoised = xs[-1].cuda()
        
        p_in = psnr(ld_center, fd)
        p_out = psnr(denoised, fd)
        s_in = ssim(ld_center, fd)
        s_out = ssim(denoised, fd)
        
        psnr_inputs.append(p_in)
        psnr_outputs.append(p_out)
        ssim_inputs.append(s_in)
        ssim_outputs.append(s_out)
        
        print(f"  Sample {i}: PSNR {p_in:.2f}→{p_out:.2f} (+{p_out-p_in:.2f}), SSIM {s_in:.3f}→{s_out:.3f}")
    
    # Average
    avg_p_in = np.mean(psnr_inputs)
    avg_p_out = np.mean(psnr_outputs)
    avg_s_in = np.mean(ssim_inputs)
    avg_s_out = np.mean(ssim_outputs)
    
    results[ckpt_name] = {
        'psnr_in': avg_p_in,
        'psnr_out': avg_p_out,
        'ssim_in': avg_s_in,
        'ssim_out': avg_s_out,
        'psnr_gain': avg_p_out - avg_p_in,
        'ssim_gain': avg_s_out - avg_s_in,
    }
    
    print(f"\n  [AVG] PSNR: {avg_p_in:.4f} → {avg_p_out:.4f} (+{avg_p_out-avg_p_in:.4f} dB)")
    print(f"  [AVG] SSIM: {avg_s_in:.4f} → {avg_s_out:.4f} (+{avg_s_out-avg_s_in:.4f})")
    
    # Free memory
    del model, states
    torch.cuda.empty_cache()

# Summary
print(f"\n\n{'='*70}")
print(f"FINAL COMPARISON (avg over {N_SAMPLES} samples)")
print(f"{'='*70}")
print(f"{'Checkpoint':<15} {'PSNR_out':<12} {'SSIM_out':<12} {'PSNR_gain':<12} {'SSIM_gain':<12}")
print('-' * 70)
for name, r in results.items():
    print(f"{name:<15} {r['psnr_out']:<12.4f} {r['ssim_out']:<12.4f} {r['psnr_gain']:<12.4f} {r['ssim_gain']:<12.4f}")

# Find winner
if results:
    best_psnr_name = max(results, key=lambda k: results[k]['psnr_out'])
    best_ssim_name = max(results, key=lambda k: results[k]['ssim_out'])
    print(f"\nBest by PSNR: {best_psnr_name}")
    print(f"Best by SSIM: {best_ssim_name}")
