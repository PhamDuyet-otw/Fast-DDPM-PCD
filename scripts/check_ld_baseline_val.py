import yaml, torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
from datasets.LDFDCT import LDFDCT
from runners.diffusion import Diffusion

def to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [to_ns(v) for v in x]
    return x

with open("configs/ldfd_v3_2p5d_5090_full.yml", "r") as f:
    cfg = to_ns(yaml.safe_load(f))

args = SimpleNamespace()
runner = Diffusion(args, cfg, torch.device("cuda:0"))

ds = LDFDCT(cfg.data.val_dataroot, cfg.data.image_size, split="val", config=cfg)
loader = DataLoader(ds, batch_size=1, shuffle=False)

psnr_vals, ssim_vals = [], []

for i, x in enumerate(loader):
    if i >= 128:
        break

    ld = x["LD"]
    fd = x["FD"]

    # 2.5D input: lấy lát giữa LD[z]
    ld_center = ld[:, 1:2, :, :]

    psnr_vals.append(runner._psnr_batch(ld_center, fd))
    ssim_vals.append(runner._ssim_batch(ld_center, fd))

print("LD baseline on val, 128 batches")
print("PSNR:", sum(psnr_vals) / len(psnr_vals))
print("SSIM:", sum(ssim_vals) / len(ssim_vals))
