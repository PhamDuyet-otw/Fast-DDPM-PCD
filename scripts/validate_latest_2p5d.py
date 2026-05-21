import os, re, glob, argparse, yaml, torch
from types import SimpleNamespace
from torch.utils.data import DataLoader

from models.diffusion import Model
from models.ema import EMAHelper
from datasets.LDFDCT import LDFDCT
from runners.diffusion import Diffusion


def to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [to_ns(v) for v in x]
    return x


def find_latest_ckpt(log_dir):
    ckpts = []
    for p in glob.glob(os.path.join(log_dir, "ckpt_*.pth")):
        m = re.search(r"ckpt_(\d+)\.pth$", os.path.basename(p))
        if m:
            ckpts.append((int(m.group(1)), p))
    if ckpts:
        ckpts.sort()
        return ckpts[-1][1]
    return os.path.join(log_dir, "ckpt.pth")


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/ldfd_v3_2p5d_5090_full.yml")
parser.add_argument("--log_dir", default="/workspace/FastDDPM_Experiments/logs/v3_2p5d_5090_full")
parser.add_argument("--ckpt", default=None)
parser.add_argument("--max_batches", type=int, default=32)
args_cli = parser.parse_args()

with open(args_cli.config, "r") as f:
    cfg = to_ns(yaml.safe_load(f))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dummy_args = SimpleNamespace(
    sample_type="generalized",
    scheduler_type="uniform",
    timesteps=10,
    eta=0.0,
    skip=1,
    dataset="LDFDCT",
    world_size=1,
    rank=0,
    local_rank=0,
    resume_training=False,
    log_path=args_cli.log_dir,
    exp="/workspace/FastDDPM_Experiments",
    doc="manual_val",
)

runner = Diffusion(dummy_args, cfg, device)
model = Model(cfg).to(device)

ckpt_path = args_cli.ckpt or find_latest_ckpt(args_cli.log_dir)
print("Loading checkpoint:", ckpt_path)

states = torch.load(ckpt_path, map_location="cpu")
print("checkpoint epoch:", states[2])
print("checkpoint step :", states[3])

state_dict = states[0]
if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=True)

if getattr(cfg.model, "ema", False) and len(states) >= 5:
    print("Applying EMA weights")
    ema = EMAHelper(mu=cfg.model.ema_rate)
    ema.register(model)
    ema.load_state_dict(states[4])
    ema.ema(model)
else:
    print("No EMA applied")

model.eval()

val_dataset = LDFDCT(
    cfg.data.val_dataroot,
    cfg.data.image_size,
    split="val",
    config=cfg
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Val samples:", len(val_dataset))
print("Max batches:", args_cli.max_batches)

psnr, ssim = runner._validate_sg(model, val_loader, max_batches=args_cli.max_batches)

print("=" * 60)
print("CKPT:", ckpt_path)
print("STEP:", states[3])
print(f"PSNR: {psnr:.6f}")
print(f"SSIM: {ssim:.6f}")
print("=" * 60)
