"""
sliding_window.py - 512x512 inference via patch denoising for M1.
"""
import numpy as np
import torch


def create_gaussian_mask(size, sigma_ratio=0.25):
    """2D Gaussian weight mask, center high / edges low."""
    sigma = size * sigma_ratio
    center = size / 2.0
    coords = np.arange(size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dist_sq = (yy - center) ** 2 + (xx - center) ** 2
    weight = np.exp(-dist_sq / (2 * sigma ** 2))
    weight = weight.astype(np.float32)
    weight = weight / weight.max()
    return torch.from_numpy(weight).unsqueeze(0).unsqueeze(0)


def sliding_window_denoise(
    input_tensor,
    model,
    betas,
    seq,
    sg_sample_fn,
    patch_size=256,
    stride=128,
    eta=0.0,
    seed=42,
    device="cuda",
    progress_callback=None,
):
    """Run Fast-DDPM sliding-window denoising on a 512x512 input."""
    assert input_tensor.shape == (1, 3, 512, 512), \
        f"Expected (1, 3, 512, 512), got {tuple(input_tensor.shape)}"

    H, W = 512, 512
    device = torch.device(device)
    input_tensor = input_tensor.to(device)

    ys = list(range(0, H - patch_size + 1, stride))
    xs_pos = list(range(0, W - patch_size + 1, stride))
    if ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    if xs_pos[-1] != W - patch_size:
        xs_pos.append(W - patch_size)

    total_patches = len(ys) * len(xs_pos)
    weight = create_gaussian_mask(patch_size).to(device)

    output_acc = torch.zeros(1, 1, H, W, device=device)
    weight_acc = torch.zeros(1, 1, H, W, device=device)

    torch.manual_seed(seed)

    patch_idx = 0
    with torch.no_grad():
        for y in ys:
            for x in xs_pos:
                patch_idx += 1
                if progress_callback:
                    progress_callback(patch_idx, total_patches)

                patch_cond = input_tensor[:, :, y:y+patch_size, x:x+patch_size]
                noise = torch.randn(1, 1, patch_size, patch_size, device=device)

                xs_list, _ = sg_sample_fn(noise, patch_cond, seq, model, betas, eta=eta)
                denoised_patch = xs_list[-1].to(device)

                output_acc[:, :, y:y+patch_size, x:x+patch_size] += denoised_patch * weight
                weight_acc[:, :, y:y+patch_size, x:x+patch_size] += weight

    output = output_acc / (weight_acc + 1e-8)
    return output, total_patches


if __name__ == "__main__":
    m = create_gaussian_mask(256)
    print(f"Gaussian mask: shape={tuple(m.shape)}, "
          f"center={m[0,0,128,128].item():.3f}, "
          f"corner={m[0,0,0,0].item():.3f}")
