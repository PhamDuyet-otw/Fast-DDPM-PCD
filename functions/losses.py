import torch
import math
import time
from medpy import metric
import numpy as np
np.bool = np.bool_

_sg_debug_logged = False


def calculate_psnr(img1, img2):
    # img1: img
    # img2: gt
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2)**2)
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)



def sr_noise_estimation_loss(model,
                          x_bw: torch.Tensor,
                          x_md: torch.Tensor,
                          x_fw: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x_md * a.sqrt() + e * (1.0 - a).sqrt()

    output = model(torch.cat([x_bw, x_fw, x], dim=1), t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)



def sg_noise_estimation_loss(model,
                          x_img: torch.Tensor,
                          x_gt: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM
    global _sg_debug_logged
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

    # Create noisy target x_t using provided noise 'e'
    x_t = x_gt * a.sqrt() + e * (1.0 - a).sqrt()

    # Model input: concat noisy target first, then condition (LD)
    model_input = torch.cat([x_t, x_img], dim=1)

    output = model(model_input, t.float())

    # Debug print once to help verify tensor shapes and ordering
    if not _sg_debug_logged:
        try:
            print(f"[SG_LOSS_DEBUG] t: {tuple(t.shape)}")
            print(f"[SG_LOSS_DEBUG] condition (x_img): {tuple(x_img.shape)}")
            print(f"[SG_LOSS_DEBUG] target (x_gt): {tuple(x_gt.shape)}")
            print(f"[SG_LOSS_DEBUG] x_t: {tuple(x_t.shape)}")
            print(f"[SG_LOSS_DEBUG] model_input: {tuple(model_input.shape)}")
            print(f"[SG_LOSS_DEBUG] pred: {tuple(output.shape)}")
            print(f"[SG_LOSS_DEBUG] noise (e): {tuple(e.shape)}")
        except Exception:
            pass
        _sg_debug_logged = True

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def noise_estimation_loss_conditional(model,
                                      condition: torch.Tensor,
                                      target: torch.Tensor,
                                      t: torch.LongTensor,
                                      e: torch.Tensor,
                                      b: torch.Tensor,
                                      keepdim=False):
    """
    Conditional noise estimation loss for models that take concatenated
    noisy target and condition as input (e.g., 2.5D LD->FD).
    Args:
        model: neural network that expects input shape [B, C=4, H, W]
        condition: [B, 3, H, W]
        target: [B, 1, H, W]
        t: [B] timesteps
        e: noise tensor same shape as target
        b: betas tensor
    """
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x_t = target * a.sqrt() + e * (1.0 - a).sqrt()
    model_input = torch.cat([x_t, condition], dim=1)
    pred = model(model_input, t.float())

    if keepdim:
        return (e - pred).square().sum(dim=(1, 2, 3))
    else:
        return (e - pred).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
    'sr': sr_noise_estimation_loss,
    'sg': sg_noise_estimation_loss,
    'simple_conditional': noise_estimation_loss_conditional,
}