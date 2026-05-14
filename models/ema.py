import copy
import torch


class EMAHelper:
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        self.shadow = {}
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().to(param.device)

    def update(self, module):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            p = param.detach()

            if name not in self.shadow:
                self.shadow[name] = p.clone().to(p.device)
                continue

            # Fix DDP resume:
            # rank 0 dùng cuda:0, rank 1 dùng cuda:1.
            # EMA shadow có thể được load từ checkpoint về cuda:0/CPU,
            # nên cần ép về đúng device của param hiện tại.
            shadow = self.shadow[name].detach().to(device=p.device, dtype=p.dtype)

            shadow.mul_(self.mu)
            shadow.add_(p, alpha=1.0 - self.mu)

            self.shadow[name] = shadow

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(
                    self.shadow[name].to(device=param.device, dtype=param.dtype).data
                )

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        # Save EMA về CPU để resume sạch trên 1 GPU / 2 GPU / GPU khác.
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# Giữ alias phòng khi chỗ khác import EMA
EMA = EMAHelper
