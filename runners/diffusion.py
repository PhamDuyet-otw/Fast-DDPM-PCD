import os
import logging
import time
import glob

import numpy as np
import pandas as pd
import math
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry, calculate_psnr
from datasets import data_transform, inverse_data_transform
from datasets.pmub import PMUB
from datasets.LDFDCT import LDFDCT
from datasets.BRATS import BRATS
from functions.ckpt_util import get_ckpt_path
from skimage.metrics import structural_similarity as ssim
import torchvision.utils as tvu
import torchvision
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from PIL import Image


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(x) + 1)
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule =='alpha_cosine':
        s = 0.008
        timesteps = np.arange(0, num_diffusion_timesteps+1, dtype=np.float64)/num_diffusion_timesteps
        alphas_cumprod = np.cos((timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=None, a_max=0.999)
    elif beta_schedule == 'alpha_sigmoid':
        x = np.linspace(-6, 6, 1001)
        alphas_cumprod = sigmoid(x)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=None, a_max=0.999)
    elif beta_schedule == 'alpha_linear':
        timesteps = np.arange(0, num_diffusion_timesteps+1, dtype=np.float64)/num_diffusion_timesteps
        alphas_cumprod = -timesteps+1
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=None, a_max=0.999)

    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas
def ddp_is_on(args):
    return getattr(args, "distributed", False)


def ddp_is_main(args):
    return (not ddp_is_on(args)) or getattr(args, "rank", 0) == 0
def ddp_is_on(args):
    return getattr(args, "distributed", False)
def ddp_is_main(args):
    return (not ddp_is_on(args)) or getattr(args, "rank", 0) == 0
class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    
    # Training Fast-DDPM for tasks that have only one condition: image translation and CT denoising.
    def sg_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        
        if self.args.dataset=='LDFDCT':
            # LDFDCT for CT image denoising
            dataset = LDFDCT(self.config.data.train_dataroot, self.config.data.image_size, split='train', config=self.config)
            print('Start training your Fast-DDPM model on LDFDCT dataset.')
        elif self.args.dataset=='BRATS':
            # BRATS for brain image translation
            dataset = BRATS(self.config.data.train_dataroot, self.config.data.image_size, split='train')
            print('Start training your Fast-DDPM model on BRATS dataset.')
        print('The scheduler sampling type is {}. The number of involved time steps is {} out of 1000.'.format(self.args.scheduler_type, self.args.timesteps))
        
        # Thiết lập Sampler cho DDP
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=self.args.world_size,
            rank=self.args.rank,
            shuffle=True,
            drop_last=True
        ) if (hasattr(self.args, 'world_size') and self.args.world_size > 1) else None

        # Khởi tạo Loader
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.config.data.num_workers > 0)
        )

        # Validation loader for Branch 2: use val.csv, not train.csv.
        # Portable rule:
        #   1) use config.data.val_dataroot if available
        #   2) else infer val.csv from train.csv
        #   3) else use $LDCT_DATASET_ROOT/manifests/val.csv
        val_loader = None
        if self.args.dataset == 'LDFDCT' and ddp_is_main(args):
            try:
                val_dataroot = getattr(self.config.data, 'val_dataroot', None)

                if val_dataroot is None:
                    train_dataroot = str(self.config.data.train_dataroot)
                    if train_dataroot.endswith('train.csv'):
                        val_dataroot = os.path.join(os.path.dirname(train_dataroot), 'val.csv')
                    else:
                        dataset_root = os.environ.get('LDCT_DATASET_ROOT', None)
                        if dataset_root is not None:
                            val_dataroot = os.path.join(dataset_root, 'manifests', 'val.csv')
                        else:
                            val_dataroot = train_dataroot

                val_dataset = LDFDCT(val_dataroot, self.config.data.image_size, split='val', config=self.config)
                val_loader = data.DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True)
                print('Validation set loaded for LDFDCT: {} samples.'.format(len(val_dataset)))
                print('Validation manifest:', val_dataroot)
            except Exception as exc:
                logging.warning('Could not create validation loader: {}'.format(exc))
                val_loader = None

        model = Model(config)
        model = model.to(self.device)

        if ddp_is_on(args):
            model = DDP(
            model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False
            )
        else:
            model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            ckpt_path = os.path.join(self.args.log_path, "ckpt.pth")
            states = torch.load(ckpt_path, map_location="cpu")

            # Load model state an toàn cho cả DDP và non-DDP
            raw_model = model.module if hasattr(model, "module") else model
            model_state = states[0]

            # Nếu checkpoint cũ lưu từ DDP có prefix "module.", bỏ prefix đi
            if any(k.startswith("module.") for k in model_state.keys()):
                model_state = {
                    k.replace("module.", "", 1): v
                    for k, v in model_state.items()
                }

            raw_model.load_state_dict(model_state, strict=True)

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])

            # Đưa optimizer state về đúng GPU của từng rank
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            start_epoch = states[2]
            step = states[3]

            if self.config.model.ema and ema_helper is not None:
                ema_helper.load_state_dict(states[4])

            del states
            torch.cuda.empty_cache()

            if ddp_is_main(args):
                logging.info(f"Resumed training from {ckpt_path} at step={step}, epoch={start_epoch}")
        best_psnr = -1.0
        best_ssim = -1.0
        max_val_batches = getattr(self.config.training, 'validation_num_batches', 8)
        n_iters = getattr(self.config.training, 'n_iters', None)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for i, x in enumerate(train_loader):
                n = x['LD'].size(0)
                model.train()
                step += 1

                x_img = x['LD'].to(self.device)
                x_gt = x['FD'].to(self.device)

                b = self.betas

                # Create timestep tensor using antithetic pairing as specified
                t = torch.randint(
                    low=0,
                    high=self.num_timesteps,
                    size=(x_gt.shape[0] // 2 + 1,),
                    device=self.device,
                )
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:x_gt.shape[0]]
                t = t.long()

                # Create noise tensor matching the target shape
                e = torch.randn_like(x_gt)

                loss = loss_registry[config.model.type](model, x_img, x_gt, t, e, b)

                # Debug logging for 2.5D mode (one-time only)
                if not hasattr(self, '_debug_25d_logged') and hasattr(config.data, 'input_mode') and config.data.input_mode == '2.5d':
                    logging.info(f"[2.5D DEBUG] x_img (condition) shape: {x_img.shape}, min: {x_img.min():.4f}, max: {x_img.max():.4f}")
                    logging.info(f"[2.5D DEBUG] x_gt (target) shape: {x_gt.shape}, min: {x_gt.min():.4f}, max: {x_gt.max():.4f}")
                    model_input = torch.cat([x_gt, x_img], dim=1)
                    logging.info(f"[2.5D DEBUG] model input shape [target/noisy, condition]: {model_input.shape}, min: {model_input.min():.4f}, max: {model_input.max():.4f}")
                    self._debug_25d_logged = True

                if ddp_is_main(args):
                    tb_logger.add_scalar("loss", loss.item(), global_step=step)
                    if ddp_is_main(args):
                        logging.info(f"step: {step}, loss: {loss.item()}")
                        tb_logger.add_scalar("loss", loss.item(), global_step=step)

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if ddp_is_main(args) and (step % self.config.training.snapshot_freq == 0 or step == 1):
                    states = self._make_training_state(model, optimizer, epoch, step, ema_helper)
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    torch.save(states, os.path.join(self.args.log_path, "latest.pth"))

                validation_freq = getattr(self.config.training, 'validation_freq', 0)
                if val_loader is not None and validation_freq > 0 and step % validation_freq == 0:
                    val_psnr, val_ssim = self._validate_sg(model, val_loader, max_batches=max_val_batches)

                    tb_logger.add_scalar("val/psnr", val_psnr, global_step=step)
                    tb_logger.add_scalar("val/ssim", val_ssim, global_step=step)

                    logging.info(
                        f"validation step: {step}, val_psnr: {val_psnr:.6f}, val_ssim: {val_ssim:.6f}"
                    )
                    print(f"[VAL] step={step} PSNR={val_psnr:.6f} SSIM={val_ssim:.6f}")

                    states = self._make_training_state(model, optimizer, epoch, step, ema_helper)

                    if val_psnr > best_psnr:
                        best_psnr = val_psnr
                        torch.save(states, os.path.join(self.args.log_path, "best_psnr.pth"))
                        logging.info(f"Saved best_psnr.pth at step {step} with PSNR {best_psnr:.6f}")

                    if val_ssim > best_ssim:
                        best_ssim = val_ssim
                        torch.save(states, os.path.join(self.args.log_path, "best_ssim.pth"))
                        logging.info(f"Saved best_ssim.pth at step {step} with SSIM {best_ssim:.6f}")
                do_final_snapshot = (
                    n_iters is not None
                    and step >= n_iters
                    and ddp_is_main(args)
                )

                if do_final_snapshot:
                    raw_model = model.module if hasattr(model, "module") else model

                    states = [
                        raw_model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]

                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt.pth"),
                    )

                    logging.info(f"Saved final checkpoint at step {step}")

                if ddp_is_on(args) and n_iters is not None and step >= n_iters:
                    dist.barrier()
                if n_iters is not None and step >= n_iters:
                    if ddp_is_main(args):
                        logging.info(f"Reached n_iters={n_iters}. Stop training at step={step}.")
                    return
                    

    # Training Fast-DDPM for tasks that have two conditions: multi image super-resolution.
    def sr_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        dataset = PMUB(self.config.data.train_dataroot, self.config.data.image_size, split='train')
        print('Start training your Fast-DDPM model on PMUB dataset.')
        print('The scheduler sampling type is {}. The number of involved time steps is {} out of 1000.'.format(self.args.scheduler_type, self.args.timesteps))
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True)

        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                n = x['BW'].size(0)
                model.train()
                step += 1

                x_bw = x['BW'].to(self.device)
                x_md = x['MD'].to(self.device)
                x_fw = x['FW'].to(self.device)

                e = torch.randn_like(x_md)
                b = self.betas

                if self.args.scheduler_type == 'uniform':
                    skip = self.num_timesteps // self.args.timesteps
                    t_intervals = torch.arange(-1, self.num_timesteps, skip)
                    t_intervals[0] = 0
                elif self.args.scheduler_type == 'non-uniform':
                    t_intervals = torch.tensor([0, 199, 399, 599, 699, 799, 849, 899, 949, 999])
                    
                    if self.args.timesteps != 10:
                        num_1 = int(self.args.timesteps*0.4)
                        num_2 = int(self.args.timesteps*0.6)
                        stage_1 = torch.linspace(0, 699, num_1+1)[:-1]
                        stage_2 = torch.linspace(699, 999, num_2)
                        stage_1 = torch.ceil(stage_1).long()
                        stage_2 = torch.ceil(stage_2).long()
                        t_intervals = torch.cat((stage_1, stage_2))
                else:
                    raise Exception("The scheduler type is either uniform or non-uniform.")

                # antithetic sampling
                idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                idx_2 = len(t_intervals)-idx_1-1
                idx = torch.cat([idx_1, idx_2], dim=0)[:n]
                t = t_intervals[idx].to(self.device)

                loss = loss_registry[config.model.type](model, x_bw, x_md, x_fw, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                do_snapshot = (
                    step % self.config.training.snapshot_freq == 0 or step == 1
                )

                if ddp_is_main(args) and do_snapshot:
                    raw_model = model.module if hasattr(model, "module") else model

                    states = [
                        raw_model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]

                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt.pth"),
                    )

                    logging.info(f"Saved checkpoint at step {step}")

                if ddp_is_on(args) and do_snapshot:
                    dist.barrier()
    
    # Training original DDPM for tasks that have only one condition: image translation and CT denoising.
    def sg_ddpm_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if self.args.dataset=='LDFDCT':
            # LDFDCT for CT image denoising
            dataset = LDFDCT(self.config.data.train_dataroot, self.config.data.image_size, split='train', config=self.config)
            print('Start training DDPM model on LDFDCT dataset.')
        elif self.args.dataset=='BRATS':
            # BRATS for brain image translation
            dataset = BRATS(self.config.data.train_dataroot, self.config.data.image_size, split='train')
            print('Start training DDPM model on BRATS dataset.')
            
        print('The number of involved time steps is {} out of 1000.'.format(self.args.timesteps))
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True)

        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                n = x['LD'].size(0)
                model.train()
                step += 1

                x_img = x['LD'].to(self.device)
                x_gt = x['FD'].to(self.device)

                e = torch.randn_like(x_gt)
                b = self.betas

                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x_img, x_gt, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                do_snapshot = (
                    step % self.config.training.snapshot_freq == 0 or step == 1)

                if ddp_is_main(args) and do_snapshot:
                    raw_model = model.module if hasattr(model, "module") else model

                    states = [
                        raw_model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]

                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt.pth"),
                    )

                    logging.info(f"Saved checkpoint at step {step}")

                if ddp_is_on(args) and do_snapshot:
                    dist.barrier()
    # Training original DDPM for tasks that have two conditions: multi image super-resolution.
    def sr_ddpm_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        dataset = PMUB(self.config.data.train_dataroot, self.config.data.image_size, split='train')
        print('Start training DDPM model on PMUB dataset.')
        print('The number of involved time steps is {} out of 1000.'.format(self.args.timesteps))
        
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True)

        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        time_start = time.time()
        total_time = 0
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                n = x['BW'].size(0)
                model.train()
                step += 1

                x_bw = x['BW'].to(self.device)
                x_md = x['MD'].to(self.device)
                x_fw = x['FW'].to(self.device)

                e = torch.randn_like(x_md)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x_bw, x_md, x_fw, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

               
    # Sampling for tasks that have two conditions: multi image super-resolution.
    def sr_sample(self):
        ckpt_list = self.config.sampling.ckpt_id
        for ckpt_idx in ckpt_list:
            self.ckpt_idx = ckpt_idx
            model = Model(self.config)
            print('Start inference on model of {} steps'.format(ckpt_idx))

            if not self.args.use_pretrained:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{ckpt_idx}.pth"
                    ),
                    map_location=self.config.device,
                )
                model = model.to(self.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(states[0], strict=True)

                if self.config.model.ema:
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                else:
                    ema_helper = None
            else:
                # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

            model.eval()

            if self.args.fid:
                self.sr_sample_fid(model)
            elif self.args.interpolation:
                self.sr_sample_interpolation(model)
            elif self.args.sequence:
                self.sample_sequence(model)
            else:
                raise NotImplementedError("Sample procedeure not defined")


    # Sampling for tasks that have only one condition: image translation and CT denoising.
    def sg_sample(self):
        ckpt_list = self.config.sampling.ckpt_id
        for ckpt_idx in ckpt_list:
            self.ckpt_idx = ckpt_idx
            model = Model(self.config)
            print('Start inference on model of {} steps'.format(ckpt_idx))

            if not self.args.use_pretrained:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{ckpt_idx}.pth"
                    ),
                    map_location=self.config.device,
                )
                model = model.to(self.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(states[0], strict=True)

                if self.config.model.ema:
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                else:
                    ema_helper = None
            else:
                # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

            model.eval()

            if self.args.fid:
                self.sg_sample_fid(model)
            elif self.args.interpolation:
                self.sr_sample_interpolation(model)
            elif self.args.sequence:
                self.sample_sequence(model)
            else:
                raise NotImplementedError("Sample procedeure not defined")

                
    def sr_sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")

        sample_dataset = PMUB(self.config.data.sample_dataroot, self.config.data.image_size, split='calculate')
        print('Start sampling model on PMUB dataset.')
        print('The inference sample type is {}. The scheduler sampling type is {}. The number of involved time steps is {} out of 1000.'.format(self.args.sample_type, self.args.scheduler_type, self.args.timesteps))
        
        sample_loader = data.DataLoader(
            sample_dataset,
            batch_size=config.sampling_fid.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers)

        with torch.no_grad():
            data_num = len(sample_dataset)
            print('The length of test set is:', data_num)
            avg_psnr = 0.0
            avg_ssim = 0.0
            time_list = []
            psnr_list = []
            ssim_list = []

            for batch_idx, img in tqdm.tqdm(enumerate(sample_loader), desc="Generating image samples for FID evaluation."):
                n = img['BW'].shape[0]
                
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                x_bw = img['BW'].to(self.device)
                x_md = img['MD'].to(self.device)
                x_fw = img['FW'].to(self.device)
                case_name = img['case_name'][0]
                
                time_start = time.time()
                x = self.sr_sample_image(x, x_bw, x_fw, model)
                time_end = time.time()
                
                x = inverse_data_transform(config, x)
                x_md = inverse_data_transform(config, x_md)
                x_tensor = x
                x_md_tensor = x_md
                x_md = x_md.squeeze().float().cpu().numpy()
                x = x.squeeze().float().cpu().numpy()
                x_md = (x_md*255.0).round()
                x = (x*255.0).round()

                PSNR = 0.0 
                SSIM = 0.0
                for i in range(x.shape[0]):
                    psnr_temp = calculate_psnr(x[i,:,:], x_md[i,:,:])
                    ssim_temp = ssim(x_md[i,:,:], x[i,:,:], data_range=255)
                    PSNR += psnr_temp
                    SSIM += ssim_temp
                    psnr_list.append(psnr_temp)
                    ssim_list.append(ssim_temp)

                PSNR_print = PSNR/x.shape[0]
                SSIM_print = SSIM/x.shape[0]

                case_time = time_end-time_start
                time_list.append(case_time)

                avg_psnr += PSNR
                avg_ssim += SSIM
                logging.info('Case {}: PSNR {}, SSIM {}, time {}'.format(case_name, PSNR_print, SSIM_print, case_time))

                for i in range(0, n):
                    # image:(0-1)
                    tvu.save_image(
                        x_tensor[i], os.path.join(self.args.image_folder, "{}_{}_pt.png".format(self.ckpt_idx, img_id))
                    )
                    tvu.save_image(
                        x_md_tensor[i], os.path.join(self.args.image_folder, "{}_{}_gt.png".format(self.ckpt_idx, img_id))
                    )
                    img_id += 1
                    
            avg_psnr = avg_psnr / data_num
            avg_ssim = avg_ssim / data_num
            # Drop first and last for time calculation.
            avg_time = sum(time_list[1:-1])/(len(time_list)-2)
            logging.info('Average: PSNR {}, SSIM {}, time {}'.format(avg_psnr, avg_ssim, avg_time))


    def sg_sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")


        if self.args.dataset=='LDFDCT':
            # LDFDCT for CT image denoising
            sample_dataset = LDFDCT(self.config.data.sample_dataroot, self.config.data.image_size, split='calculate', config=self.config)
            print('Start training model on LDFDCT dataset.')
        elif self.args.dataset=='BRATS':
            # BRATS for brain image translation
            sample_dataset = BRATS(self.config.data.sample_dataroot, self.config.data.image_size, split='calculate')
            print('Start training model on BRATS dataset.')
        print('The inference sample type is {}. The scheduler sampling type is {}. The number of involved time steps is {} out of 1000.'.format(self.args.sample_type, self.args.scheduler_type, self.args.timesteps))
        
        sample_loader = data.DataLoader(
            sample_dataset,
            batch_size=config.sampling_fid.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers)

        with torch.no_grad():
            data_num = len(sample_dataset)
            print('The length of test set is:', data_num)
            avg_psnr = 0.0
            avg_ssim = 0.0
            time_list = []
            psnr_list = []
            ssim_list = []

            for batch_idx, sample in tqdm.tqdm(enumerate(sample_loader), desc="Generating image samples for FID evaluation."):
                n = sample['LD'].shape[0]

                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                x_img = sample['LD'].to(self.device)
                x_gt = sample['FD'].to(self.device)
                case_name = sample['case_name']
                
                time_start = time.time()
                x = self.sg_sample_image(x, x_img, model)
                time_end = time.time()
                
                x = inverse_data_transform(config, x)
                x_gt = inverse_data_transform(config, x_gt)
                x_tensor = x
                x_gt_tensor = x_gt
                x_gt = x_gt.squeeze().float().cpu().numpy()
                x = x.squeeze().float().cpu().numpy()
                x_gt = x_gt*255
                x = x*255
                x = np.asarray(x)
                x_gt = np.asarray(x_gt)

                if x.ndim == 2:
                    x = x[None, :, :]

                if x_gt.ndim == 2:
                    x_gt = x_gt[None, :, :]
                PSNR = 0.0 
                SSIM = 0.0
                for i in range(x.shape[0]):
                    psnr_temp = calculate_psnr(x[i] if x.ndim == 3 else x, x_gt[i] if x_gt.ndim == 3 else x_gt)
                    ssim_temp = ssim(x_gt[i,:,:], x[i,:,:], data_range=255)
                    PSNR += psnr_temp
                    SSIM += ssim_temp
                    psnr_list.append(psnr_temp)
                    ssim_list.append(ssim_temp)
                
                PSNR_print = PSNR/x.shape[0]
                SSIM_print = SSIM/x.shape[0]

                case_time = time_end-time_start
                time_list.append(case_time)

                avg_psnr += PSNR
                avg_ssim += SSIM
                logging.info('Case {}: PSNR {}, SSIM {}, time {}'.format(case_name[0], PSNR_print, SSIM_print, case_time))

                for i in range(0, n):
                    # image:(0-1)
                    tvu.save_image(
                        x_tensor[i], os.path.join(self.args.image_folder, "{}_{}_pt.png".format(self.ckpt_idx, img_id))
                    )
                    tvu.save_image(
                        x_gt_tensor[i], os.path.join(self.args.image_folder, "{}_{}_gt.png".format(self.ckpt_idx, img_id))
                    )
                    img_id += 1

            avg_psnr = avg_psnr / data_num
            avg_ssim = avg_ssim / data_num
            # Drop first and last for time calculation.
            avg_time = sum(time_list[1:-1])/(len(time_list)-2)
            logging.info('Average: PSNR {}, SSIM {}, time {}'.format(avg_psnr, avg_ssim, avg_time))


    def sr_sample_image(self, x, x_bw, x_fw, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.scheduler_type == 'uniform':
                skip = self.num_timesteps // self.args.timesteps
                seq = range(-1, self.num_timesteps, skip)
                seq = list(seq)
                seq[0] = 0
            elif self.args.scheduler_type == 'non-uniform':
                seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]

                if self.args.timesteps != 10:
                    num_1 = int(self.args.timesteps*0.4)
                    num_2 = int(self.args.timesteps*0.6)
                    stage_1 = np.linspace(0, 699, num_1+1)[:-1]
                    stage_2 = np.linspace(699, 999, num_2)
                    stage_1 = np.ceil(stage_1).astype(int)
                    stage_2 = np.ceil(stage_2).astype(int)
                    seq = np.concatenate((stage_1, stage_2))
            else:
                raise Exception("The scheduler type is either uniform or non-uniform.")

            from functions.denoising import generalized_steps, sr_generalized_steps

            xs = sr_generalized_steps(x, x_bw, x_fw, seq, model, self.betas, eta=self.args.eta)
            x = xs

        elif self.args.sample_type == "ddpm_noisy":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)

            from functions.denoising import ddpm_steps, sr_ddpm_steps

            x = sr_ddpm_steps(x, x_bw, x_fw, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x


    def sg_sample_image(self, x, x_img, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.scheduler_type == 'uniform':
                skip = self.num_timesteps // self.args.timesteps
                seq = range(-1, self.num_timesteps, skip)
                seq = list(seq)
                seq[0] = 0
            elif self.args.scheduler_type == 'non-uniform':
                seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]

                if self.args.timesteps != 10:
                    num_1 = int(self.args.timesteps*0.4)
                    num_2 = int(self.args.timesteps*0.6)
                    stage_1 = np.linspace(0, 699, num_1+1)[:-1]
                    stage_2 = np.linspace(699, 999, num_2)
                    stage_1 = np.ceil(stage_1).astype(int)
                    stage_2 = np.ceil(stage_2).astype(int)
                    seq = np.concatenate((stage_1, stage_2))
            else:
                raise Exception("The scheduler type is either uniform or non-uniform.")
                
            from functions.denoising import generalized_steps, sr_generalized_steps, sg_generalized_steps

            xs = sg_generalized_steps(x, x_img, seq, model, self.betas, eta=self.args.eta)
            x = xs

        elif self.args.sample_type == "ddpm_noisy":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)

            from functions.denoising import ddpm_steps, sr_ddpm_steps, sg_ddpm_steps

            x = sg_ddpm_steps(x, x_img, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x



    def _make_training_state(self, model, optimizer, epoch, step, ema_helper):
        states = [
            model.state_dict(),
            optimizer.state_dict(),
            epoch,
            step,
        ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())
        return states

    def _to_01(self, x):
        # Dataset/model tensors are expected in [-1, 1].
        return torch.clamp((x.detach() + 1.0) / 2.0, 0.0, 1.0)

    def _psnr_batch(self, pred, target):
        # Metric computation is done on CPU to avoid cuda/cpu mismatch.
        pred = self._to_01(pred).detach().cpu()
        target = self._to_01(target).detach().cpu()
        mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))
        return psnr.mean().item()

    def _ssim_batch(self, pred, target):
        # Lightweight validation SSIM using skimage on a small validation subset.
        from skimage.metrics import structural_similarity as ssim_fn
        pred = self._to_01(pred).detach().cpu().numpy()
        target = self._to_01(target).detach().cpu().numpy()

        vals = []
        for i in range(pred.shape[0]):
            p_img = pred[i, 0]
            t_img = target[i, 0]
            vals.append(ssim_fn(t_img, p_img, data_range=1.0))
        return float(sum(vals) / max(len(vals), 1))

    def _validate_sg(self, model, val_loader, max_batches=None):
        model.eval()
        psnr_vals, ssim_vals = [], []

        with torch.no_grad():
            for batch_idx, x in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                x_img = x['LD'].to(self.device)
                x_gt = x['FD'].to(self.device)

                # Start Fast-DDPM sampling from Gaussian noise, conditioned on LDCT.
                x_noise = torch.randn_like(x_gt)
                x_pred = self.sg_sample_image(x_noise, x_img, model, last=True)

                psnr_vals.append(self._psnr_batch(x_pred, x_gt))
                ssim_vals.append(self._ssim_batch(x_pred, x_gt))

        model.train()
        mean_psnr = float(sum(psnr_vals) / max(len(psnr_vals), 1))
        mean_ssim = float(sum(ssim_vals) / max(len(ssim_vals), 1))
        return mean_psnr, mean_ssim

    def test(self):
        pass
