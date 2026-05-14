import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import torch.distributed as dist

from runners.diffusion import Diffusion


class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def is_main_process():
    return get_rank() == 0


def setup_ddp(args):
    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        args.device = torch.device("cuda", args.local_rank)
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def cleanup_ddp():
    if is_dist():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="ldfd_npy_manifest_v2_smoke.yml")
    parser.add_argument("--dataset", type=str, default="LDFDCT")
    parser.add_argument("--seed", type=int, default=1244)
    parser.add_argument("--exp", type=str, default="/workspace/FastDDPM_Experiments")
    parser.add_argument("--doc", type=str, default="v2_ddp_v100_test")
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--verbose", type=str, default="info")

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("-i", "--image_folder", type=str, default="images")
    parser.add_argument("--ni", action="store_false")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="generalized")
    parser.add_argument("--scheduler_type", type=str, default="uniform")
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args = setup_ddp(args)

    args.log_path = os.path.join(args.exp, "logs", args.doc)
    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)

    config = dict2namespace(config)

    if is_main_process():
        if not args.test and not args.sample and not args.resume_training:
            if os.path.exists(args.log_path):
                shutil.rmtree(args.log_path)
            if os.path.exists(tb_path):
                shutil.rmtree(tb_path)

        os.makedirs(args.log_path, exist_ok=True)
        os.makedirs(tb_path, exist_ok=True)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
    else:
        config.tb_logger = DummyWriter()

    if is_dist():
        dist.barrier()

    level = getattr(logging, args.verbose.upper(), logging.INFO)
    logging.getLogger().handlers.clear()

    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )

    handler1 = logging.StreamHandler()
    handler1.setFormatter(formatter)
    logging.getLogger().addHandler(handler1)

    if is_main_process():
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        handler2.setFormatter(formatter)
        logging.getLogger().addHandler(handler2)

    logging.getLogger().setLevel(level)

    config.device = args.device

    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.rank)
        torch.backends.cudnn.benchmark = True

    logging.info(
        f"Rank {args.rank}/{args.world_size}, local_rank={args.local_rank}, device={args.device}"
    )

    return args, config


def main():
    args, config = parse_args_and_config()

    if is_main_process():
        logging.info("Writing log file to {}".format(args.log_path))
        logging.info("Exp instance id = {}".format(os.getpid()))
        logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)

        if args.sample:
            if args.dataset == "PMUB":
                runner.sr_sample()
            elif args.dataset in ["LDFDCT", "BRATS"]:
                runner.sg_sample()
            else:
                raise Exception("Unsupported sampling dataset.")
        elif args.test:
            runner.test()
        else:
            if args.dataset == "PMUB":
                runner.sr_train()
            elif args.dataset in ["LDFDCT", "BRATS"]:
                runner.sg_train()
            else:
                raise Exception("Unsupported training dataset.")

    except Exception:
        logging.error(traceback.format_exc())
        cleanup_ddp()
        return 1

    cleanup_ddp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
