#!/usr/bin/env bash
set -e

cd /workspace/Fast-DDPM-PCD
source /workspace/venvs/fastddpm5090/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8

RUN_NAME="rtx5090_2gpu_bs6_full_400k"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  fast_ddpm_main_ddp.py \
  --config ldfd_npy_manifest_v2_full_5090_bs6.yml \
  --dataset LDFDCT \
  --exp /workspace/FastDDPM_Experiments \
  --doc ${RUN_NAME} \
  --scheduler_type uniform \
  --timesteps 10 \
  2>&1 | tee /workspace/${RUN_NAME}.log
