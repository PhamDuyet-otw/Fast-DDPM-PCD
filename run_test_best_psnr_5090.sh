#!/usr/bin/env bash
set -e

cd /workspace/Fast-DDPM-PCD
source /workspace/venvs/fastddpm5090/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

TRAIN_DOC="rtx5090_2gpu_bs6_full_400k"
EVAL_DOC="rtx5090_2gpu_bs6_test_best_psnr"

TRAIN_DIR="/workspace/FastDDPM_Experiments/logs/${TRAIN_DOC}"
EVAL_DIR="/workspace/FastDDPM_Experiments/logs/${EVAL_DOC}"

mkdir -p "${EVAL_DIR}"

# dùng best checkpoint làm ckpt.pth để code sample load
cp "${TRAIN_DIR}/best_psnr.pth" "${EVAL_DIR}/ckpt_500.pth"
cp "${TRAIN_DIR}/config.yml" "${EVAL_DIR}/config.yml" || true

echo "Using checkpoint:"
ls -lh "${EVAL_DIR}/ckpt_500.pth"

python fast_ddpm_main.py \
  --config ldfd_npy_manifest_v2_full_5090_bs6.yml \
  --dataset LDFDCT \
  --exp /workspace/FastDDPM_Experiments \
  --doc ${EVAL_DOC} \
  --sample \
  --fid \
  --scheduler_type uniform \
  --timesteps 10 \
  2>&1 | tee /workspace/${EVAL_DOC}.log
