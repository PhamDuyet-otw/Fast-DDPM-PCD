#!/usr/bin/env bash

LOG="/workspace/rtx5090_2gpu_bs6_test_best_psnr.log"
MANIFEST="/workspace/FastDDPM_Data/manifests/test.csv"
OUTDIR="/workspace/FastDDPM_Experiments/test_metrics/best_psnr"

cd /workspace/Fast-DDPM-PCD
source /workspace/venvs/fastddpm5090/bin/activate

echo "Watching log: ${LOG}"
echo "Output dir: ${OUTDIR}"

while true; do
  clear
  python scripts/summarize_test_log.py \
    --log "${LOG}" \
    --manifest "${MANIFEST}" \
    --outdir "${OUTDIR}" || true

  echo ""
  echo "Last test log lines:"
  tail -20 "${LOG}" || true

  python - <<PY
import json
from pathlib import Path
p = Path("${OUTDIR}/test_metrics_summary.json")
if p.exists():
    s = json.loads(p.read_text())
    if s.get("complete"):
        print("\\nTEST COMPLETE. Summary saved.")
        raise SystemExit(0)
raise SystemExit(1)
PY

  if [ $? -eq 0 ]; then
    break
  fi

  sleep 60
done
