import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to test log file")
    parser.add_argument("--manifest", default="/workspace/FastDDPM_Data/manifests/test.csv")
    parser.add_argument("--outdir", default="/workspace/FastDDPM_Experiments/test_metrics")
    args = parser.parse_args()

    log_path = Path(args.log)
    manifest_path = Path(args.manifest)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    text = log_path.read_text(errors="ignore")

    pattern = re.compile(
        r"Case\s+([A-Za-z0-9_\\-]+):\s*PSNR\s+([0-9eE+\\-.]+),\s*SSIM\s+([0-9eE+\\-.]+),\s*time\s+([0-9eE+\\-.]+)"
    )

    rows = []
    for m in pattern.finditer(text):
        case_id = m.group(1)
        psnr = float(m.group(2))
        ssim = float(m.group(3))
        infer_time = float(m.group(4))
        patient_id = case_id.split("_")[0]

        rows.append({
            "case_id": case_id,
            "patient_id": patient_id,
            "psnr": psnr,
            "ssim": ssim,
            "inference_time_sec": infer_time,
        })

    if not rows:
        print("No metric lines found yet.")
        return

    df = pd.DataFrame(rows)

    expected = None
    if manifest_path.exists():
        expected = len(pd.read_csv(manifest_path))

    per_case_csv = outdir / "test_metrics_per_case.csv"
    per_patient_csv = outdir / "test_metrics_per_patient.csv"
    summary_json = outdir / "test_metrics_summary.json"
    summary_txt = outdir / "test_metrics_summary.txt"

    df.to_csv(per_case_csv, index=False)

    per_patient = df.groupby("patient_id").agg(
        num_slices=("case_id", "count"),
        mean_psnr=("psnr", "mean"),
        mean_ssim=("ssim", "mean"),
        mean_inference_time_sec=("inference_time_sec", "mean"),
    ).reset_index()
    per_patient.to_csv(per_patient_csv, index=False)

    summary = {
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
        "num_cases_parsed": int(len(df)),
        "num_cases_expected": int(expected) if expected is not None else None,
        "complete": bool(expected is not None and len(df) >= expected),
        "mean_psnr": float(df["psnr"].mean()),
        "std_psnr": float(df["psnr"].std()),
        "min_psnr": float(df["psnr"].min()),
        "max_psnr": float(df["psnr"].max()),
        "mean_ssim": float(df["ssim"].mean()),
        "std_ssim": float(df["ssim"].std()),
        "min_ssim": float(df["ssim"].min()),
        "max_ssim": float(df["ssim"].max()),
        "mean_inference_time_sec": float(df["inference_time_sec"].mean()),
        "total_inference_time_sec": float(df["inference_time_sec"].sum()),
    }

    summary_json.write_text(json.dumps(summary, indent=2))
    summary_txt.write_text(
        "\n".join([
            "===== TEST SET METRICS SUMMARY =====",
            f"Parsed cases: {summary['num_cases_parsed']} / {summary['num_cases_expected']}",
            f"Complete: {summary['complete']}",
            f"Mean PSNR: {summary['mean_psnr']:.6f}",
            f"Std PSNR: {summary['std_psnr']:.6f}",
            f"Min PSNR: {summary['min_psnr']:.6f}",
            f"Max PSNR: {summary['max_psnr']:.6f}",
            f"Mean SSIM: {summary['mean_ssim']:.6f}",
            f"Std SSIM: {summary['std_ssim']:.6f}",
            f"Min SSIM: {summary['min_ssim']:.6f}",
            f"Max SSIM: {summary['max_ssim']:.6f}",
            f"Mean inference time/image: {summary['mean_inference_time_sec']:.6f} sec",
            f"Total inference time: {summary['total_inference_time_sec']:.2f} sec",
            "",
            f"Per-case CSV: {per_case_csv}",
            f"Per-patient CSV: {per_patient_csv}",
            f"Summary JSON: {summary_json}",
        ])
    )

    print(summary_txt.read_text())


if __name__ == "__main__":
    main()
