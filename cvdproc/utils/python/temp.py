import os
import glob
import subprocess
from typing import List, Dict, Optional

import pandas as pd

from cvdproc.utils.python.basic_image_processor import extract_roi_means


DWI_PIPELINE_ROOT = "/mnt/f/BIDS/SVD_BIDS/derivatives/dwi_pipeline"
REMOVE_TRACK_REGION_SCRIPT = "/mnt/e/codes/cvdproc/cvdproc/pipelines/bash/mrtrix3/remove_conn_region.sh"

OUTPUT_CSV = os.path.join(DWI_PIPELINE_ROOT, "NAWM_without_track_mean_FA_MD.csv")

THR = "2e-4"  # passed via environment variable THR to the bash script
FORCE_RERUN_MASK = False  # if True, rerun mask generation even if outputs exist


def parse_subject_session(dwi_pipeline_output_dir: str) -> Dict[str, str]:
    parts = os.path.normpath(dwi_pipeline_output_dir).split(os.sep)
    subj = ""
    ses = ""
    for p in parts:
        if p.startswith("sub-"):
            subj = p
        if p.startswith("ses-"):
            ses = p
    return {"subject": subj, "session": ses}


def required_inputs_exist(dwi_dir: str) -> Optional[Dict[str, str]]:
    fa_file = os.path.join(dwi_dir, "dti_FA.nii.gz")
    md_file = os.path.join(dwi_dir, "dti_MD.nii.gz")
    nawm_mask = os.path.join(dwi_dir, "dwi_metrics_stats", "WM_final.nii.gz")
    track = os.path.join(dwi_dir, "tckgen_output", "tracked.tck")

    req = {
        "fa_file": fa_file,
        "md_file": md_file,
        "nawm_mask": nawm_mask,
        "track": track,
    }

    for k, v in req.items():
        if not os.path.isfile(v):
            return None

    return req


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def run_remove_track_region(
    nawm_mask: str,
    track: str,
    out_tract_mask: str,
    out_tdi_norm: str,
    out_wm_mask: str,
) -> None:
    ensure_dir(out_tract_mask)
    ensure_dir(out_tdi_norm)
    ensure_dir(out_wm_mask)

    env = os.environ.copy()
    env["THR"] = THR

    cmd = [
        "bash",
        REMOVE_TRACK_REGION_SCRIPT,
        nawm_mask,
        track,
        out_tract_mask,
        out_tdi_norm,
        out_wm_mask,
    ]
    subprocess.run(cmd, check=True, env=env)


def extract_mean_in_mask(metric_nii: str, mask_nii: str, out_csv: str) -> float:
    ensure_dir(out_csv)
    _, means = extract_roi_means(
        input_image=metric_nii,
        roi_image=mask_nii,
        ignore_background=True,
        roi_label=[1],
        output_csv=out_csv,
    )
    if means is None or len(means) == 0:
        raise RuntimeError(f"Empty mean result for: metric={metric_nii}, mask={mask_nii}")
    return float(means[0])


def main() -> None:
    if not os.path.isfile(REMOVE_TRACK_REGION_SCRIPT):
        raise FileNotFoundError(f"remove_track_region script not found: {REMOVE_TRACK_REGION_SCRIPT}")

    session_dirs = sorted(glob.glob(os.path.join(DWI_PIPELINE_ROOT, "sub-*", "ses-*")))
    rows: List[Dict[str, object]] = []

    for dwi_dir in session_dirs:
        meta = parse_subject_session(dwi_dir)
        subject = meta["subject"]
        session = meta["session"]

        req = required_inputs_exist(dwi_dir)
        if req is None:
            continue

        fa_file = req["fa_file"]
        md_file = req["md_file"]
        nawm_mask = req["nawm_mask"]
        track = req["track"]

        out_tract_mask = os.path.join(dwi_dir, "tckgen_output", "track_mask.nii.gz")
        out_tdi_norm = os.path.join(dwi_dir, "tckgen_output", "track_tdi_norm.nii.gz")
        out_wm_mask = os.path.join(dwi_dir, "dwi_metrics_stats", "NAWM_without_track.nii.gz")

        try:
            if FORCE_RERUN_MASK or (not os.path.isfile(out_wm_mask)):
                run_remove_track_region(
                    nawm_mask=nawm_mask,
                    track=track,
                    out_tract_mask=out_tract_mask,
                    out_tdi_norm=out_tdi_norm,
                    out_wm_mask=out_wm_mask,
                )

            fa_mean = extract_mean_in_mask(
                metric_nii=fa_file,
                mask_nii=out_wm_mask,
                out_csv=os.path.join(dwi_dir, "dwi_metrics_stats", "NAWM_without_track_FA.csv"),
            )
            md_mean = extract_mean_in_mask(
                metric_nii=md_file,
                mask_nii=out_wm_mask,
                out_csv=os.path.join(dwi_dir, "dwi_metrics_stats", "NAWM_without_track_MD.csv"),
            )

            rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "FA": fa_mean,
                    "MD": md_mean,
                }
            )

        except Exception as e:
            print(f"[WARN] Failed: {subject} {session} | {dwi_dir} | {e}")

    df = pd.DataFrame(rows, columns=["subject", "session", "FA", "MD"])
    df = df.sort_values(["subject", "session"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
