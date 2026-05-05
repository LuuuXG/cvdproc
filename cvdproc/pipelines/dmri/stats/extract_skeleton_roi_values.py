#!/usr/bin/env python3

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract atlas ROI mean values from TBSS/GBSS 4D skeleton images."
    )
    parser.add_argument(
        "--pipeline_output_dir",
        required=True,
        help="Pipeline output directory, e.g. /mnt/f/UKBdata/WCH_output/TBSS_GBSS",
    )
    parser.add_argument(
        "--atlas_nifti",
        required=True,
        help="Path to atlas NIfTI file (multi-label atlas)",
    )
    parser.add_argument(
        "--tissue_type",
        required=True,
        choices=["wm", "gm"],
        help="wm for TBSS white matter skeleton, gm for GBSS gray matter skeleton",
    )
    parser.add_argument(
        "--output_xlsx",
        required=True,
        help="Output Excel file path",
    )
    return parser.parse_args()


def load_subject_order(merge_dir: Path) -> list[str]:
    candidates = [
        merge_dir / "subject_order.txt",
        merge_dir / "subject_session_order.txt",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    raise FileNotFoundError(
        f"No subject order file found in {merge_dir}. "
        f"Expected one of: {[str(p) for p in candidates]}"
    )


def get_metric_files(merge_dir: Path, tissue_type: str) -> dict[str, Path]:
    if tissue_type == "wm":
        suffix = "_skeletonised_TBSS.nii.gz"
    else:
        suffix = "_skeletonised_GBSS.nii.gz"

    metric_files = {
        "FA": merge_dir / "FA" / f"all_FA{suffix}",
        "MD": merge_dir / "MD" / f"all_MD{suffix}",
        "NDI": merge_dir / "NDI" / f"all_NDI{suffix}",
        "ODI": merge_dir / "ODI" / f"all_ODI{suffix}",
        "ISOVF": merge_dir / "ISOVF" / f"all_ISOVF{suffix}",
    }

    missing = [str(v) for v in metric_files.values() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            "Some skeleton files are missing:\n" + "\n".join(missing)
        )

    return metric_files


def main():
    args = parse_args()

    pipeline_output_dir = Path(args.pipeline_output_dir)
    merge_dir = pipeline_output_dir / "merged_4d"
    atlas_path = Path(args.atlas_nifti)
    output_xlsx = Path(args.output_xlsx)

    if not merge_dir.exists():
        raise FileNotFoundError(f"Merged 4D directory not found: {merge_dir}")
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

    subject_order = load_subject_order(merge_dir)
    metric_files = get_metric_files(merge_dir, args.tissue_type)

    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata(dtype=np.float32)

    if atlas_data.ndim != 3:
        raise ValueError(f"Atlas image must be 3D, got shape {atlas_data.shape}")

    atlas_labels = sorted([int(x) for x in np.unique(atlas_data) if x > 0])
    if len(atlas_labels) == 0:
        raise ValueError("Atlas contains no labels > 0")

    first_metric_img = nib.load(str(next(iter(metric_files.values()))))
    first_metric_data = first_metric_img.get_fdata(dtype=np.float32)

    if first_metric_data.shape[:3] != atlas_data.shape:
        raise ValueError(
            f"Spatial shape mismatch: skeleton {first_metric_data.shape[:3]} vs atlas {atlas_data.shape}"
        )

    n_subjects = len(subject_order)
    if first_metric_data.shape[3] != n_subjects:
        raise ValueError(
            f"Volume count mismatch: skeleton {first_metric_data.shape[3]} vs subject order {n_subjects}"
        )

    metric_arrays = {}
    for metric_name, metric_path in metric_files.items():
        img = nib.load(str(metric_path))
        data = img.get_fdata(dtype=np.float32)

        if data.ndim != 4:
            raise ValueError(f"{metric_name} image must be 4D, got shape {data.shape}")

        if data.shape[:3] != atlas_data.shape:
            raise ValueError(
                f"{metric_name} spatial shape mismatch: {data.shape[:3]} vs atlas {atlas_data.shape}"
            )

        if data.shape[3] != n_subjects:
            raise ValueError(
                f"{metric_name} volume count mismatch: {data.shape[3]} vs subject order {n_subjects}"
            )

        metric_arrays[metric_name] = data

    long_records = []

    for vol_idx, subject_id in enumerate(subject_order):
        for label in atlas_labels:
            roi_mask = atlas_data == label

            row = {
                "volume_index_0based": vol_idx,
                "volume_number_1based": vol_idx + 1,
                "subject_id": subject_id,
                "roi_label": label,
                "tissue_type": args.tissue_type,
            }

            for metric_name, data in metric_arrays.items():
                vol = data[..., vol_idx]
                valid_mask = roi_mask & (vol != 0)
                values = vol[valid_mask]

                row[metric_name] = float(np.mean(values)) if values.size > 0 else np.nan
                row[f"{metric_name}_nvox"] = int(values.size)

            long_records.append(row)

    long_df = pd.DataFrame(long_records)

    # Wide format: one row per subject, columns like FA_ROI_1, FA_ROI_2, ...
    wide_parts = []
    for metric_name in ["FA", "MD", "NDI", "ODI", "ISOVF"]:
        tmp = long_df.pivot(
            index="subject_id",
            columns="roi_label",
            values=metric_name,
        )
        tmp.columns = [f"{metric_name}_ROI_{int(c)}" for c in tmp.columns]
        wide_parts.append(tmp)

    wide_df = pd.concat(wide_parts, axis=1).reset_index()

    meta_df = pd.DataFrame(
        {
            "item": [
                "pipeline_output_dir",
                "merge_dir",
                "atlas_nifti",
                "tissue_type",
                "n_subjects",
                "n_rois",
                "value_definition",
            ],
            "value": [
                str(pipeline_output_dir),
                str(merge_dir),
                str(atlas_path),
                args.tissue_type,
                n_subjects,
                len(atlas_labels),
                "Mean of atlas-label ROI ∩ nonzero skeleton voxels for each subject and metric",
            ],
        }
    )

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="long_format", index=False)
        wide_df.to_excel(writer, sheet_name="wide_format", index=False)
        meta_df.to_excel(writer, sheet_name="meta", index=False)

    print("Finished.")
    print(f"Output xlsx: {output_xlsx}")
    print(f"Subjects: {n_subjects}")
    print(f"ROIs: {len(atlas_labels)}")


if __name__ == "__main__":
    main()