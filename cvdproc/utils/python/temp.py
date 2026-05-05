import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
odi_4d_path = Path(r"/mnt/e/Neuroimage/workdir/TBSS_GBSS/ODI/all_ODI_skeletonised_GBSS.nii.gz")
odi_dir = Path(r"/mnt/e/Neuroimage/workdir/TBSS_GBSS/ODI")
atlas_path = Path(r"/mnt/e/Codes/cvdproc/cvdproc/data/atlas/desikan_killiany/atlas-desikankilliany_1mm_MNI152.nii.gz")
output_xlsx = Path(r"/mnt/e/Neuroimage/workdir/TBSS_GBSS/ODI/ODI_GBSS_DK_ROI.xlsx")

# Optional ROI lookup table.
# If this file does not exist, ROI names will be ROI_<label>.
roi_lookup_candidates = [
    Path(r"/mnt/e/Codes/cvdproc/cvdproc/data/atlas/desikan_killiany/atlas-desikankilliany_1mm_MNI152_labels.csv"),
    Path(r"/mnt/e/Codes/cvdproc/cvdproc/data/atlas/desikan_killiany/atlas-desikankilliany_1mm_MNI152.tsv"),
    Path(r"/mnt/e/Codes/cvdproc/cvdproc/data/atlas/desikan_killiany/atlas-desikankilliany_1mm_MNI152.txt"),
]

# =========================
# Helper functions
# =========================
def load_roi_lookup(candidates):
    for path in candidates:
        if not path.exists():
            continue

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            try:
                df = pd.read_csv(path, sep=None, engine="python")
            except Exception:
                continue

        lowered = {c.lower(): c for c in df.columns}
        possible_label_cols = ["label", "index", "id", "value"]
        possible_name_cols = ["name", "region", "roi", "label_name"]

        label_col = next((lowered[c] for c in possible_label_cols if c in lowered), None)
        name_col = next((lowered[c] for c in possible_name_cols if c in lowered), None)

        if label_col is None or name_col is None:
            continue

        mapping = {}
        for _, row in df.iterrows():
            try:
                label = int(row[label_col])
                name = str(row[name_col])
                mapping[label] = name
            except Exception:
                continue
        if mapping:
            return mapping, str(path)

    return {}, None


def get_subject_file_list(odi_directory):
    pattern = re.compile(
        r"^(sub-[^_]+_ses-[^_]+)_acq-DSIb4000_dir-AP_space-MNI152NLin6ASym_model-noddi_param-odi_dwimap\.nii\.gz$"
    )

    files = []
    for f in odi_directory.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            files.append((m.group(1), f.name, f))

    files = sorted(files, key=lambda x: x[1])
    return files


def check_shapes(img_4d, atlas_img):
    data_4d = img_4d.get_fdata(dtype=np.float32)
    atlas = atlas_img.get_fdata(dtype=np.float32)

    if data_4d.ndim != 4:
        raise ValueError(f"ODI image must be 4D, got shape {data_4d.shape}")
    if atlas.ndim != 3:
        raise ValueError(f"Atlas image must be 3D, got shape {atlas.shape}")
    if data_4d.shape[:3] != atlas.shape:
        raise ValueError(
            f"Spatial shape mismatch: ODI {data_4d.shape[:3]} vs atlas {atlas.shape}"
        )

    return data_4d, atlas


# =========================
# Main
# =========================
if not odi_4d_path.exists():
    raise FileNotFoundError(f"4D ODI file not found: {odi_4d_path}")
if not atlas_path.exists():
    raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
if not odi_dir.exists():
    raise FileNotFoundError(f"ODI directory not found: {odi_dir}")

subject_files = get_subject_file_list(odi_dir)
if len(subject_files) == 0:
    raise RuntimeError("No per-subject ODI files were found in the ODI directory.")

odi_img = nib.load(str(odi_4d_path))
atlas_img = nib.load(str(atlas_path))
odi_4d, atlas_data = check_shapes(odi_img, atlas_img)

n_volumes = odi_4d.shape[3]
if n_volumes != len(subject_files):
    raise ValueError(
        f"Volume count mismatch: 4D ODI has {n_volumes} volumes, "
        f"but found {len(subject_files)} per-subject ODI files. "
        "Please confirm the merge order and file filtering."
    )

atlas_int = np.rint(atlas_data).astype(np.int32)
roi_labels = sorted([x for x in np.unique(atlas_int) if x > 0])

roi_lookup, lookup_source = load_roi_lookup(roi_lookup_candidates)
roi_names = {
    label: roi_lookup.get(label, f"ROI_{label}")
    for label in roi_labels
}

mean_records = []
count_records = []
order_records = []

for vol_idx, (subject_session, filename, filepath) in enumerate(subject_files):
    vol = odi_4d[..., vol_idx]

    mean_row = {
        "volume_index_0based": vol_idx,
        "subject_session": subject_session,
        "source_file": filename,
    }
    count_row = {
        "volume_index_0based": vol_idx,
        "subject_session": subject_session,
        "source_file": filename,
    }

    for label in roi_labels:
        roi_mask = atlas_int == label
        roi_vals = vol[roi_mask]

        # GBSS skeletonized maps are sparse.
        # Use only non-zero voxels within each ROI.
        nz = roi_vals[roi_vals != 0]

        roi_col = roi_names[label]
        if nz.size == 0:
            mean_row[roi_col] = np.nan
            count_row[roi_col] = 0
        else:
            mean_row[roi_col] = float(np.mean(nz))
            count_row[roi_col] = int(nz.size)

    mean_records.append(mean_row)
    count_records.append(count_row)
    order_records.append(
        {
            "volume_index_0based": vol_idx,
            "volume_number_1based": vol_idx + 1,
            "subject_session": subject_session,
            "source_file": filename,
            "source_path": str(filepath),
        }
    )

mean_df = pd.DataFrame(mean_records)
count_df = pd.DataFrame(count_records)
order_df = pd.DataFrame(order_records)

meta_df = pd.DataFrame(
    {
        "item": [
            "odi_4d_path",
            "atlas_path",
            "n_volumes",
            "n_rois",
            "roi_lookup_source",
            "mean_definition",
        ],
        "value": [
            str(odi_4d_path),
            str(atlas_path),
            n_volumes,
            len(roi_labels),
            lookup_source if lookup_source is not None else "Not found; using ROI_<label>",
            "Mean ODI of non-zero skeleton voxels within each DK ROI",
        ],
    }
)

with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
    mean_df.to_excel(writer, sheet_name="ODI_mean", index=False)
    count_df.to_excel(writer, sheet_name="ROI_voxel_count", index=False)
    order_df.to_excel(writer, sheet_name="Subject_Order", index=False)
    meta_df.to_excel(writer, sheet_name="Meta", index=False)

print("Finished.")
print(f"Output xlsx: {output_xlsx}")
print(f"Volumes: {n_volumes}")
print(f"ROIs: {len(roi_labels)}")
if lookup_source is not None:
    print(f"ROI lookup loaded from: {lookup_source}")
else:
    print("ROI lookup not found. ROI columns are named as ROI_<label>.")