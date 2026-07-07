#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash fmriprep_single.sh <bids_dir> <subject_id> <session_id> [license_file]"
  exit 1
fi

bids_dir="$1"
subject_id="$2"
session_id="$3"
license_file="${4:-/home/lxg/license.txt}"

fmriprep_dir="${bids_dir}/derivatives/fmriprep"
work_dir="${bids_dir}/derivatives/workflows/sub-${subject_id}/ses-${session_id}"
fmriprep_func_dir="${fmriprep_dir}/sub-${subject_id}/ses-${session_id}/func"

template_space="MNI152NLin2009cAsym"
template_res="res-2"
template_spec="${template_space}:res-2"

anat_dir="${bids_dir}/sub-${subject_id}/ses-${session_id}/anat"
synth_t2w_file="${anat_dir}/sub-${subject_id}_ses-${session_id}_acq-synth_T2w.nii.gz"
synth_t2w_json="${anat_dir}/sub-${subject_id}_ses-${session_id}_acq-synth_T2w.json"
runtime_filter_file="${work_dir}/fmriprep_filter_synth_t2w.json"

mkdir -p "$fmriprep_dir" "$work_dir"

if [ ! -f "$license_file" ]; then
  echo "FreeSurfer license file not found: $license_file"
  exit 1
fi

if [ ! -d "$anat_dir" ]; then
  echo "Anat directory not found: $anat_dir"
  exit 1
fi

mni_bold_file=$(find "$fmriprep_func_dir" -maxdepth 1 -type f -name "*_task-rest*_space-${template_space}_${template_res}_desc-preproc_bold.nii.gz" 2>/dev/null | sort | head -n 1 || true)

# if [ -n "$mni_bold_file" ]; then
#   echo "MNI preprocessed BOLD already exists: $mni_bold_file"
#   echo "Skipping fMRIPrep."
#   exit 0
# fi

python3 - <<PY
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

bids_dir = Path("${bids_dir}")
subject_id = "${subject_id}"
session_id = "${session_id}"
anat_dir = Path("${anat_dir}")
out_file = Path("${synth_t2w_file}")
out_json = Path("${synth_t2w_json}")

if shutil.which("mri_synthstrip") is None:
    raise RuntimeError("Command not found: mri_synthstrip")

if shutil.which("N4BiasFieldCorrection") is None:
    raise RuntimeError("Command not found: N4BiasFieldCorrection")

candidates = sorted(anat_dir.glob(f"sub-{subject_id}_ses-{session_id}_acq-highres_T1w.nii.gz"))
if not candidates:
    candidates = sorted(anat_dir.glob(f"sub-{subject_id}_ses-{session_id}*_T1w.nii.gz"))

if not candidates:
    raise FileNotFoundError(f"No T1w file found in {anat_dir}")

t1_file = candidates[0]

if out_file.exists() and out_json.exists():
    print(f"Synth T2w already exists: {out_file}")
else:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mask_file = tmpdir / "brain_mask.nii.gz"
        n4_file = tmpdir / "t1_n4.nii.gz"

        subprocess.run([
            "mri_synthstrip",
            "-i", str(t1_file),
            "-m", str(mask_file),
        ], check=True)

        subprocess.run([
            "N4BiasFieldCorrection",
            "-d", "3",
            "-i", str(t1_file),
            "-x", str(mask_file),
            "-o", str(n4_file),
        ], check=True)

        img = nib.load(str(n4_file))
        data = img.get_fdata(dtype=np.float32)
        mask = nib.load(str(mask_file)).get_fdata() > 0

        if int(mask.sum()) == 0:
            raise RuntimeError(f"Empty brain mask: {t1_file}")

        brain = data[mask]
        p1, p99 = np.percentile(brain, [1, 99])

        if p99 <= p1:
            raise RuntimeError(f"Invalid intensity range: {t1_file}")

        x = np.clip((data - p1) / (p99 - p1), 0, 1)

        pseudo = np.zeros_like(x, dtype=np.float32)
        pseudo[mask] = 1.0 - x[mask]

        p = pseudo[mask]
        q1, q99 = np.percentile(p, [1, 99])

        if q99 > q1:
            pseudo = np.clip((pseudo - q1) / (q99 - q1), 0, 1)

        pseudo[~mask] = 0

        out_img = nib.Nifti1Image(pseudo.astype(np.float32), img.affine, img.header)
        out_img.header.set_data_dtype(np.float32)
        nib.save(out_img, str(out_file))

    try:
        source = str(t1_file.relative_to(bids_dir))
    except ValueError:
        source = str(t1_file)

    metadata = {
        "Modality": "MR",
        "ImageType": [
            "DERIVED",
            "SECONDARY",
            "SYNTHETIC"
        ],
        "ProtocolName": "pseudo_T2w_from_T1w",
        "SeriesDescription": "pseudo_T2w_from_T1w",
        "Sources": [
            source
        ],
        "Description": "Pseudo-T2w image generated from T1w by N4 correction, SynthStrip brain masking, robust intensity normalization, and brain-wise intensity inversion. This image is intended for exploratory BOLD-to-anatomical initialization."
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        f.write("\n")

    print(f"Generated synth T2w: {out_file}")
    print(f"Generated synth T2w JSON: {out_json}")
PY

cat > "$runtime_filter_file" <<'JSON'
{
  "t1w": {
    "datatype": "anat",
    "suffix": "T1w"
  },
  "t2w": {
    "datatype": "anat",
    "suffix": "T2w",
    "acquisition": "synth"
  },
  "bold": {
    "datatype": "func",
    "suffix": "bold",
    "task": "rest"
  }
}
JSON

docker run -ti --rm \
  -v "${bids_dir}:/data" \
  -v "${fmriprep_dir}:/out" \
  -v "${work_dir}:/work" \
  -v "${license_file}:/opt/freesurfer/license.txt" \
  -v "${runtime_filter_file}:/opt/fmri_filter.json" \
  nipreps/fmriprep:25.2.5 \
  /data /out \
  participant \
  --participant-label "$subject_id" \
  --session-label "$session_id" \
  -w /work \
  --nprocs 18 \
  --subject-anatomical-reference sessionwise \
  --use-syn-sdc \
  --fs-license-file /opt/freesurfer/license.txt \
  --bids-filter-file /opt/fmri_filter.json \
  --output-spaces func "$template_spec" \
  --skip-bids-validation \
  --ignore flair \
  --fs-no-reconall \
  --bold2anat-init t2w \
  --force no-bbr

mni_bold_file=$(find "$fmriprep_func_dir" -maxdepth 1 -type f -name "*_task-rest*_space-${template_space}_${template_res}_desc-preproc_bold.nii.gz" 2>/dev/null | sort | head -n 1 || true)

if [ -n "$mni_bold_file" ]; then
  echo "fMRIPrep completed. MNI BOLD output: $mni_bold_file"
else
  echo "fMRIPrep finished, but no MNI152NLin2009cAsym res-2 preprocessed BOLD file was found."
  exit 1
fi