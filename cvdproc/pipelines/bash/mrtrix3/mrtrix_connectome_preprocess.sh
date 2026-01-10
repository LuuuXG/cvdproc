#!/usr/bin/env bash

set -euo pipefail

# ------------------------ Usage & argument parsing ------------------------ #

if [ "$#" -ne 19 ]; then
  echo "Usage: $0 \\"
  echo "  <preproc_dwi.mif> <dwi_mask.mif> <fs_to_dwi.mat> <output_dir> <aseg.nii.gz> \\"
  echo "  <wm_response.mif> <wm_fod.mif> <wm_fod_norm.mif> \\"
  echo "  <gm_response.mif> <gm_fod.mif> <gm_fod_norm.mif> \\"
  echo "  <csf_response.mif> <csf_fod.mif> <csf_fod_norm.mif> \\"
  echo "  <sift_mu.txt> <aseg_dwi.nii.gz> <five_tt_dwi.mif> <gmwmSeed_dwi.mif> <streamlines.tck> <sift_weights.txt>"
  exit 1
fi

preproc_dwi_mif=$1
dwi_mask_mif=$2
output_dir=$3
aseg_dwi_nifti=$4

output_wm_response=$5
output_wm_fod=$6
output_wm_fod_norm=$7

output_gm_response=${8}
output_gm_fod=${9}
output_gm_fod_norm=${10}

output_csf_response=${11}
output_csf_fod=${12}
output_csf_fod_norm=${13}
output_sift_mu=${14}
five_tt_dwi=${15}
gmwmSeed_dwi=${16}
streamlines_tck=${17}
output_sift_weights=${18}

mkdir -p "${output_dir}"

# Resolve full paths for MRtrix outputs
wm_resp_path="${output_dir}/${output_wm_response}"
wm_fod_path="${output_dir}/${output_wm_fod}"
wm_fod_norm_path="${output_dir}/${output_wm_fod_norm}"

gm_resp_path="${output_dir}/${output_gm_response}"
gm_fod_path="${output_dir}/${output_gm_fod}"
gm_fod_norm_path="${output_dir}/${output_gm_fod_norm}"

csf_resp_path="${output_dir}/${output_csf_response}"
csf_fod_path="${output_dir}/${output_csf_fod}"
csf_fod_norm_path="${output_dir}/${output_csf_fod_norm}"

sift_mu_path="${output_dir}/${output_sift_mu}"
aseg_dwi_path="${output_dir}/${output_aseg_dwi}"
five_tt_dwi_path="${output_dir}/${five_tt_dwi}"
gmwmSeed_dwi_path="${output_dir}/${gmwmSeed_dwi}"
streamlines_tck_path="${output_dir}/${streamlines_tck}"
sift_weights_path="${output_dir}/${output_sift_weights}"

# ------------------------ Helper functions ------------------------ #

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Command '$1' not found in PATH."
    exit 1
  fi
}

# Count unique non-zero shells using mrinfo -shell_bvalues
count_unique_nonzero_shells_from_mif() {
  local mif_file=$1
  local shell_bvals
  shell_bvals=$(mrinfo "${mif_file}" -shell_bvalues 2>/dev/null || true)

  if [ -z "${shell_bvals}" ]; then
    echo "[ERROR] Failed to read shell b-values from MIF: ${mif_file}" >&2
    exit 1
  fi

  echo "${shell_bvals}" | awk '
    {
      for (i=1; i<=NF; i++) {
        x=$i
        gsub(/[^0-9.]/, "", x)
        if (x != "" && x+0 > 50) a[int(x+0)] = 1
      }
    }
    END {
      n = 0
      for (k in a) n++
      print n
    }'
}

# ------------------------ Check dependencies ------------------------ #

for cmd in mrinfo dwi2response dwi2fod mtnormalise flirt 5ttgen 5tt2gmwmi tckgen tcksift2; do
  check_cmd "${cmd}"
done

# ------------------------ Basic input checks ------------------------ #

if [ ! -f "${preproc_dwi_mif}" ]; then
  echo "[ERROR] preproc_dwi.mif not found: ${preproc_dwi_mif}"
  exit 1
fi

if [ ! -f "${dwi_mask_mif}" ]; then
  echo "[ERROR] dwi_mask.mif not found: ${dwi_mask_mif}"
  exit 1
fi

if [ ! -f "${aseg_dwi_nifti}" ]; then
  echo "[ERROR] aseg.nii.gz not found: ${aseg_dwi_nifti}"
  exit 1
fi

# ------------------------ Check outputs ------------------------ #
# streamline and sift weights
if [ -f "${streamlines_tck_path}" ] && [ -f "${sift_weights_path}" ]; then
  log "Outputs already exist. Exiting."
  exit 0
fi

# ------------------------ Detect shell type ------------------------ #

unique_nonzero_shells=$(count_unique_nonzero_shells_from_mif "${preproc_dwi_mif}")
log "Number of non-zero unique shells (b>50): ${unique_nonzero_shells}"

# ------------------------ Response functions & FODs ------------------------ #

if [ "${unique_nonzero_shells}" -gt 1 ]; then
  log "Detected multi-shell data -> using dwi2response dhollander + msmt_csd."

  # WM / GM / CSF response
  dwi2response dhollander \
    "${preproc_dwi_mif}" \
    "${wm_resp_path}" \
    "${gm_resp_path}" \
    "${csf_resp_path}" \
    -mask "${dwi_mask_mif}" \
    -nthreads 4 \
    -force

  # Multi-shell multi-tissue CSD
  dwi2fod msmt_csd \
    "${preproc_dwi_mif}" \
    -mask "${dwi_mask_mif}" \
    "${wm_resp_path}"  "${wm_fod_path}" \
    "${gm_resp_path}"  "${gm_fod_path}" \
    "${csf_resp_path}" "${csf_fod_path}" \
    -nthreads 4 \
    -force

  # Multi-tissue normalisation (WM + GM + CSF)
  log "Running mtnormalise for WM/GM/CSF FODs."
  mtnormalise \
    "${wm_fod_path}"  "${wm_fod_norm_path}" \
    "${gm_fod_path}"  "${gm_fod_norm_path}" \
    "${csf_fod_path}" "${csf_fod_norm_path}" \
    -mask "${dwi_mask_mif}" \
    -force

else
  log "Detected single-shell data -> using dwi2response tournier + single-shell CSD (WM only)."
  log "GM/CSF response and FOD outputs will NOT be generated in this mode."

  # Single-shell WM response
  dwi2response tournier \
    "${preproc_dwi_mif}" \
    "${wm_resp_path}" \
    -mask "${dwi_mask_mif}" \
    -nthreads 4 \
    -force

  # Single-shell CSD (WM only)
  dwi2fod csd \
    "${preproc_dwi_mif}" \
    "${wm_resp_path}" \
    "${wm_fod_path}" \
    -mask "${dwi_mask_mif}" \
    -nthreads 4 \
    -force

  # Normalisation only for WM FOD
  log "Running mtnormalise for WM FOD only."
  mtnormalise \
    "${wm_fod_path}" "${wm_fod_norm_path}" \
    -mask "${dwi_mask_mif}" \
    -force
fi

# ------------------------ Segmentation to DWI space & ACT preparation ------------------------ #

# log "Transforming aseg to DWI space with FLIRT."
# flirt \
#   -in "${aseg_nifti}" \
#   -ref "${preproc_dwiref}" \
#   -out "${aseg_dwi_path}" \
#   -applyxfm -init "${fs_to_dwi_mat}" \
#   -interp nearestneighbour

log "Generating 5TT image from aseg (FreeSurfer-based)."
5ttgen freesurfer \
  "${aseg_dwi_nifti}" \
  "${five_tt_dwi_path}" \
  -force

log "Generating GM-WM interface (gmwmi) seed."
5tt2gmwmi \
  "${five_tt_dwi_path}" \
  "${gmwmSeed_dwi_path}" \
  -force

# ------------------------ Tractography & SIFT2 ------------------------ #

log "Running ACT-based tractography with tckgen."
tckgen \
  -act "${five_tt_dwi_path}" \
  -backtrack \
  -seed_gmwmi "${gmwmSeed_dwi_path}" \
  -select 1000000 \
  -cutoff 0.06 \
  -maxlength 250 \
  -nthreads 4 \
  "${wm_fod_norm_path}" \
  "${streamlines_tck_path}" \
  -force

log "Running SIFT2."
tcksift2 \
  -act "${five_tt_dwi_path}" \
  -out_mu "${sift_mu_path}" \
  -nthreads 4 \
  "${streamlines_tck_path}" \
  "${wm_fod_norm_path}" \
  "${sift_weights_path}" \
  -force

log "Pipeline finished successfully."
