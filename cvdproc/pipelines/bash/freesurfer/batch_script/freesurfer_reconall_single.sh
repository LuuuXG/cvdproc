#!/bin/bash
set -euo pipefail

bids_dir=$1     # BIDS root directory
subject_id=$2   # Subject ID (e.g., HC0001)
session_id=$3   # Session ID (e.g., baseline)

#######################
# Prepare directories #
#######################
subject="sub-${subject_id}"
session="ses-${session_id}"
freesurfer_subjects_dir="${bids_dir}/derivatives/freesurfer/${subject}"
mkdir -p "${freesurfer_subjects_dir}"

xfm_dir="${bids_dir}/derivatives/xfm/${subject}/${session}"
mkdir -p "${xfm_dir}"

export SUBJECTS_DIR="${freesurfer_subjects_dir}"

anat_dir="${bids_dir}/${subject}/${session}/anat"

# T1w: match anything ending with _T1w.nii.gz
t1w_image=$(find "$anat_dir" -maxdepth 1 -type f -name "${subject}_${session}_acq-highres_desc-lesionfilled_T1w.nii.gz" | head -n 1 || true)

if [[ -z "$t1w_image" ]]; then
    echo "[ERROR] No T1w image found for ${subject} ${session} in ${anat_dir}"
    exit 1
fi
echo "[INFO] Found T1w image: $t1w_image"

# Check if have brain mask
brain_mask="${xfm_dir}/${subject}_${session}_space-T1w_desc-brain_mask.nii.gz"
if [[ ! -f "$brain_mask" ]]; then
    echo "[ERROR] Brain mask not found: $brain_mask"
    mri_synthstrip -i "$t1w_image" -m "$brain_mask" --no-csf
    #mri_synthstrip -i "$t1w_image" -m "$brain_mask"
fi

##############################
# Main FreeSurfer processing #
##############################

# 01: recon-all
reconall_file_to_check="${freesurfer_subjects_dir}/${session}/surf/lh.area.fwhm25.fsaverage.mgh"

if [[ ! -f "$reconall_file_to_check" ]]; then
    echo "[INFO] Running recon-all for ${subject} ${session} (with -i T1w)"
    if recon-all -i "$t1w_image" -s "$session" -all -qcache -sd "$freesurfer_subjects_dir" -no-isrunning -xmask "$brain_mask"; then
        echo "[INFO] recon-all with -i succeeded."
    else
        echo "[WARN] recon-all with -i failed, trying resume without -i ..."
        recon-all -s "$session" -all -qcache -sd "$freesurfer_subjects_dir" -no-isrunning -xmask "$brain_mask"
    fi
else
    echo "[INFO] recon-all already completed for ${subject} ${session}, skipping."
fi
##########################################
# Subfield segmentation with MCR runtime #
##########################################

# # Create subject/session-specific MCR cache
# export MCR_CACHE_ROOT="/tmp/mcr_cache_${subject}_${session}_$$"
# mkdir -p "$MCR_CACHE_ROOT"

# cleanup_mcr() {
#     rm -rf "$MCR_CACHE_ROOT"
# }
# trap cleanup_mcr EXIT

# # 02: hippocampal subfields
# ha_file_to_check="${freesurfer_subjects_dir}/${session}/mri/lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz"
# if [[ ! -f "$ha_file_to_check" ]]; then
#     echo "[INFO] Running hippocampal subfield segmentation for ${subject} ${session}"
#     segmentHA_T1.sh "$session" "$freesurfer_subjects_dir"
# else
#     echo "[INFO] Hippocampal subfield segmentation already completed for ${subject} ${session}, skipping."
# fi

# # 03: brainstem
# bs_file_to_check="${freesurfer_subjects_dir}/${session}/mri/brainstemSsLabels.v13.FSvoxelSpace.mgz"
# if [[ ! -f "$bs_file_to_check" ]]; then
#     echo "[INFO] Running brainstem segmentation for ${subject} ${session}"
#     segmentBS.sh "$session" "$freesurfer_subjects_dir"
# else
#     echo "[INFO] Brainstem segmentation already completed for ${subject} ${session}, skipping."
# fi

# # 04: thalamic nuclei
# thalamic_file_to_check="${freesurfer_subjects_dir}/${session}/mri/ThalamicNuclei.v13.T1.FSvoxelSpace.mgz"
# if [[ ! -f "$thalamic_file_to_check" ]]; then
#     echo "[INFO] Running thalamic nuclei segmentation for ${subject} ${session}"
#     segmentThalamicNuclei.sh "$session" "$freesurfer_subjects_dir"
# else
#     echo "[INFO] Thalamic nuclei segmentation already completed for ${subject} ${session}, skipping."
# fi

# # 05: hypothalamic subunits
# hypothalamic_file_to_check="${freesurfer_subjects_dir}/${session}/mri/hypothalamic_subunits_seg.v1.mgz"
# if [[ ! -f "$hypothalamic_file_to_check" ]]; then
#     echo "[INFO] Running hypothalamic subunits segmentation for ${subject} ${session}"
#     mri_segment_hypothalamic_subunits --s "$session" --sd "$freesurfer_subjects_dir"
# else
#     echo "[INFO] Hypothalamic subunits segmentation already completed for ${subject} ${session}, skipping."
# fi

# echo "[INFO] All processing completed for ${subject} ${session}"
