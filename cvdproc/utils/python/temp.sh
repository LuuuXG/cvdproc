#!/usr/bin/env bash
set -euo pipefail

xfm_root="/mnt/f/BIDS/WCH_AF_Project/derivatives/xfm"
qsiprep_root="/mnt/f/BIDS/WCH_AF_Project/derivatives/qsiprep"
dwi_root="/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline"

out_root="/mnt/e/Neuroimage/workdir/TBSS_GBSS"
gm_out_dir="$out_root/GM_fraction"
tmp_root="$out_root/_tmp_gbss_preprocess"
log_dir="$out_root/logs"

gbss_script="/mnt/e/codes/cvdproc/cvdproc/pipelines/bash/tbss_and_gbss/gbss_preprocess.sh"

threads=16

mkdir -p "$gm_out_dir" "$tmp_root" "$log_dir"

process_one() {
    local sub="$1"
    local ses="$2"

    local xfm_dir="$xfm_root/$sub/$ses"

    local dwi_mask="$qsiprep_root/$sub/$ses/dwi/${sub}_${ses}_acq-DSIb4000_dir-AP_space-ACPC_desc-brain_mask.nii.gz"
    local fa_image="$dwi_root/$sub/$ses/dtifit/${sub}_${ses}_acq-DSIb4000_dir-AP_space-ACPC_model-tensor_param-fa_dwimap.nii.gz"
    local iso_image="$dwi_root/$sub/$ses/NODDI/${sub}_${ses}_acq-DSIb4000_dir-AP_space-ACPC_model-noddi_param-isovf_dwimap.nii.gz"

    local acpc_to_t1_mat="$xfm_dir/${sub}_${ses}_from-ACPC_to-T1w_xfm.mat"
    local t1_ref="$xfm_dir/${sub}_${ses}_acq-highres_desc-brain_T1w.nii.gz"
    local t1_to_mni_warp="$xfm_dir/${sub}_${ses}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz"

    local missing=0
    local f
    for f in \
        "$dwi_mask" \
        "$fa_image" \
        "$iso_image" \
        "$acpc_to_t1_mat" \
        "$t1_ref" \
        "$t1_to_mni_warp" \
        "$gbss_script"; do
        if [[ ! -f "$f" ]]; then
            echo "SKIP ${sub} ${ses} : missing $f"
            missing=1
        fi
    done

    if [[ "$missing" -ne 0 ]]; then
        return 0
    fi

    local work_dir="$tmp_root/${sub}_${ses}"
    mkdir -p "$work_dir"

    local pseudo_t1_name="${sub}_${ses}_acq-DSIb4000_dir-AP_space-ACPC_desc-pseudoT1w.nii.gz"
    local gm_name_acpc="${sub}_${ses}_acq-DSIb4000_dir-AP_space-ACPC_label-GM_probability.nii.gz"
    local gm_name_t1="${sub}_${ses}_acq-DSIb4000_dir-AP_space-T1w_label-GM_probability.nii.gz"
    local gm_name_mni="${sub}_${ses}_acq-DSIb4000_dir-AP_space-MNI152NLin6ASym_label-GM_probability.nii.gz"

    local gm_acpc="$work_dir/$gm_name_acpc"
    local gm_t1="$work_dir/$gm_name_t1"
    local gm_mni="$gm_out_dir/$gm_name_mni"

    echo "START ${sub} ${ses}"

    bash "$gbss_script" \
        "$dwi_mask" \
        "$fa_image" \
        "$iso_image" \
        "$work_dir" \
        "$pseudo_t1_name" \
        "$gm_name_acpc"

    if [[ ! -f "$gm_acpc" ]]; then
        echo "ERROR ${sub} ${ses} : GBSS preprocess did not generate $gm_acpc"
        return 1
    fi

    flirt \
        -in "$gm_acpc" \
        -ref "$t1_ref" \
        -out "$gm_t1" \
        -applyxfm \
        -init "$acpc_to_t1_mat" \
        -interp trilinear

    mri_convert -at "$t1_to_mni_warp" "$gm_t1" "$gm_mni"

    echo "DONE ${sub} ${ses}"
}

export xfm_root
export qsiprep_root
export dwi_root
export out_root
export gm_out_dir
export tmp_root
export log_dir
export gbss_script
export -f process_one

task_list="$log_dir/gm_fraction_subject_session_list.txt"
: > "$task_list"

for sub_dir in "$xfm_root"/sub-*; do
    [[ -d "$sub_dir" ]] || continue
    sub=$(basename "$sub_dir")

    for ses_dir in "$sub_dir"/ses-*; do
        [[ -d "$ses_dir" ]] || continue
        ses=$(basename "$ses_dir")
        echo "$sub $ses" >> "$task_list"
    done
done

cat "$task_list" | xargs -n 2 -P "$threads" bash -c 'process_one "$1" "$2"' _

gm_n=$(find "$gm_out_dir" -maxdepth 1 -type f -name "*.nii.gz" | wc -l)

echo "Done."
echo "Final GM_fraction file count: $gm_n"
echo "Output directory: $gm_out_dir"