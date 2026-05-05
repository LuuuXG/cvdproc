#!/usr/bin/env bash

set -euo pipefail

bids_dir="$1"
output_dir="$2"
threads="${3:-16}"

dwi_pipeline_dir="$bids_dir/derivatives/dwi_pipeline"
qsiprep_dir="$bids_dir/derivatives/qsiprep"
xfm_dir="$bids_dir/derivatives/xfm"

mkdir -p "$output_dir"

raw_img_dir="$output_dir/raw_images"
t1w_img_dir="$output_dir/t1w_space_images"
mni_img_dir="$output_dir/mni_space_images"
mean_img_dir="$output_dir/mean_images"
skeleton_dir="$output_dir/skeleton_images"
tmp_dir="$output_dir/tmp"
log_dir="$output_dir/logs"
merge_dir="$output_dir/merged_4d"

mkdir -p "$mean_img_dir" "$tmp_dir" "$skeleton_dir" "$log_dir" "$merge_dir"
mkdir -p "$raw_img_dir"/{FA,MD,NDI,ODI,ISOVF,GM_fraction,WM_fraction,pseudoT1w}
mkdir -p "$t1w_img_dir"/{FA,MD,NDI,ODI,ISOVF,GM_fraction,WM_fraction,pseudoT1w}
mkdir -p "$mni_img_dir"/{FA,MD,NDI,ODI,ISOVF,GM_fraction,WM_fraction,pseudoT1w}
mkdir -p "$merge_dir"/{FA,MD,NDI,ODI,ISOVF,GM_fraction,WM_fraction,pseudoT1w}
mkdir -p "$skeleton_dir"/{WM_skeleton,GM_skeleton}

get_first_match() {
    local search_dir="$1"
    local pattern="$2"
    if [[ ! -d "$search_dir" ]]; then
        echo ""
        return 0
    fi
    find "$search_dir" -type f -name "$pattern" | sort | head -n 1
}

process_one_subject_session() {
    local subject_id="$1"
    local session_id="$2"

    local dwi_pipeline_single_dir="$dwi_pipeline_dir/$subject_id/$session_id"
    local xfm_single_dir="$xfm_dir/$subject_id/$session_id"
    local qsiprep_single_dir="$qsiprep_dir/$subject_id/$session_id/dwi"
    local tmp_single_dir="$tmp_dir/$subject_id/$session_id"

    mkdir -p "$tmp_single_dir"

    echo "Processing: $subject_id $session_id"

    local fa_img
    local md_img
    local ndi_img
    local odi_img
    local iso_img
    local dwi_mask1_img
    local dwi_mask2_img
    local dwi_mask_img
    local space_entity
    local base_prefix
    local dwi_to_t1w_xfm
    local t1_to_mni_warp
    local t1w_ref

    fa_img="$(get_first_match "$dwi_pipeline_single_dir/dtifit" "*fa_dwimap.nii.gz")"
    md_img="$(get_first_match "$dwi_pipeline_single_dir/dtifit" "*md_dwimap.nii.gz")"
    ndi_img="$(get_first_match "$dwi_pipeline_single_dir/NODDI" "*icvf_dwimap.nii.gz")"
    odi_img="$(get_first_match "$dwi_pipeline_single_dir/NODDI" "*odi_dwimap.nii.gz")"
    iso_img="$(get_first_match "$dwi_pipeline_single_dir/NODDI" "*isovf_dwimap.nii.gz")"

    dwi_mask1_img="$(get_first_match "$qsiprep_single_dir" "*_desc-brain_mask.nii.gz")"
    dwi_mask2_img="$(get_first_match "$dwi_pipeline_single_dir" "*_desc-brain_mask.nii.gz")"

    if [[ -n "$dwi_mask1_img" ]]; then
        dwi_mask_img="$dwi_mask1_img"
    elif [[ -n "$dwi_mask2_img" ]]; then
        dwi_mask_img="$dwi_mask2_img"
    else
        echo "Skip $subject_id $session_id: no brain mask found"
        return 0
    fi

    if [[ -z "$fa_img" || -z "$md_img" || -z "$ndi_img" || -z "$odi_img" || -z "$iso_img" ]]; then
        echo "Skip $subject_id $session_id: missing one or more diffusion maps"
        return 0
    fi

    space_entity="$(basename "$fa_img" | sed -E 's/.*_space-([^_]+).*/\1/')"
    base_prefix="$(basename "$fa_img" .nii.gz | sed -E 's/_space-[^_]+.*$//')"

    dwi_to_t1w_xfm="$(get_first_match "$xfm_single_dir" "*from-${space_entity}_to-T1w_xfm.mat")"
    t1_to_mni_warp="$(get_first_match "$xfm_single_dir" "*from-T1w_to-MNI152NLin6ASym_warp.nii.gz")"
    t1w_ref="$(get_first_match "$xfm_single_dir" "*desc-brain_T1w.nii.gz")"

    if [[ -z "$dwi_to_t1w_xfm" || -z "$t1_to_mni_warp" || -z "$t1w_ref" ]]; then
        echo "Skip $subject_id $session_id: missing transform files"
        return 0
    fi

    local atropos_seg
    local atropos_prob01
    local atropos_prob02
    local wm_prob_raw
    local gm_prob_raw
    local pseudo_t1_raw
    local wm_largest_component
    local wm_rim
    local wm_con
    local gm_con
    local fa_mean_1
    local fa_mean_2
    local wm_prob_source

    atropos_seg="$tmp_single_dir/${base_prefix}_space-${space_entity}_desc-atropos_seg.nii.gz"
    atropos_prob01="$tmp_single_dir/${base_prefix}_space-${space_entity}_desc-atropos_prob01.nii.gz"
    atropos_prob02="$tmp_single_dir/${base_prefix}_space-${space_entity}_desc-atropos_prob02.nii.gz"

    wm_prob_raw="$raw_img_dir/WM_fraction/${base_prefix}_space-${space_entity}_label-WM_probability.nii.gz"
    gm_prob_raw="$raw_img_dir/GM_fraction/${base_prefix}_space-${space_entity}_label-GM_probability.nii.gz"
    pseudo_t1_raw="$raw_img_dir/pseudoT1w/${base_prefix}_space-${space_entity}_desc-pseudoT1w_T1w.nii.gz"

    wm_largest_component="$tmp_single_dir/${base_prefix}_space-${space_entity}_desc-wm_largest_component.nii.gz"
    wm_rim="$tmp_single_dir/${base_prefix}_space-${space_entity}_desc-wm_rim.nii.gz"
    wm_con="$tmp_single_dir/${base_prefix}_space-${space_entity}_desc-wm_con.nii.gz"
    gm_con="$tmp_single_dir/${base_prefix}_space-${space_entity}_desc-gm_con.nii.gz"

    Atropos \
      -d 3 \
      -a "$fa_img" \
      -x "$dwi_mask_img" \
      -i Kmeans[2] \
      -m [0.3,1x1x1] \
      -o ["$atropos_seg","$tmp_single_dir/${base_prefix}_space-${space_entity}_desc-atropos_prob%02d.nii.gz"]

    fa_mean_1="$(fslstats "$fa_img" -k "$atropos_prob01" -M)"
    fa_mean_2="$(fslstats "$fa_img" -k "$atropos_prob02" -M)"

    if (( $(echo "$fa_mean_1 > $fa_mean_2" | bc -l) )); then
        wm_prob_source="$atropos_prob01"
    else
        wm_prob_source="$atropos_prob02"
    fi

    cp "$wm_prob_source" "$wm_prob_raw"

    fslmaths "$wm_prob_raw" \
      -add "$iso_img" \
      -sub 1 -mul -1 \
      -thr 0 \
      -mul "$dwi_mask_img" \
      "$gm_prob_raw"

    ImageMath 3 "$wm_largest_component" \
      GetLargestComponent "$wm_prob_raw"

    fslmaths "$wm_prob_raw" \
      -bin \
      -sub "$wm_largest_component" \
      -thr 0 -bin \
      "$wm_rim"

    fslmaths "$wm_prob_raw" \
      -mul "$wm_largest_component" \
      -mul 2 \
      "$wm_con"

    fslmaths "$gm_prob_raw" \
      -thr 0 \
      -mul 1 \
      "$gm_con"

    fslmaths "$gm_con" \
      -add "$wm_con" \
      -mul "$dwi_mask_img" \
      "$pseudo_t1_raw"

    cp "$fa_img" "$raw_img_dir/FA/"
    cp "$md_img" "$raw_img_dir/MD/"
    cp "$ndi_img" "$raw_img_dir/NDI/"
    cp "$odi_img" "$raw_img_dir/ODI/"
    cp "$iso_img" "$raw_img_dir/ISOVF/"

    local fa_t1w
    local md_t1w
    local ndi_t1w
    local odi_t1w
    local iso_t1w
    local wm_t1w
    local gm_t1w
    local pseudo_t1w

    fa_t1w="$t1w_img_dir/FA/${base_prefix}_space-T1w_model-tensor_param-fa_dwimap.nii.gz"
    md_t1w="$t1w_img_dir/MD/${base_prefix}_space-T1w_model-tensor_param-md_dwimap.nii.gz"
    ndi_t1w="$t1w_img_dir/NDI/${base_prefix}_space-T1w_model-noddi_param-icvf_dwimap.nii.gz"
    odi_t1w="$t1w_img_dir/ODI/${base_prefix}_space-T1w_model-noddi_param-odi_dwimap.nii.gz"
    iso_t1w="$t1w_img_dir/ISOVF/${base_prefix}_space-T1w_model-noddi_param-isovf_dwimap.nii.gz"
    wm_t1w="$t1w_img_dir/WM_fraction/${base_prefix}_space-T1w_label-WM_probability.nii.gz"
    gm_t1w="$t1w_img_dir/GM_fraction/${base_prefix}_space-T1w_label-GM_probability.nii.gz"
    pseudo_t1w="$t1w_img_dir/pseudoT1w/${base_prefix}_space-T1w_desc-pseudoT1w_T1w.nii.gz"

    flirt -in "$fa_img" -ref "$t1w_ref" -out "$fa_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$md_img" -ref "$t1w_ref" -out "$md_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$ndi_img" -ref "$t1w_ref" -out "$ndi_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$odi_img" -ref "$t1w_ref" -out "$odi_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$iso_img" -ref "$t1w_ref" -out "$iso_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$wm_prob_raw" -ref "$t1w_ref" -out "$wm_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$gm_prob_raw" -ref "$t1w_ref" -out "$gm_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$pseudo_t1_raw" -ref "$t1w_ref" -out "$pseudo_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear

    local fa_mni
    local md_mni
    local ndi_mni
    local odi_mni
    local iso_mni
    local wm_mni
    local gm_mni
    local pseudo_mni

    fa_mni="$mni_img_dir/FA/${base_prefix}_space-MNI152NLin6ASym_model-tensor_param-fa_dwimap.nii.gz"
    md_mni="$mni_img_dir/MD/${base_prefix}_space-MNI152NLin6ASym_model-tensor_param-md_dwimap.nii.gz"
    ndi_mni="$mni_img_dir/NDI/${base_prefix}_space-MNI152NLin6ASym_model-noddi_param-icvf_dwimap.nii.gz"
    odi_mni="$mni_img_dir/ODI/${base_prefix}_space-MNI152NLin6ASym_model-noddi_param-odi_dwimap.nii.gz"
    iso_mni="$mni_img_dir/ISOVF/${base_prefix}_space-MNI152NLin6ASym_model-noddi_param-isovf_dwimap.nii.gz"
    wm_mni="$mni_img_dir/WM_fraction/${base_prefix}_space-MNI152NLin6ASym_label-WM_probability.nii.gz"
    gm_mni="$mni_img_dir/GM_fraction/${base_prefix}_space-MNI152NLin6ASym_label-GM_probability.nii.gz"
    pseudo_mni="$mni_img_dir/pseudoT1w/${base_prefix}_space-MNI152NLin6ASym_desc-pseudoT1w_T1w.nii.gz"

    mri_convert -at "$t1_to_mni_warp" "$fa_t1w" "$fa_mni"
    mri_convert -at "$t1_to_mni_warp" "$md_t1w" "$md_mni"
    mri_convert -at "$t1_to_mni_warp" "$ndi_t1w" "$ndi_mni"
    mri_convert -at "$t1_to_mni_warp" "$odi_t1w" "$odi_mni"
    mri_convert -at "$t1_to_mni_warp" "$iso_t1w" "$iso_mni"
    mri_convert -at "$t1_to_mni_warp" "$wm_t1w" "$wm_mni"
    mri_convert -at "$t1_to_mni_warp" "$gm_t1w" "$gm_mni"
    mri_convert -at "$t1_to_mni_warp" "$pseudo_t1w" "$pseudo_mni"

    echo "Done: $subject_id $session_id"
}

export bids_dir
export output_dir
export threads
export dwi_pipeline_dir
export qsiprep_dir
export xfm_dir
export raw_img_dir
export t1w_img_dir
export mni_img_dir
export mean_img_dir
export skeleton_dir
export tmp_dir
export log_dir
export merge_dir
export -f get_first_match
export -f process_one_subject_session

echo "===== Part 1: Prepare Scalar Images ====="

task_list="$log_dir/subject_session_tasks.txt"
: > "$task_list"

for sub_dir in "$dwi_pipeline_dir"/sub-*; do
    [[ -d "$sub_dir" ]] || continue
    subject_id="$(basename "$sub_dir")"

    for ses_dir in "$sub_dir"/ses-*; do
        [[ -d "$ses_dir" ]] || continue
        session_id="$(basename "$ses_dir")"
        echo "$subject_id $session_id" >> "$task_list"
    done
done

cat "$task_list" | xargs -n 2 -P "$threads" bash -c 'process_one_subject_session "$1" "$2"' _

echo "===== Part 2: Merge 4D Images and Calculate Mean Images ====="

extract_ids() {
    local input_dir="$1"
    local output_txt="$2"
    find "$input_dir" -maxdepth 1 -type f -name "*.nii.gz" -printf "%f\n" \
        | sed -E 's/_acq-.*$//' \
        | sort -u > "$output_txt"
}

fa_mni_dir="$mni_img_dir/FA"
md_mni_dir="$mni_img_dir/MD"
ndi_mni_dir="$mni_img_dir/NDI"
odi_mni_dir="$mni_img_dir/ODI"
isovf_mni_dir="$mni_img_dir/ISOVF"
gm_mni_dir="$mni_img_dir/GM_fraction"

extract_ids "$fa_mni_dir" "$log_dir/fa_ids.txt"
extract_ids "$md_mni_dir" "$log_dir/md_ids.txt"
extract_ids "$ndi_mni_dir" "$log_dir/ndi_ids.txt"
extract_ids "$odi_mni_dir" "$log_dir/odi_ids.txt"
extract_ids "$isovf_mni_dir" "$log_dir/isovf_ids.txt"
extract_ids "$gm_mni_dir" "$log_dir/gm_ids.txt"

comm -12 "$log_dir/fa_ids.txt" "$log_dir/md_ids.txt" > "$log_dir/_tmp1.txt"
comm -12 "$log_dir/_tmp1.txt" "$log_dir/ndi_ids.txt" > "$log_dir/_tmp2.txt"
comm -12 "$log_dir/_tmp2.txt" "$log_dir/odi_ids.txt" > "$log_dir/_tmp3.txt"
comm -12 "$log_dir/_tmp3.txt" "$log_dir/isovf_ids.txt" > "$log_dir/_tmp4.txt"
comm -12 "$log_dir/_tmp4.txt" "$log_dir/gm_ids.txt" > "$log_dir/common_ids.txt"

subject_order="$merge_dir/subject_session_order.txt"
: > "$subject_order"

fa_list="$log_dir/fa_merge_list.txt"
md_list="$log_dir/md_merge_list.txt"
ndi_list="$log_dir/ndi_merge_list.txt"
odi_list="$log_dir/odi_merge_list.txt"
isovf_list="$log_dir/isovf_merge_list.txt"
gm_list="$log_dir/gm_merge_list.txt"

: > "$fa_list"
: > "$md_list"
: > "$ndi_list"
: > "$odi_list"
: > "$isovf_list"
: > "$gm_list"

while IFS= read -r id; do
    fa_file="$(find "$fa_mni_dir" -maxdepth 1 -type f -name "${id}_acq-*_space-MNI152NLin6ASym_model-tensor_param-fa_dwimap.nii.gz" | sort | head -n 1)"
    md_file="$(find "$md_mni_dir" -maxdepth 1 -type f -name "${id}_acq-*_space-MNI152NLin6ASym_model-tensor_param-md_dwimap.nii.gz" | sort | head -n 1)"
    ndi_file="$(find "$ndi_mni_dir" -maxdepth 1 -type f -name "${id}_acq-*_space-MNI152NLin6ASym_model-noddi_param-icvf_dwimap.nii.gz" | sort | head -n 1)"
    odi_file="$(find "$odi_mni_dir" -maxdepth 1 -type f -name "${id}_acq-*_space-MNI152NLin6ASym_model-noddi_param-odi_dwimap.nii.gz" | sort | head -n 1)"
    isovf_file="$(find "$isovf_mni_dir" -maxdepth 1 -type f -name "${id}_acq-*_space-MNI152NLin6ASym_model-noddi_param-isovf_dwimap.nii.gz" | sort | head -n 1)"
    gm_file="$(find "$gm_mni_dir" -maxdepth 1 -type f -name "${id}_acq-*_space-MNI152NLin6ASym_label-GM_probability.nii.gz" | sort | head -n 1)"

    if [[ -z "$fa_file" || -z "$md_file" || -z "$ndi_file" || -z "$odi_file" || -z "$isovf_file" || -z "$gm_file" ]]; then
        continue
    fi

    echo "$fa_file" >> "$fa_list"
    echo "$md_file" >> "$md_list"
    echo "$ndi_file" >> "$ndi_list"
    echo "$odi_file" >> "$odi_list"
    echo "$isovf_file" >> "$isovf_list"
    echo "$gm_file" >> "$gm_list"
    echo "$id" >> "$subject_order"
done < "$log_dir/common_ids.txt"

fslmerge -t "$merge_dir/FA/all_FA.nii.gz" $(cat "$fa_list")
fslmerge -t "$merge_dir/MD/all_MD.nii.gz" $(cat "$md_list")
fslmerge -t "$merge_dir/NDI/all_NDI.nii.gz" $(cat "$ndi_list")
fslmerge -t "$merge_dir/ODI/all_ODI.nii.gz" $(cat "$odi_list")
fslmerge -t "$merge_dir/ISOVF/all_ISOVF.nii.gz" $(cat "$isovf_list")
fslmerge -t "$merge_dir/GM_fraction/all_GM_fraction.nii.gz" $(cat "$gm_list")

fslmaths "$merge_dir/FA/all_FA.nii.gz" -Tmean "$mean_img_dir/mean_FA.nii.gz"
fslmaths "$merge_dir/GM_fraction/all_GM_fraction.nii.gz" -Tmean "$mean_img_dir/mean_GM_fraction.nii.gz"

echo "===== Part 3: Skeletonize Images ====="

wm_skel_dir="$skeleton_dir/WM_skeleton"
gm_skel_dir="$skeleton_dir/GM_skeleton"

mean_fa="$mean_img_dir/mean_FA.nii.gz"
mean_gm="$mean_img_dir/mean_GM_fraction.nii.gz"

mean_fa_mask="$wm_skel_dir/mean_FA_mask.nii.gz"
mean_fa_skeleton="$wm_skel_dir/mean_FA_skeleton.nii.gz"
mean_fa_skeleton_mask="$wm_skel_dir/mean_FA_skeleton_mask.nii.gz"
mean_fa_skeleton_mask_dst="$wm_skel_dir/mean_FA_skeleton_mask_dst.nii.gz"

mean_gm_mask="$gm_skel_dir/mean_GM_mask.nii.gz"
mean_gm_skeleton="$gm_skel_dir/mean_GM_fraction_skeleton.nii.gz"
mean_gm_skeleton_mask="$gm_skel_dir/mean_GM_fraction_skeleton_mask.nii.gz"
mean_gm_skeleton_mask_dst="$gm_skel_dir/mean_GM_fraction_skeleton_mask_dst.nii.gz"
zero_mask="$gm_skel_dir/zero_mask.nii.gz"

search_mask="${FSLDIR}/data/standard/LowerCingulum_1mm"

mkdir -p "$wm_skel_dir" "$gm_skel_dir"
mkdir -p "$merge_dir/FA" "$merge_dir/MD" "$merge_dir/NDI" "$merge_dir/ODI" "$merge_dir/ISOVF" "$merge_dir/GM_fraction"

fslmaths "$mean_fa" -max 0 -Tmin -bin "$mean_fa_mask"
fslmaths "$mean_fa" -mas "$mean_fa_mask" "$mean_fa"
tbss_skeleton -i "$mean_fa" -o "$mean_fa_skeleton"
fslmaths "$mean_fa_skeleton" -thr 0.2 -bin "$mean_fa_skeleton_mask"
fslmaths "$mean_fa_mask" -mul -1 -add 1 -add "$mean_fa_skeleton_mask" "$mean_fa_skeleton_mask_dst"
distancemap -i "$mean_fa_skeleton_mask_dst" -o "$mean_fa_skeleton_mask_dst"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_fa" \
-p 0.2 \
"$mean_fa_skeleton_mask_dst" \
"$search_mask" \
"$merge_dir/FA/all_FA.nii.gz" \
"$merge_dir/FA/all_FA_skeletonised.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_fa" \
-p 0.2 \
"$mean_fa_skeleton_mask_dst" \
"$search_mask" \
"$merge_dir/FA/all_FA.nii.gz" \
"$merge_dir/MD/all_MD_skeletonised.nii.gz" \
-a "$merge_dir/MD/all_MD.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_fa" \
-p 0.2 \
"$mean_fa_skeleton_mask_dst" \
"$search_mask" \
"$merge_dir/FA/all_FA.nii.gz" \
"$merge_dir/NDI/all_NDI_skeletonised.nii.gz" \
-a "$merge_dir/NDI/all_NDI.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_fa" \
-p 0.2 \
"$mean_fa_skeleton_mask_dst" \
"$search_mask" \
"$merge_dir/FA/all_FA.nii.gz" \
"$merge_dir/ODI/all_ODI_skeletonised.nii.gz" \
-a "$merge_dir/ODI/all_ODI.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_fa" \
-p 0.2 \
"$mean_fa_skeleton_mask_dst" \
"$search_mask" \
"$merge_dir/FA/all_FA.nii.gz" \
"$merge_dir/ISOVF/all_ISOVF_skeletonised.nii.gz" \
-a "$merge_dir/ISOVF/all_ISOVF.nii.gz"

fslmaths "$mean_gm" -thr 0.5 -bin "$mean_gm_mask"
tbss_skeleton -i "$mean_gm" -o "$mean_gm_skeleton"
fslmaths "$mean_gm_skeleton" -thr 0.5 -bin "$mean_gm_skeleton_mask"
fslmaths "$mean_gm_mask" -mul -1 -add 1 -add "$mean_gm_skeleton_mask" "$mean_gm_skeleton_mask_dst"
distancemap -i "$mean_gm_skeleton_mask_dst" -o "$mean_gm_skeleton_mask_dst"
fslmaths "${FSLDIR}/data/standard/LowerCingulum_1mm" -mul 0 "$zero_mask"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_gm" \
-p 0.5 \
"$mean_gm_skeleton_mask_dst" \
"$zero_mask" \
"$merge_dir/GM_fraction/all_GM_fraction.nii.gz" \
"$merge_dir/GM_fraction/all_GM_fraction_skeletonised_GBSS.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_gm" \
-p 0.5 \
"$mean_gm_skeleton_mask_dst" \
"$zero_mask" \
"$merge_dir/GM_fraction/all_GM_fraction.nii.gz" \
"$merge_dir/FA/all_FA_skeletonised_GBSS.nii.gz" \
-a "$merge_dir/FA/all_FA.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_gm" \
-p 0.5 \
"$mean_gm_skeleton_mask_dst" \
"$zero_mask" \
"$merge_dir/GM_fraction/all_GM_fraction.nii.gz" \
"$merge_dir/MD/all_MD_skeletonised_GBSS.nii.gz" \
-a "$merge_dir/MD/all_MD.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_gm" \
-p 0.5 \
"$mean_gm_skeleton_mask_dst" \
"$zero_mask" \
"$merge_dir/GM_fraction/all_GM_fraction.nii.gz" \
"$merge_dir/NDI/all_NDI_skeletonised_GBSS.nii.gz" \
-a "$merge_dir/NDI/all_NDI.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_gm" \
-p 0.5 \
"$mean_gm_skeleton_mask_dst" \
"$zero_mask" \
"$merge_dir/GM_fraction/all_GM_fraction.nii.gz" \
"$merge_dir/ODI/all_ODI_skeletonised_GBSS.nii.gz" \
-a "$merge_dir/ODI/all_ODI.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_gm" \
-p 0.5 \
"$mean_gm_skeleton_mask_dst" \
"$zero_mask" \
"$merge_dir/GM_fraction/all_GM_fraction.nii.gz" \
"$merge_dir/ISOVF/all_ISOVF_skeletonised_GBSS.nii.gz" \
-a "$merge_dir/ISOVF/all_ISOVF.nii.gz"

echo "Done."
echo "Merged subject order: $subject_order"
echo "Output root: $output_dir"