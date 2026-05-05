#!/usr/bin/env bash

set -euo pipefail

input_root="$1"
output_dir="$2"
threads="${3:-16}"

alps_dir="$input_root/WCH_output/WCH_DTI_ALPS"

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

process_one_subject() {
    local subject_id="$1"

    local sub_dir="$alps_dir/$subject_id"
    local tmp_single_dir="$tmp_dir/$subject_id"

    mkdir -p "$tmp_single_dir"

    echo "Processing: $subject_id"

    local fa_img="$sub_dir/dti_FA.nii.gz"
    local md_img="$sub_dir/dti_MD.nii.gz"
    local ndi_img_orig="$sub_dir/fit_NDI.nii.gz"
    local odi_img_orig="$sub_dir/fit_ODI.nii.gz"
    local iso_img_orig="$sub_dir/fit_FWF.nii.gz"

    local dwi_to_t1w_xfm="$sub_dir/dti2struct.mat"
    local t1w_ref="$sub_dir/t1w.nii.gz"
    local t1_to_mni_warp="$sub_dir/struct2template_warp.nii.gz"

    if [[ ! -f "$fa_img" || ! -f "$md_img" || ! -f "$ndi_img_orig" || ! -f "$odi_img_orig" || ! -f "$iso_img_orig" ]]; then
        echo "Skip $subject_id: missing one or more diffusion maps"
        return 0
    fi

    if [[ ! -f "$dwi_to_t1w_xfm" || ! -f "$t1w_ref" || ! -f "$t1_to_mni_warp" ]]; then
        echo "Skip $subject_id: missing transform files"
        return 0
    fi

    local base_prefix="$subject_id"

    # ------------------------------------------------------------------
    # Step A: create FA-based mask
    # ------------------------------------------------------------------
    local fa_mask_img="$tmp_single_dir/${base_prefix}_desc-fa_nonzero_mask.nii.gz"
    fslmaths "$fa_img" -bin "$fa_mask_img"

    if [[ ! -f "$fa_mask_img" ]]; then
        echo "Skip $subject_id: failed to create FA-based mask"
        return 1
    fi

    # ------------------------------------------------------------------
    # Step B: fix NODDI geometry using FA header
    # ------------------------------------------------------------------
    local ndi_img="$tmp_single_dir/${base_prefix}_NDI_fixedgeom.nii.gz"
    local odi_img="$tmp_single_dir/${base_prefix}_ODI_fixedgeom.nii.gz"
    local iso_img="$tmp_single_dir/${base_prefix}_ISOVF_fixedgeom.nii.gz"

    cp "$ndi_img_orig" "$ndi_img"
    cp "$odi_img_orig" "$odi_img"
    cp "$iso_img_orig" "$iso_img"

    fslcpgeom "$fa_img" "$ndi_img"
    fslcpgeom "$fa_img" "$odi_img"
    fslcpgeom "$fa_img" "$iso_img"

    # ------------------------------------------------------------------
    # Step C: Atropos segmentation
    # ------------------------------------------------------------------
    local atropos_seg="$tmp_single_dir/${base_prefix}_desc-atropos_seg.nii.gz"
    local atropos_prob01="$tmp_single_dir/${base_prefix}_desc-atropos_prob01.nii.gz"
    local atropos_prob02="$tmp_single_dir/${base_prefix}_desc-atropos_prob02.nii.gz"

    local wm_prob_raw="$raw_img_dir/WM_fraction/${base_prefix}_label-WM_probability.nii.gz"
    local gm_prob_raw="$raw_img_dir/GM_fraction/${base_prefix}_label-GM_probability.nii.gz"
    local pseudo_t1_raw="$raw_img_dir/pseudoT1w/${base_prefix}_desc-pseudoT1w_T1w.nii.gz"

    local wm_largest_component="$tmp_single_dir/${base_prefix}_desc-wm_largest_component.nii.gz"
    local wm_rim="$tmp_single_dir/${base_prefix}_desc-wm_rim.nii.gz"
    local wm_con="$tmp_single_dir/${base_prefix}_desc-wm_con.nii.gz"
    local gm_con="$tmp_single_dir/${base_prefix}_desc-gm_con.nii.gz"

    Atropos \
      -d 3 \
      -a "$fa_img" \
      -x "$fa_mask_img" \
      -i Kmeans[2] \
      -m [0.3,1x1x1] \
      -o ["$atropos_seg","$tmp_single_dir/${base_prefix}_desc-atropos_prob%02d.nii.gz"]

    if [[ ! -f "$atropos_prob01" || ! -f "$atropos_prob02" ]]; then
        echo "Skip $subject_id: Atropos probability maps not found"
        ls -lh "$tmp_single_dir" || true
        return 1
    fi

    local fa_mean_1
    local fa_mean_2
    local wm_prob_source

    fa_mean_1="$(fslstats "$fa_img" -k "$atropos_prob01" -M)"
    fa_mean_2="$(fslstats "$fa_img" -k "$atropos_prob02" -M)"

    if (( $(echo "$fa_mean_1 > $fa_mean_2" | bc -l) )); then
        wm_prob_source="$atropos_prob01"
    else
        wm_prob_source="$atropos_prob02"
    fi

    fslmaths "$wm_prob_source" "$wm_prob_raw"

    if [[ ! -f "$wm_prob_raw" ]]; then
        echo "Skip $subject_id: failed to create WM fraction map"
        return 1
    fi

    # ------------------------------------------------------------------
    # Step D: compute GM fraction
    # GM = max(0, 1 - WM - ISO)
    # ------------------------------------------------------------------
    fslmaths "$wm_prob_raw" \
      -add "$iso_img" \
      -sub 1 -mul -1 \
      -thr 0 \
      -mul "$fa_mask_img" \
      "$gm_prob_raw"

    # ------------------------------------------------------------------
    # Step E: remove rim artifact and create pseudo-T1
    # ------------------------------------------------------------------
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
      -mul "$fa_mask_img" \
      "$pseudo_t1_raw"

    # ------------------------------------------------------------------
    # Step F: copy raw images
    # ------------------------------------------------------------------
    cp "$fa_img" "$raw_img_dir/FA/${base_prefix}_FA.nii.gz"
    cp "$md_img" "$raw_img_dir/MD/${base_prefix}_MD.nii.gz"
    cp "$ndi_img" "$raw_img_dir/NDI/${base_prefix}_NDI.nii.gz"
    cp "$odi_img" "$raw_img_dir/ODI/${base_prefix}_ODI.nii.gz"
    cp "$iso_img" "$raw_img_dir/ISOVF/${base_prefix}_ISOVF.nii.gz"

    # ------------------------------------------------------------------
    # Step G: transform to T1w space
    # ------------------------------------------------------------------
    local fa_t1w="$t1w_img_dir/FA/${base_prefix}_space-T1w_FA.nii.gz"
    local md_t1w="$t1w_img_dir/MD/${base_prefix}_space-T1w_MD.nii.gz"
    local ndi_t1w="$t1w_img_dir/NDI/${base_prefix}_space-T1w_NDI.nii.gz"
    local odi_t1w="$t1w_img_dir/ODI/${base_prefix}_space-T1w_ODI.nii.gz"
    local iso_t1w="$t1w_img_dir/ISOVF/${base_prefix}_space-T1w_ISOVF.nii.gz"
    local wm_t1w="$t1w_img_dir/WM_fraction/${base_prefix}_space-T1w_label-WM_probability.nii.gz"
    local gm_t1w="$t1w_img_dir/GM_fraction/${base_prefix}_space-T1w_label-GM_probability.nii.gz"
    local pseudo_t1w="$t1w_img_dir/pseudoT1w/${base_prefix}_space-T1w_desc-pseudoT1w_T1w.nii.gz"

    flirt -in "$fa_img" -ref "$t1w_ref" -out "$fa_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$md_img" -ref "$t1w_ref" -out "$md_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$ndi_img" -ref "$t1w_ref" -out "$ndi_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$odi_img" -ref "$t1w_ref" -out "$odi_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$iso_img" -ref "$t1w_ref" -out "$iso_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$wm_prob_raw" -ref "$t1w_ref" -out "$wm_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$gm_prob_raw" -ref "$t1w_ref" -out "$gm_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear
    flirt -in "$pseudo_t1_raw" -ref "$t1w_ref" -out "$pseudo_t1w" -applyxfm -init "$dwi_to_t1w_xfm" -interp trilinear

    # ------------------------------------------------------------------
    # Step H: transform to MNI space
    # ------------------------------------------------------------------
    local fa_mni="$mni_img_dir/FA/${base_prefix}_space-MNI152NLin6ASym_FA.nii.gz"
    local md_mni="$mni_img_dir/MD/${base_prefix}_space-MNI152NLin6ASym_MD.nii.gz"
    local ndi_mni="$mni_img_dir/NDI/${base_prefix}_space-MNI152NLin6ASym_NDI.nii.gz"
    local odi_mni="$mni_img_dir/ODI/${base_prefix}_space-MNI152NLin6ASym_ODI.nii.gz"
    local iso_mni="$mni_img_dir/ISOVF/${base_prefix}_space-MNI152NLin6ASym_ISOVF.nii.gz"
    local wm_mni="$mni_img_dir/WM_fraction/${base_prefix}_space-MNI152NLin6ASym_label-WM_probability.nii.gz"
    local gm_mni="$mni_img_dir/GM_fraction/${base_prefix}_space-MNI152NLin6ASym_label-GM_probability.nii.gz"
    local pseudo_mni="$mni_img_dir/pseudoT1w/${base_prefix}_space-MNI152NLin6ASym_desc-pseudoT1w_T1w.nii.gz"

    mri_convert -at "$t1_to_mni_warp" "$fa_t1w" "$fa_mni"
    mri_convert -at "$t1_to_mni_warp" "$md_t1w" "$md_mni"
    mri_convert -at "$t1_to_mni_warp" "$ndi_t1w" "$ndi_mni"
    mri_convert -at "$t1_to_mni_warp" "$odi_t1w" "$odi_mni"
    mri_convert -at "$t1_to_mni_warp" "$iso_t1w" "$iso_mni"
    mri_convert -at "$t1_to_mni_warp" "$wm_t1w" "$wm_mni"
    mri_convert -at "$t1_to_mni_warp" "$gm_t1w" "$gm_mni"
    mri_convert -at "$t1_to_mni_warp" "$pseudo_t1w" "$pseudo_mni"

    echo "Done: $subject_id"
}

export input_root
export output_dir
export threads
export alps_dir
export raw_img_dir
export t1w_img_dir
export mni_img_dir
export mean_img_dir
export skeleton_dir
export tmp_dir
export log_dir
export merge_dir
export -f process_one_subject

echo "===== Part 1: Prepare Scalar Images ====="

task_list="$log_dir/subject_tasks.txt"
: > "$task_list"

for sub_dir in "$alps_dir"/sub-*; do
    [[ -d "$sub_dir" ]] || continue
    subject_id="$(basename "$sub_dir")"
    echo "$subject_id" >> "$task_list"
done

cat "$task_list" | xargs -n 1 -P "$threads" bash -c 'process_one_subject "$1"' _

echo "===== Part 2: Merge 4D Images and Calculate Mean Images ====="

extract_ids() {
    local input_dir="$1"
    local output_txt="$2"
    find "$input_dir" -maxdepth 1 -type f -name "*.nii.gz" -printf "%f\n" \
        | sed -E 's/_space-.*$//' \
        | sed -E 's/_(FA|MD|NDI|ODI|ISOVF)$//' \
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

subject_order="$merge_dir/subject_order.txt"
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
    fa_file="$fa_mni_dir/${id}_space-MNI152NLin6ASym_FA.nii.gz"
    md_file="$md_mni_dir/${id}_space-MNI152NLin6ASym_MD.nii.gz"
    ndi_file="$ndi_mni_dir/${id}_space-MNI152NLin6ASym_NDI.nii.gz"
    odi_file="$odi_mni_dir/${id}_space-MNI152NLin6ASym_ODI.nii.gz"
    isovf_file="$isovf_mni_dir/${id}_space-MNI152NLin6ASym_ISOVF.nii.gz"
    gm_file="$gm_mni_dir/${id}_space-MNI152NLin6ASym_label-GM_probability.nii.gz"

    if [[ ! -f "$fa_file" || ! -f "$md_file" || ! -f "$ndi_file" || ! -f "$odi_file" || ! -f "$isovf_file" || ! -f "$gm_file" ]]; then
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

# TBSS
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
"$merge_dir/FA/all_FA_skeletonised_TBSS.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_fa" \
-p 0.2 \
"$mean_fa_skeleton_mask_dst" \
"$search_mask" \
"$merge_dir/FA/all_FA.nii.gz" \
"$merge_dir/MD/all_MD_skeletonised_TBSS.nii.gz" \
-a "$merge_dir/MD/all_MD.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_fa" \
-p 0.2 \
"$mean_fa_skeleton_mask_dst" \
"$search_mask" \
"$merge_dir/FA/all_FA.nii.gz" \
"$merge_dir/NDI/all_NDI_skeletonised_TBSS.nii.gz" \
-a "$merge_dir/NDI/all_NDI.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_fa" \
-p 0.2 \
"$mean_fa_skeleton_mask_dst" \
"$search_mask" \
"$merge_dir/FA/all_FA.nii.gz" \
"$merge_dir/ODI/all_ODI_skeletonised_TBSS.nii.gz" \
-a "$merge_dir/ODI/all_ODI.nii.gz"

${FSLDIR}/bin/tbss_skeleton \
-i "$mean_fa" \
-p 0.2 \
"$mean_fa_skeleton_mask_dst" \
"$search_mask" \
"$merge_dir/FA/all_FA.nii.gz" \
"$merge_dir/ISOVF/all_ISOVF_skeletonised_TBSS.nii.gz" \
-a "$merge_dir/ISOVF/all_ISOVF.nii.gz"

# GBSS
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