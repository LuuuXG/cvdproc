#!/usr/bin/env bash

set -euo pipefail

dwi_mask="$1"
fa_image="$2"
iso_image="$3"   # NODDI isotropic fraction (CSF)
output_dir="$4"
output_pseudo_t1_filename="$5"

mkdir -p "$output_dir"

echo "==== Step 1: Atropos segmentation ===="

Atropos \
  -d 3 \
  -a "$fa_image" \
  -x "$dwi_mask" \
  -i Kmeans[2] \
  -m [0.3,1x1x1] \
  -o ["$output_dir/segmentation.nii.gz","$output_dir/prob_%02d.nii.gz"]

echo "==== Step 2: Identify WM class ===="

fa_mean_1=$(fslstats "$fa_image" -k "$output_dir/prob_01.nii.gz" -M)
fa_mean_2=$(fslstats "$fa_image" -k "$output_dir/prob_02.nii.gz" -M)

echo "FA mean prob_01: $fa_mean_1"
echo "FA mean prob_02: $fa_mean_2"

if (( $(echo "$fa_mean_1 > $fa_mean_2" | bc -l) )); then
    wm_prob="$output_dir/prob_01.nii.gz"
else
    wm_prob="$output_dir/prob_02.nii.gz"
fi

echo "Selected WM probability map: $wm_prob"

cp "$wm_prob" "$output_dir/wm_fraction.nii.gz"

echo "==== Step 3: Compute GM fraction ===="

# GM = max(0, 1 - WM - ISO)

fslmaths "$output_dir/wm_fraction.nii.gz" \
  -add "$iso_image" \
  -sub 1 -mul -1 \
  -thr 0 \
  -mul "$dwi_mask" \
  "$output_dir/gm_fraction.nii.gz"

echo "==== Step 4: Remove rim artifact ===="

ImageMath 3 "$output_dir/wm_largest_component.nii.gz" \
GetLargestComponent "$output_dir/wm_fraction.nii.gz"

fslmaths "$output_dir/wm_fraction.nii.gz" \
  -bin \
  -sub "$output_dir/wm_largest_component.nii.gz" \
  -thr 0 -bin \
  "$output_dir/wm_rim.nii.gz"

echo "==== Step 5: Create pseudo-T1 components ===="

# WM contribution
fslmaths "$output_dir/wm_fraction.nii.gz" \
  -mul "$output_dir/wm_largest_component.nii.gz" \
  -mul 2 \
  "$output_dir/wm_con.nii.gz"

# GM contribution
fslmaths "$output_dir/gm_fraction.nii.gz" \
  -thr 0 \
  -mul 1 \
  "$output_dir/gm_con.nii.gz"

echo "==== Step 6: Construct pseudo-T1 ===="

fslmaths "$output_dir/gm_con.nii.gz" \
  -add "$output_dir/wm_con.nii.gz" \
  -mul "$dwi_mask" \
  "$output_dir/$output_pseudo_t1_filename"

echo "==== Done ===="
echo "Pseudo-T1 image: $output_dir/$output_pseudo_t1_filename"