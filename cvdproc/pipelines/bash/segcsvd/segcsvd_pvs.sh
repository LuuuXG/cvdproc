#!/usr/bin/env bash

set -e

t1w_file=$1 # skull-stripped, bias-corrected T1w image
synthseg_file=$2 # SynthSeg output file
wmh_file=$3 # optional, if not provided, a pseudo WMH (all=0) will be created
output_dir=$4
pvs_probmap_filename=$5 # pvs_probmap.nii.gz
pvs_binary_filename=$6 # thr_pvs_seg.nii.gz
threshold=$7 # threshold for binary PVS map

# create a temp dir (the same level as output_dir to avoid permission issues)
temp_dir="${output_dir}/temp_segcsvd_$$"
mkdir -p $output_dir
mkdir -p $temp_dir

# if wmh_file is provided, copy it to temp_dir; otherwise, create a pseudo WMH (all=0)
if [ -n "$wmh_file" ] && [ -f "$wmh_file" ]; then
    cp $wmh_file $temp_dir/wmh.nii.gz
else
    fslmaths $t1w_file -mul 0 $temp_dir/wmh.nii.gz
fi

# copy t1w and synthseg
cp $t1w_file $temp_dir/t1w.nii.gz
cp $synthseg_file $temp_dir/synthseg.nii.gz

DOCKER_IMAGE="segcsvd_rc03:latest"

echo "Running segcsvd container..."
docker run --rm \
    -v "${temp_dir}:/indir" \
    -v "${output_dir}:/outdir" \
    -w / \
    $DOCKER_IMAGE \
    segment_pvs \
    /indir/t1w.nii.gz \
    /indir/synthseg.nii.gz \
    /indir/wmh.nii.gz \
    /outdir/pvs_seg.nii.gz \
    "1.0,1.4" 0 \
    true \
    true \
    $threshold

rm -rf "$temp_dir"

# rename output files
mv ${output_dir}/pvs_seg.nii.gz ${output_dir}/${pvs_probmap_filename}
mv ${output_dir}/thr_pvs_seg.nii.gz ${output_dir}/${pvs_binary_filename}

echo "PVS segmentation completed. Results saved in ${output_dir}"