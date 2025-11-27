#!/usr/bin/env bash

set -e

preproc_dwi=$1
preproc_bvec=$2
preproc_bval=$3
dwi_mask=$4
fs_subjects_dir=$5
fs_subject_id=$6
fs_to_t1w_mat=$7
t1w_to_dwi_mat=$8
output_dir=$9
preproc_dwiref=${10}
aseg_nifti=${11}

mkdir -p ${output_dir}

# replace preproc_dwi .nii.gz with .mif (same directory and filename)
preproc_dwi_mif=${preproc_dwi/.nii.gz/.mif}
mrconvert ${preproc_dwi} ${preproc_dwi_mif} -fslgrad ${preproc_bvec} ${preproc_bval} -force
mrconvert ${dwi_mask} ${output_dir}/mask.mif

# ---- Auto-detect whether DWI is multi-shell ----
unique_nonzero_bvals=$(awk '
  {
    for (i=1; i<=NF; i++) if ($i > 50) a[int($i)] = 1
  }
  END {
    n = 0
    for (k in a) n++
    print n
  }' "${preproc_bval}")

echo "[INFO] Number of non-zero unique b-values: ${unique_nonzero_bvals}"

if [ "${unique_nonzero_bvals}" -gt 1 ]; then
    echo "[INFO] Detected multi-shell data -> using dwi2response dhollander"
    dwi2response dhollander \
      "${preproc_dwi_mif}" \
      "${output_dir}/wm_response.txt" \
      "${output_dir}/gm_response.txt" \
      "${output_dir}/csf_response.txt" \
      -voxels "${output_dir}/voxels.mif"

    dwi2fod msmt_csd \
      "${preproc_dwi_mif}" \
      -mask "${output_dir}/mask.mif" \
      "${output_dir}/wm_response.txt"  "${output_dir}/wm_fod.mif" \
      "${output_dir}/gm_response.txt"  "${output_dir}/gm_fod.mif" \
      "${output_dir}/csf_response.txt" "${output_dir}/csf_fod.mif"

    mtnormalise ${output_dir}/wm_fod.mif ${output_dir}/wm_fod_norm.mif ${output_dir}/gm_fod.mif ${output_dir}/gm_fod_norm.mif ${output_dir}/csf_fod.mif ${output_dir}/csf_fod_norm.mif -mask ${output_dir}/mask.mif

else
    echo "[INFO] Detected single-shell data -> using dwi2response tournier + single-shell CSD"
    dwi2response tournier \
      "${preproc_dwi_mif}" \
      "${output_dir}/wm_response.txt" \
      -voxels "${output_dir}/voxels.mif"

    dwi2fod csd \
      "${preproc_dwi_mif}" \
      "${output_dir}/wm_response.txt" \
      "${output_dir}/wm_fod.mif" \
      -mask "${output_dir}/mask.mif"

    mtnormalise ${output_dir}/wm_fod.mif ${output_dir}/wm_fod_norm.mif -mask ${output_dir}/mask.mif

fi

# concatenate transforms: fs_to_t1w_mat and t1w_to_dwi_mat
fs_to_dwi_mat=${output_dir}/fs_to_dwi.mat
convert_xfm -omat ${fs_to_dwi_mat} -concat ${t1w_to_dwi_mat} ${fs_to_t1w_mat}

# transform aparc+aseg to DWI space
flirt -in ${aseg_nifti} -ref ${preproc_dwiref} -out ${output_dir}/aparc_aseg_dwi.nii.gz -applyxfm -init ${fs_to_dwi_mat} -interp nearestneighbour

5ttgen freesurfer ${output_dir}/aparc_aseg_dwi.nii.gz ${output_dir}/5tt_dwi.mif
5tt2gmwmi ${output_dir}/5tt_dwi.mif ${output_dir}/gmwmSeed_dwi.mif

tckgen -act ${output_dir}/5tt_dwi.mif -backtrack -seed_gmwmi ${output_dir}/gmwmSeed_dwi.mif -select 1000000 -cutoff 0.06 ${output_dir}/wm_fod_norm.mif ${output_dir}/tracks_1M.tck -maxlength 250 -nthreads 4

tcksift2 -act ${output_dir}/5tt_dwi.mif -out_coeffs ${output_dir}/sift2_coeffs.txt -out_mu ${output_dir}/sift2_mu.txt ${output_dir}/tracks_1M.tck ${output_dir}/wm_fod_norm.mif ${output_dir}/sift2_weights.txt -nthreads 4

mrtrix3_path=$(dirname $(which mrconvert))/..
fs_default_text=${mrtrix3_path}/share/mrtrix3/labelconvert/fs_default.txt

labelconvert ${output_dir}/aparc_aseg_dwi.nii.gz $FREESURFER_HOME/FreeSurferColorLUT.txt ${fs_default_text} ${output_dir}/aparc_aseg_dwi_parcel.mif

tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in ${output_dir}/sift2_weights.txt ${output_dir}/tracks_1M.tck ${output_dir}/aparc_aseg_dwi_parcel.mif ${output_dir}/connectome.csv