#!/bin/bash

set -e

# WCH RSSI Project 2025-11-29
# We want RSSI lesion masks to be in 3D T1w space for further analysis.

# Input arguments
BIDS_DIR=$1
SUBJECT_ID=$2
SESSION_ID=$3

# Flags
have_dwi_lesion_mask=false
have_t1w_lesion_mask=false
have_mni_lesion_mask=false

#
lesion_mask_dir="${BIDS_DIR}/derivatives/lesion_mask/sub-${SUBJECT_ID}/ses-${SESSION_ID}"

# Check whether already have T1w lesion mask (*_space-T1w_desc-RSSI_mask.nii.gz)
t1w_lesion_mask="${lesion_mask_dir}/sub-${SUBJECT_ID}_ses-${SESSION_ID}_space-T1w_desc-RSSI_mask.nii.gz"
if [ -f "${t1w_lesion_mask}" ]; then
    echo "T1w lesion mask already exists: ${t1w_lesion_mask}"
    have_t1w_lesion_mask=true
fi

# Check whether have DWI lesion mask (*_space-DWIb1000_desc-RSSI_mask.nii.gz)
dwi_lesion_mask="${lesion_mask_dir}/sub-${SUBJECT_ID}_ses-${SESSION_ID}_space-DWIb1000_desc-RSSI_mask.nii.gz"
if [ -f "${dwi_lesion_mask}" ]; then
    echo "DWI lesion mask already exists: ${dwi_lesion_mask}"
    have_dwi_lesion_mask=true
fi

# if have DWI lesion mask but not T1w lesion mask, transform it to T1w space
if [ "${have_dwi_lesion_mask}" = true ] && [ "${have_t1w_lesion_mask}" = false ]; then
    echo "Transforming DWI lesion mask to T1w space..."
    # make a tempdir in lesion_mask_dir
    temp_dir=$(mktemp -d "${lesion_mask_dir}/temp_transform_XXXX")

    # Search for DWI file
    dwi_dir="${BIDS_DIR}/sub-${SUBJECT_ID}/ses-${SESSION_ID}/dwi"
    # Find DWI files (acq-DWIb1000_dir-AP or acq-DWIb1000_dir-PA)
    # If only one exists, use it. If both exist, prefer AP (We used AP dir DWI to draw ROI).
    dwi_file_ap=$(find "${dwi_dir}" -type f -name "sub-${SUBJECT_ID}_ses-${SESSION_ID}_acq-DWIb1000_dir-AP_dwi.nii.gz" | head -n 1)
    dwi_file_pa=$(find "${dwi_dir}" -type f -name "sub-${SUBJECT_ID}_ses-${SESSION_ID}_acq-DWIb1000_dir-PA_dwi.nii.gz" | head -n 1)
    if [ -n "${dwi_file_ap}" ]; then
        dwi_file="${dwi_file_ap}"
    elif [ -n "${dwi_file_pa}" ]; then
        dwi_file="${dwi_file_pa}"
    else
        echo "No DWI file found for subject ${SUBJECT_ID}, session ${SESSION_ID}."
        exit 1
    fi

    dwi_bval_file=${dwi_file%.nii.gz}.bval
    # If dwi_file is a 4D file, extract the b0 image. If 3D, copy it.
    dwi_dim=$(fslval "${dwi_file}" dim4)
    if [ "${dwi_dim}" -gt 1 ]; then
        echo "Extracting b0 image from 4D DWI file..."
        # According to bval file, find the index of the first b0 volume (b=0)
        b0_index=$(awk '{for(i=1;i<=NF;i++) if($i==0) {print i-1; exit}}' "${dwi_bval_file}")
        fslroi "${dwi_file}" "${temp_dir}/dwi_ref.nii.gz" "${b0_index}" 1
    else
        echo "Copying 3D DWI file as b0 image..."
        cp "${dwi_file}" "${temp_dir}/dwi_ref.nii.gz"
    fi

    # Search for T1w file
    anat_dir="${BIDS_DIR}/sub-${SUBJECT_ID}/ses-${SESSION_ID}/anat"
    # 3D T1w file (*_acq-highres_T1w.nii.gz or *_acq-SynthSR_T1w.nii.gz). Prefer highres if both exist.
    t1w_file_highres=$(find "${anat_dir}" -type f -name "sub-${SUBJECT_ID}_ses-${SESSION_ID}_acq-highres_T1w.nii.gz" | head -n 1)
    t1w_file_synthsr=$(find "${anat_dir}" -type f -name "sub-${SUBJECT_ID}_ses-${SESSION_ID}_acq-SynthSR_T1w.nii.gz" | head -n 1)
    if [ -n "${t1w_file_highres}" ]; then
        t1w_file="${t1w_file_highres}"
    elif [ -n "${t1w_file_synthsr}" ]; then
        t1w_file="${t1w_file_synthsr}"
    else
        echo "No T1w file found for subject ${SUBJECT_ID}, session ${SESSION_ID}."
        exit 1
    fi

    # Register DWI b0 to T1w
    xfm_dir="${BIDS_DIR}/derivatives/xfm/sub-${SUBJECT_ID}/ses-${SESSION_ID}"
    mkdir -p "${xfm_dir}"

    mri_synthmorph -o "${xfm_dir}/sub-${SUBJECT_ID}_ses-${SESSION_ID}_acq-DWIb1000_space-T1w_dwiref.nii.gz" \
        -t "${xfm_dir}/sub-${SUBJECT_ID}_ses-${SESSION_ID}_from-DWIb1000_to-T1w_warp.nii.gz" \
        -T "${xfm_dir}/sub-${SUBJECT_ID}_ses-${SESSION_ID}_from-T1w_to-DWIb1000_warp.nii.gz" \
        "${temp_dir}/dwi_ref.nii.gz" \
        "${t1w_file}" -g
    
    # Apply warp to DWI lesion mask
    mri_convert -at "${xfm_dir}/sub-${SUBJECT_ID}_ses-${SESSION_ID}_from-DWIb1000_to-T1w_warp.nii.gz" \
        "${dwi_lesion_mask}" \
        "${t1w_lesion_mask}" \
        -rt nearest
    
    # Post-process the warped mask to make edges smoother in T1 space
    # 1) Ensure binary mask
    fslmaths "${t1w_lesion_mask}" -thr 0.5 -bin "${t1w_lesion_mask}"

    # 2) Morphological closing: dilate then erode to smooth jagged edges
    tmp_mask="${lesion_mask_dir}/sub-${SUBJECT_ID}_ses-${SESSION_ID}_space-T1w_desc-RSSI_mask_tmp.nii.gz"

    # Slight dilation
    fslmaths "${t1w_lesion_mask}" -dilM "${tmp_mask}"

    # Slight erosion
    fslmaths "${tmp_mask}" -ero "${t1w_lesion_mask}"

    # 3) Optional light Gaussian smoothing + re-binarize
    #    This can help further soften stair-step artifacts while keeping a clean binary mask
    fslmaths "${t1w_lesion_mask}" -s 0.5 -thr 0.5 -bin "${t1w_lesion_mask}"

    # Remove temp file
    rm -f "${tmp_mask}"

    echo "Transformed T1w lesion mask saved at: ${t1w_lesion_mask}"
    # Clean up temp dir
    rm -rf "${temp_dir}"
fi