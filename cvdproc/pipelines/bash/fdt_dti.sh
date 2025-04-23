#!/bin/bash

# FSL FDT pipeline for DTI analysis

process_subject() {
    DWI_PATH=$1
    B0_ALL_PATH=$2
    ACQPARAMS_PATH=$3
    BVAL_PATH=$4
    BVEC_PATH=$5
    OUTPUT_DIR=$6

    # 1. TOPUP
    # 如果REVERSE_B0_PATH不为''，则需要进行topup

    if [ -f "$B0_ALL_PATH" ]; then
        echo "Performing topup"
        
        topup --imain="$B0_ALL_PATH" \
              --datain="$ACQPARAMS_PATH" \
              --config="b02b0.cnf" \
              --out="${OUTPUT_DIR}/topup_results" \
              --iout="${OUTPUT_DIR}/hifi_b0"
        
        # Get the b0 image and its brain mask (assume the first volume is the b0 image)
        fslroi "${OUTPUT_DIR}/hifi_b0" "${OUTPUT_DIR}/dwi_b0" 0 1
        mri_synthstrip -i "${OUTPUT_DIR}/dwi_b0.nii.gz" \
                -o "${OUTPUT_DIR}/dwi_b0.nii.gz" \
                -m "${OUTPUT_DIR}/dwi_b0_brain_mask.nii.gz"
    else
        fslroi $DWI_PATH "${OUTPUT_DIR}/dwi_b0" 0 1
        mri_synthstrip -i "${OUTPUT_DIR}/dwi_b0.nii.gz" \
            -o "${OUTPUT_DIR}/dwi_b0_brain.nii.gz" \
            -m "${OUTPUT_DIR}/dwi_b0_brain_mask.nii.gz"
    fi

    # 2. EDDY
    echo "Performing eddy"

    b_number=$(cat $BVAL_PATH | wc -w)
    #echo "b_number: $b_number"

    indx=""
    for ((i=1; i<=$b_number; i+=1)); do indx="$indx 1"; done
    echo $indx > "${OUTPUT_DIR}/index.txt"

    eddy_cuda10.2 diffusion --imain="${DWI_PATH}" \
                        --mask="${OUTPUT_DIR}/dwi_b0_brain_mask.nii.gz" \
                        --acqp="${ACQPARAMS_PATH}" \
                        --index="${OUTPUT_DIR}/index.txt" \
                        --bvecs="${BVEC_PATH}" --bvals="${BVAL_PATH}" \
                        --topup="${OUTPUT_DIR}/topup_results" \
                        --out="${OUTPUT_DIR}/eddy_corrected_data" \
                        --verbose
    
    # 3. DTIFIT
    echo "Performing dtifit"
    # Here should use the corrected bvec and bval!
    dtifit -k "${OUTPUT_DIR}/eddy_corrected_data" \
        -o "${OUTPUT_DIR}/dti" \
        -m "${OUTPUT_DIR}/dwi_b0_brain_mask.nii.gz" \
        -r "${OUTPUT_DIR}/eddy_corrected_data.eddy_rotated_bvecs" -b "${BVAL_PATH}"

    # # 4. BedpostX
    # echo "Make input folder for bedpostx"
    # # preprocessing
    # BedpostX_input_dir="${OUTPUT_DIR}/bedpostX_input"
    # mkdir -p "${BedpostX_input_dir}"

    # cp "${OUTPUT_DIR}/eddy_corrected_data.nii.gz" "${BedpostX_input_dir}/data.nii.gz"
    # cp "${OUTPUT_DIR}/dwi_b0_brain_mask.nii.gz" "${BedpostX_input_dir}/nodif_brain_mask.nii.gz"
    # cp "${OUTPUT_DIR}/eddy_corrected_data.eddy_rotated_bvecs" "${BedpostX_input_dir}/bvecs"
    # cp "${BVAL_PATH}" "${BedpostX_input_dir}/bvals"

    # bedpostx_datacheck "${BedpostX_input_dir}"
    # bedpostx_gpu "${BedpostX_input_dir}" -NJOBS 2
}

# Call the function with the provided paths
process_subject $1 $2 $3 $4 $5 $6