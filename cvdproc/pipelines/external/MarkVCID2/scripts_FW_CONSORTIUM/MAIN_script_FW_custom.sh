#!/bin/bash

# MODIFY FOLLOWING DIRECTORIES 
# FSL_PATH IS THE MAIN FSL PATH AND SHOULD CONTAIN BIN/, CONFIG/, LIB/... SUBDIRECTORIES)
# PYTHON_EXEC IS THE PATH FOR THE PYTHON EXECUTABLE

# Example: set for MRN Arvind
PYTHON_EXEC=python3
FSL_PATH=${FSL_DIR}
BINFSL=${FSL_PATH}/bin

###############################################################################
#DO NOT MODIFY
#set SUBJ_PATH = `ls -d $1`
DATAFW_PATH=$5
FWMRN_PATH=$6

OUTPUT_FW_IMG=$7

echo ${DATAFW_PATH}

MASKWM_PATH=${FWMRN_PATH}/FMRIB58_FA_1mm_thr  
DATAFSL_PATH=${FSL_PATH}/data/standard/FMRIB58_FA_1mm
DATA_FILE=${DATAFW_PATH}/data.nii.gz
BRAINMASK_FILE=${DATAFW_PATH}/brain_mask.nii.gz
BVAL_FILE=${DATAFW_PATH}/file.bval
BVEC_FILE=${DATAFW_PATH}/file.bvec

if [ ! -d ${DATAFW_PATH} ]; then
    mkdir -p ${DATAFW_PATH}
fi

cd ${DATAFW_PATH}

cp $1 ${DATA_FILE}
cp $2 ${BRAINMASK_FILE}
cp $3 ${BVAL_FILE}
cp $4 ${BVEC_FILE}

# GENERATE THE FW MAP
${PYTHON_EXEC} ${FWMRN_PATH}/fw_mrn.py ${DATAFW_PATH} 

# COMPUTE TRANSFORMATION PARAMETERS DTI -> FSL FA TEMPLATE
${BINFSL}/fsl_reg wls_dti_FA ${DATAFSL_PATH} nat2std -e -FA

# TRANSFORM FW, FA AND MD MAPS INTO FSL SPACE
${BINFSL}/applywarp -i wls_dti_FW -o wls_dti_FW_warp -r ${DATAFSL_PATH} -w nat2std_warp
${BINFSL}/applywarp -i fwc_wls_dti_FA -o fwc_wls_dti_FA_warp -r ${DATAFSL_PATH} -w nat2std_warp

# COMPUTE AND STORE IN summary.txt FILE MEAN FW and FA WITHIN WM VOXELS
echo "$(${BINFSL}/fslstats wls_dti_FW_warp -k ${MASKWM_PATH} -M) $(${BINFSL}/fslstats fwc_wls_dti_FA_warp -k ${MASKWM_PATH} -M)" > summary.txt

# clean up
rm  ${DATA_FILE}
rm  ${BRAINMASK_FILE}
rm  ${BVAL_FILE}
rm  ${BVEC_FILE}

rm ${DATAFW_PATH}/*log
rm ${DATAFW_PATH}/*MD*

DEFAULT_OUTPUT_FW_IMG="${DATAFW_PATH}/wls_dti_FW.nii.gz"
# Copy to final output location
cp "${DEFAULT_OUTPUT_FW_IMG}" "${OUTPUT_FW_IMG}"