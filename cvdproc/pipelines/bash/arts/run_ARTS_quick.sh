#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 8 ]; then
  echo "Usage: bash run_ARTS_quick.sh <subject_id> <age> <sex> <synthseg.nii.gz> <wmh_mask.nii.gz> <fa_mni.nii.gz> <output_dir> <mni_to_iit_warp.nii.gz>" >&2
  exit 1
fi

SUBJECT="$1"
AGE="$2"
SEX="$3"
SYNTHSEG_IN="$(realpath "$4")"
WMH_IN="$(realpath "$5")"
FA_MNI_IN="$(realpath "$6")"
SUB_OUT="$(realpath -m "$7")"
MNI_TO_IIT_WARP="$(realpath "$8")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTS_DATA_DIR="$(realpath "${SCRIPT_DIR}/../../../data/arts")"

IIT_FA="${ARTS_DATA_DIR}/IITmean_FA.nii.gz"
IIT_ATLAS_DIR="${ARTS_DATA_DIR}/IIT_atlas"
CALC_DTI_ROIS="${ARTS_DATA_DIR}/scripts/feature_extraction/calculate_DTI_ROIs"
CLASSIFIER_PY="${ARTS_DATA_DIR}/scripts/classifier/final_classifier.py"
CLASSIFIER_MODEL="${ARTS_DATA_DIR}/scripts/classifier/model_params.json"
TBSS_PREP_SCRIPT="${IIT_ATLAS_DIR}/tbss_4_prestats_iit"

for cmd in fslmaths fslstats fslinfo tbss_skeleton distancemap mri_convert; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing command: ${cmd}" >&2
    exit 1
  fi
done

for file in "${IIT_FA}" "${SYNTHSEG_IN}" "${WMH_IN}" "${FA_MNI_IN}" "${MNI_TO_IIT_WARP}" "${CALC_DTI_ROIS}" "${CLASSIFIER_PY}" "${CLASSIFIER_MODEL}" "${TBSS_PREP_SCRIPT}"; do
  if [ ! -f "${file}" ]; then
    echo "Missing file: ${file}" >&2
    exit 1
  fi
done

for file in "${IIT_ATLAS_DIR}/IITmean_FA_mask.nii.gz" "${IIT_ATLAS_DIR}/IITmean_FA_skeleton.nii.gz" "${IIT_ATLAS_DIR}/IITmean_lower_cingulum.nii.gz"; do
  if [ ! -f "${file}" ]; then
    echo "Missing IIT atlas file: ${file}" >&2
    exit 1
  fi
done

chmod +x "${CALC_DTI_ROIS}" "${TBSS_PREP_SCRIPT}"

fa_dim=($(fslinfo "${FA_MNI_IN}" | awk '/^dim1/{print $2} /^dim2/{print $2} /^dim3/{print $2}'))
iit_dim=($(fslinfo "${IIT_FA}" | awk '/^dim1/{print $2} /^dim2/{print $2} /^dim3/{print $2}'))
if [ "${fa_dim[0]}" != "${iit_dim[0]}" ] || [ "${fa_dim[1]}" != "${iit_dim[1]}" ] || [ "${fa_dim[2]}" != "${iit_dim[2]}" ]; then
  echo "FA dimensions do not match IITmean_FA. The v2 method requires a 1 mm MNI152NLin6Asym FA map." >&2
  exit 1
fi

ss_dim=($(fslinfo "${SYNTHSEG_IN}" | awk '/^dim1/{print $2} /^dim2/{print $2} /^dim3/{print $2}'))
wmh_dim=($(fslinfo "${WMH_IN}" | awk '/^dim1/{print $2} /^dim2/{print $2} /^dim3/{print $2}'))
if [ "${ss_dim[0]}" != "${wmh_dim[0]}" ] || [ "${ss_dim[1]}" != "${wmh_dim[1]}" ] || [ "${ss_dim[2]}" != "${wmh_dim[2]}" ]; then
  echo "SynthSeg and WMH dimensions do not match." >&2
  exit 1
fi

rm -rf "${SUB_OUT}/analysis" "${SUB_OUT}/DTI" "${SUB_OUT}/GMWM" \
  "${SUB_OUT}/WMH" "${SUB_OUT}/WMH_processing" "${SUB_OUT}/FA_processing" "${SUB_OUT}/QC"
mkdir -p "${SUB_OUT}/analysis" "${SUB_OUT}/DTI" "${SUB_OUT}/GMWM" "${SUB_OUT}/WMH" "${SUB_OUT}/WMH_processing" "${SUB_OUT}/FA_processing/tbss/stats" "${SUB_OUT}/QC"
export FSLOUTPUTTYPE=NIFTI_GZ

echo "${SUBJECT} ${AGE} ${SEX}" > "${SUB_OUT}/demo.txt"
echo "ARTS quick run for ${SUBJECT}" > "${SUB_OUT}/biomarker_input.txt"

fslmaths "${WMH_IN}" -thr 0.5 -bin "${SUB_OUT}/WMH/WMH_mask.nii.gz"
fslmaths "${SYNTHSEG_IN}" -thr 2 -uthr 2 -bin "${SUB_OUT}/GMWM/wm_left.nii.gz"
fslmaths "${SYNTHSEG_IN}" -thr 41 -uthr 41 -bin "${SUB_OUT}/GMWM/wm_right.nii.gz"
fslmaths "${SUB_OUT}/GMWM/wm_left.nii.gz" -add "${SUB_OUT}/GMWM/wm_right.nii.gz" -bin "${SUB_OUT}/GMWM/WM_mask.nii.gz"
rm -f "${SUB_OUT}/GMWM/wm_left.nii.gz" "${SUB_OUT}/GMWM/wm_right.nii.gz"
cp -f "${SYNTHSEG_IN}" "${SUB_OUT}/QC/synthseg.nii.gz"
cp -f "${SUB_OUT}/GMWM/WM_mask.nii.gz" "${SUB_OUT}/QC/WM_mask_from_synthseg.nii.gz"

fslmaths "${SUB_OUT}/GMWM/WM_mask.nii.gz" -thr 0.5 -bin "${SUB_OUT}/WMH_processing/WM_no_cerebellum.nii.gz"
fslmaths "${SUB_OUT}/WMH/WMH_mask.nii.gz" -thr 0.5 -bin -mul "${SUB_OUT}/WMH_processing/WM_no_cerebellum.nii.gz" "${SUB_OUT}/WMH_processing/WMH_no_cerebellum.nii.gz"
wmh_vol=$(fslstats "${SUB_OUT}/WMH_processing/WMH_no_cerebellum.nii.gz" -V | awk '{print $2}')
wm_vol=$(fslstats "${SUB_OUT}/WMH_processing/WM_no_cerebellum.nii.gz" -V | awk '{print $2}')
wmh_ratio=$(awk -v a="${wmh_vol}" -v b="${wm_vol}" 'BEGIN{if (b > 0) print a / b; else print "nan"}')
echo "${wmh_ratio}" > "${SUB_OUT}/WMH_processing/features.txt"
cp -f "${SUB_OUT}/WMH_processing/WM_no_cerebellum.nii.gz" "${SUB_OUT}/QC/WM_no_cerebellum.nii.gz"
cp -f "${SUB_OUT}/WMH_processing/WMH_no_cerebellum.nii.gz" "${SUB_OUT}/QC/WMH_no_cerebellum.nii.gz"

mri_convert -at "${MNI_TO_IIT_WARP}" "${FA_MNI_IN}" "${SUB_OUT}/FA_processing/tbss/stats/all_FA.nii.gz"
cp -f "${IIT_FA}" "${SUB_OUT}/QC/IITmean_FA.nii.gz"
cp -f "${SUB_OUT}/FA_processing/tbss/stats/all_FA.nii.gz" "${SUB_OUT}/QC/all_FA_to_IIT.nii.gz"

TBSS_DIR="${SUB_OUT}/FA_processing/tbss"
cd "${TBSS_DIR}"
cp -f "${IIT_ATLAS_DIR}/IITmean_FA_mask.nii.gz" .
cp -f "${IIT_ATLAS_DIR}/IITmean_FA_skeleton.nii.gz" .
cp -f "${IIT_ATLAS_DIR}/IITmean_lower_cingulum.nii.gz" .
cp -f "${IIT_FA}" .
fslmaths stats/all_FA -max 0 -Tmin -bin stats/mean_FA_mask -odt char
fslmaths stats/all_FA -mas stats/mean_FA_mask stats/all_FA
fslmaths "${IIT_FA}" stats/mean_FA.nii.gz
fslmaths stats/mean_FA -bin stats/mean_FA_mask
fslmaths stats/all_FA -mas stats/mean_FA_mask stats/all_FA
fslmaths "${IIT_ATLAS_DIR}/IITmean_FA_skeleton.nii.gz" stats/mean_FA_skeleton
bash "${TBSS_PREP_SCRIPT}" 0.25 > /dev/null

if [ ! -f "${TBSS_DIR}/stats/all_FA_skeletonised.nii.gz" ]; then
  echo "Missing all_FA_skeletonised.nii.gz" >&2
  exit 1
fi

cd "${SUB_OUT}/FA_processing"
gunzip -f tbss/stats/all_FA_skeletonised.nii.gz
"${CALC_DTI_ROIS}" tbss/stats/all_FA_skeletonised.nii
gzip -f tbss/stats/all_FA_skeletonised.nii
if [ ! -f "${SUB_OUT}/FA_processing/features.txt" ]; then
  echo "Missing FA_processing/features.txt" >&2
  exit 1
fi

cp -f "${SUB_OUT}/FA_processing/tbss/stats/all_FA_skeletonised.nii.gz" "${SUB_OUT}/QC/all_FA_skeletonised.nii.gz"
cp -f "${SUB_OUT}/FA_processing/features.txt" "${SUB_OUT}/QC/FA_features.txt"
cp -f "${SUB_OUT}/WMH_processing/features.txt" "${SUB_OUT}/QC/WMH_features.txt"

echo "projid age sex wmh_total/wm_total ROI_artscler_1_FA ROI_artscler_2_FA ROI_artscler_3_FA ROI_artscler_4_FA" > "${SUB_OUT}/analysis/classifier_input.txt"
paste -d " " "${SUB_OUT}/demo.txt" "${SUB_OUT}/WMH_processing/features.txt" "${SUB_OUT}/FA_processing/features.txt" >> "${SUB_OUT}/analysis/classifier_input.txt"
python3 "${CLASSIFIER_PY}" "${SUB_OUT}/analysis/classifier_input.txt" "${SUB_OUT}/analysis/score.csv" "${CLASSIFIER_MODEL}"
if [ ! -f "${SUB_OUT}/analysis/score.csv" ]; then
  echo "Missing score.csv" >&2
  exit 1
fi
cp -f "${SUB_OUT}/analysis/classifier_input.txt" "${SUB_OUT}/QC/classifier_input.txt"
cat "${SUB_OUT}/analysis/score.csv"
