#!/bin/bash

NC="\e[39m"
BLUE="\e[34m"
GREEN="\e[32m"
RED="\e[31m"

# 参数
subject_id=$1        # 单个 subject
source_dir=$2        # 代码路径
SUBJECTS_DIR=$3      # freesurfer SUBJECTS_DIR
output_dir=$4        # 输出路径

root_dir=$(pwd)

echo -e "${BLUE}######################################"
echo -e "##### Starting Bullseye Pipeline #####"
echo -e "######################################${NC}"

# 找到输入目录
root_patient_dir=${SUBJECTS_DIR}/${subject_id}
if [[ ! -d $root_patient_dir ]]; then
  echo -e "${RED} XXXXX Subject ${subject_id} not found in ${SUBJECTS_DIR} XXXXX${NC}"
  exit 1
fi

# 建立输出目录
#out_patient_dir=${output_dir}/${subject_id}
out_patient_dir=$output_dir
mkdir -p $out_patient_dir

echo -e "${BLUE}################################################"
echo -e "##### Processing patient ${subject_id}  #####"
echo -e "###########################################${NC}"

###############################################
##### Concentric Parcellation #####
###############################################
echo -e "${BLUE}###############################################"
echo -e "##### Starting Concentric Parcellation #####"
echo -e "###########################################${NC}"

if [[ -f ${root_patient_dir}/mri/aseg.mgz && -f ${root_patient_dir}/mri/ribbon.mgz ]]; then 
  echo -e "${GREEN} ***** Files Ribbon and Aseg found *****${NC}"
  mri_convert ${root_patient_dir}/mri/aseg.mgz ${out_patient_dir}/aseg.nii.gz
  mri_convert ${root_patient_dir}/mri/ribbon.mgz ${out_patient_dir}/ribbon.nii.gz
else 
  echo -e "${RED} XXXXX ${subject_id} lacks Aseg or Ribbon files XXXXX${NC}"
  exit 1
fi

echo -e "${BLUE} ***** Creating Distance Maps *****${NC}"

python3 $source_dir/concentric_regions/get_cortex_mask.py "${out_patient_dir}/ribbon.nii.gz" "${out_patient_dir}/"
python3 $source_dir/concentric_regions/get_ventricles_mask.py "${out_patient_dir}/aseg.nii.gz" "${out_patient_dir}/"
python3 $source_dir/concentric_regions/get_distance.py "${out_patient_dir}/ventricles_mask.nii.gz" "${out_patient_dir}/cortex_mask.nii.gz" "${out_patient_dir}/"
python3 $source_dir/concentric_regions/get_distance_groups.py "${out_patient_dir}/dist_map.nii.gz" "${out_patient_dir}/"
python3 $source_dir/concentric_regions/get_distance_groups_masking.py "${out_patient_dir}/dist_map_grouped.nii.gz" "${out_patient_dir}/aseg.nii.gz" "${out_patient_dir}/"

if [[ -f ${out_patient_dir}/dist_map_grouped_masked.nii.gz ]]; then
  echo -e "${GREEN} ***** Concentric Parcellation Confirmed *****${NC}"
else
  echo -e "${RED} XXXXX Concentric output not confirmed XXXXX${NC}"
fi

###############################################
##### LOBAR Parcellation #####
###############################################
echo -e "${BLUE}########################################"
echo -e "##### Starting LOBAR Parcellation #####"
echo -e "######################################${NC}"

export SUBJECTS_DIR=$SUBJECTS_DIR

if [[ -f ${root_patient_dir}/label/lh.aparc.annot && -f ${root_patient_dir}/label/rh.aparc.annot ]]; then
  mkdir -p ${out_patient_dir}/aparc_labels
  mri_annotation2label --subject $subject_id --hemi rh --outdir ${out_patient_dir}/aparc_labels
  mri_annotation2label --subject $subject_id --hemi lh --outdir ${out_patient_dir}/aparc_labels
else
  echo -e "${RED} XXXXX ${subject_id} lacks aparc.annot XXXXX${NC}"
  exit 1
fi

# 合并 lobes
bash $source_dir/lobar_regions/get_lobes_labels.sh ${out_patient_dir}/aparc_labels/

# 生成 lobes colortable
cat > ${out_patient_dir}/aparc_lobes.ctab <<EOF
1  frontal_lobe   255   0     0   0
2  parietal_lobe  0   255     0   0
3  occipital_lobe 255 255     0   0
4  temporal_lobe  0     0   255   0
EOF

# 生成 lh 注释
mris_label2annot --s $subject_id \
  --ctab ${out_patient_dir}/aparc_lobes.ctab \
  --h lh --a aparc_lobes \
  --l ${out_patient_dir}/aparc_labels/lh.frontal_lobe.label \
  --l ${out_patient_dir}/aparc_labels/lh.parietal_lobe.label \
  --l ${out_patient_dir}/aparc_labels/lh.occipital_lobe.label \
  --l ${out_patient_dir}/aparc_labels/lh.temporal_lobe.label

# 生成 rh 注释
mris_label2annot --s $subject_id \
  --ctab ${out_patient_dir}/aparc_lobes.ctab \
  --h rh --a aparc_lobes \
  --l ${out_patient_dir}/aparc_labels/rh.frontal_lobe.label \
  --l ${out_patient_dir}/aparc_labels/rh.parietal_lobe.label \
  --l ${out_patient_dir}/aparc_labels/rh.occipital_lobe.label \
  --l ${out_patient_dir}/aparc_labels/rh.temporal_lobe.label

# 转换到 aseg 空间
mri_aparc2aseg --s $subject_id --annot aparc_lobes --labelwm --wmparc-dmax 1000 \
  --rip-unknown --hypo-as-wm --o ${out_patient_dir}/lobar_map.nii.gz --annot-table ${out_patient_dir}/aparc_lobes.ctab

python3 $source_dir/lobar_regions/get_lobar_aseg_mask.py "${out_patient_dir}/lobar_map.nii.gz" "${out_patient_dir}/"

if [[ -f ${out_patient_dir}/lobar_aseg_masked.nii.gz ]]; then
  echo -e "${GREEN} ***** Lobar Parcellation Confirmed *****${NC}"
else
  echo -e "${RED} XXXXX Lobar output not confirmed XXXXX${NC}"
fi

###############################################
##### Intersection #####
###############################################
echo -e "${BLUE}#################################"
echo -e "##### Starting Intersection #####"
echo -e "################################${NC}"

python3 $source_dir/intersect_regions/get_intersect_parcellation.py \
  "${out_patient_dir}/lobar_aseg_masked.nii.gz" \
  "${out_patient_dir}/dist_map_grouped_masked.nii.gz" \
  "${out_patient_dir}/" "bullseye_parcellation.nii.gz"

python3 $source_dir/intersect_regions/get_intersect_parcellation.py \
  "${out_patient_dir}/lobar_aseg_masked_bis.nii.gz" \
  "${out_patient_dir}/dist_map_grouped_masked_bis.nii.gz" \
  "${out_patient_dir}/" "bullseye_parcellation_bis.nii.gz"

###############################################
##### Finish #####
###############################################
echo -e "${BLUE}###############################################"
echo -e "${BLUE}########### PROGRAM FINISHED #################"
echo -e "${BLUE}#############################################${NC}"

bullseye_json=$source_dir/ctab/bullseye.json
lobarseg_json=$source_dir/ctab/lobarseg.json
# copy ctab files to output directory
cp $bullseye_json ${out_patient_dir}/bullseye.json
cp $lobarseg_json ${out_patient_dir}/lobarseg.json

echo -e "Outputs are saved in: ${out_patient_dir}"
