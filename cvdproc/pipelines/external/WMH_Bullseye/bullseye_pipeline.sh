#!bin/bash

NC="\e[39m"
BLUE="\e[34m"
GREEN="\e[32m"
RED="\e[31m"

subjects_list=$1
subjects=( $(cat $subjects_list) )
root_dir=`pwd`
source_dir=$2

echo -e "${BLUE}######################################"
echo -e "##### Starting Bullseye Pipeline #####"
echo -e "######################################${NC}"

echo -e "${BLUE}################################"
echo -e "##### Showing Subjects list  #####"
echo -e "##################################${NC}"

cat $subjects_list
cont_conc=0
cont_lob=0

for patient in ${subjects[@]}; do
  cd $root_dir
  
  echo -e "${BLUE}################################################"
  echo -e "##### Iterating in patient ${patient}  #####"
  echo -e "###########################################${NC}"

  cd $patient
  root_patient_dir=`pwd`
  echo -e "${BLUE}###############################################"
  echo -e "##### Starting Concentric Parcellation #####"
  echo -e "###########################################${NC}"
  
  if [[ -f mri/aseg.mgz && -f mri/ribbon.mgz ]]; then 
    echo -e "${GREEN} ***** Files Ribbon and Aseg found in ${root_patient}/mri *****${NC}"
    echo -e "${BLUE} ***** Converting to nii.gz *****${NC}"
    mri_convert mri/aseg.mgz mri/aseg.nii.gz
    mri_convert mri/ribbon.mgz mri/ribbon.nii.gz
  else 
    echo -e "${RED} XXXXX Patient ${patient} lacks Aseg or Ribbon files XXXXX"
    echo -e "${RED} XXXXX Patient ${patient} lacks Aseg or Ribbon files XXXXX" >> $root_dir/log_bullseye.txt
    echo -e "INTERRUPTING Iteration${NC}"
    continue
  fi 
  
  echo -e "${BLUE} ***** Creating Distance Maps *****${NC}"
  
  python3 $source_dir/concentric_regions/get_cortex_mask.py "${root_patient_dir}/mri/ribbon.nii.gz" "${root_patient_dir}/mri/" 
  python3 $source_dir/concentric_regions/get_ventricles_mask.py "${root_patient_dir}/mri/aseg.nii.gz" "${root_patient_dir}/mri/" 
  python3 $source_dir/concentric_regions/get_distance.py "${root_patient_dir}/mri/ventricles_mask.nii.gz" "${root_patient_dir}/mri/cortex_mask.nii.gz" "${root_patient_dir}/mri/" 
  python3 $source_dir/concentric_regions/get_distance_groups.py "${root_patient_dir}/mri/dist_map.nii.gz" "${root_patient_dir}/mri/" 
  python3 $source_dir/concentric_regions/get_distance_groups_masking.py "${root_patient_dir}/mri/dist_map_grouped.nii.gz" "${root_patient_dir}/mri/aseg.nii.gz" "${root_patient_dir}/mri/"
  
  if [[ -f mri/dist_map_grouped_masked.nii.gz ]]; then 
    echo -e "${GREEN} ***** Concentric Parcellation Confirmed in patient ${patient} *****${NC}"
    cont_conc=$(( $cont_conc+1 )) 
  else 
    echo -e "${RED} XXXXX Concentric output from patient ${patient} could not be confirmed XXXXX${NC}"
    echo -e "${RED} XXXXX Concentric output from patient ${patient} could not be confirmed XXXXX${NC}" >> $root_dir/log_bullseye.txt
  fi
  
  echo -e "${BLUE}########################################"
  echo -e "##### Starting LOBAR Parcellation #####"
  echo -e "######################################${NC}"
  
  export SUBJECTS_DIR=$root_dir
  
  if [[ -f label/lh.aparc.annot && -f label/rh.aparc.annot ]]; then 
    mri_annotation2label --subject $patient --hemi rh --outdir ./aparc_labels
    mri_annotation2label --subject $patient --hemi lh --outdir ./aparc_labels
  else 
    echo -e "${RED} XXXXX Patient ${patient} lacks aparc.annot XXXXX"
    echo -e "${RED} XXXXX Patient ${patient} lacks aparc.annot XXXXX" >> $root_dir/log_bullseye.txt
    echo -e "INTERRUPTING Iteration${NC}"
    continue
  fi 
  
  bash $source_dir/lobar_regions/get_lobes_labels.sh ./aparc_labels/
  
  mris_label2annot --s $patient --ctab $root_patient_dir/label/aparc.annot.ctab  --h lh --a aparc_lobes --l $root_patient_dir/aparc_labels/lh.frontal_lobe.label --l $root_patient_dir/aparc_labels/lh.parietal_lobe.label --l $root_patient_dir/aparc_labels/lh.occipital_lobe.label --l $root_patient_dir/aparc_labels/lh.temporal_lobe.label
  mris_label2annot --s $patient --ctab $root_patient_dir/label/aparc.annot.ctab  --h rh --a aparc_lobes --l $root_patient_dir/aparc_labels/rh.frontal_lobe.label --l $root_patient_dir/aparc_labels/rh.parietal_lobe.label --l $root_patient_dir/aparc_labels/rh.occipital_lobe.label --l $root_patient_dir/aparc_labels/rh.temporal_lobe.label
      
  mri_aparc2aseg --s $patient --annot aparc_lobes --labelwm --wmparc-dmax 1000 --rip-unknown --hypo-as-wm --o mri/lobar_map.nii.gz
  
  python3 $source_dir/lobar_regions/get_lobar_aseg_mask.py "${root_patient_dir}/mri/lobar_map.nii.gz" "${root_patient_dir}/mri/" 
  
  if [[ -f mri/'lobar_aseg_masked.nii.gz' ]]; then 
    echo -e "${GREEN} ***** Lobar Parcellation Confirmed in patient ${patient} *****${NC}"
    cont_conc=$(( $cont_lob+1 )) 
  else 
    echo -e "${RED} XXXXX Lobar output from patient ${patient} could not be confirmed XXXXX${NC}"
    echo -e "${RED} XXXXX Lobar output from patient ${patient} could not be confirmed XXXXX${NC}" >> $root_dir/log_bullseye.txt
  fi
  
  echo -e "${BLUE}#################################"
  echo -e "##### Starting Intersection #####"
  echo -e "################################${NC}"
  
  python3 $source_dir/intersect_regions/get_intersect_parcellation.py "${root_patient_dir}/mri/lobar_aseg_masked.nii.gz" "${root_patient_dir}/mri/dist_map_grouped_masked.nii.gz" "${root_patient_dir}/mri/" "bullseye_parcellation.nii.gz"
  python3 $source_dir/intersect_regions/get_intersect_parcellation.py "${root_patient_dir}/mri/lobar_aseg_masked_bis.nii.gz" "${root_patient_dir}/mri/dist_map_grouped_masked_bis.nii.gz" "${root_patient_dir}/mri/" "bullseye_parcellation_bis.nii.gz"
  
done

echo -e "${BLUE}###############################################"
echo -e "${BLUE}########### PROGRAM FINISHED #################"
echo -e "${BLUE}#############################################"
echo -e "Number of concentric parcellations: ${cont_conc}"
echo -e "Number of lobar parcellations: ${cont_lob}"  
     
