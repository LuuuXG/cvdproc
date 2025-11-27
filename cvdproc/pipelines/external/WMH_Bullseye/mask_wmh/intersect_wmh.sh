#!bin/bash

NC="\e[39m"
BLUE="\e[34m"
GREEN="\e[32m"
RED="\e[31m"
CYAN="\e[36m" 

subjects_list=$1
subjects=( $(cat $subjects_list) )

root_dir=`pwd`
source_dir=$2

root_dir_target1=$3
root_dir_target2=$4
root_dir_target3=$5

echo -e "${BLUE}################################"
echo -e "##### Starting WMH Masking #####"
echo -e "################################${NC}"

echo -e "${BLUE}################################"
echo -e "##### Showing Subjects list  #####"
echo -e "##################################${NC}"

cat $subjects_list

cont=0
pattern="bianca_class._bin_edited"

for patient in ${subjects[@]}; do
  cd $root_dir
  file_ok=0
  
  echo -e "${BLUE}###########################################"
  echo -e "##### Iterating in patient ${patient}  #####"
  echo -e "###########################################${NC}"
  
  if [[ ! -f $root_dir/$patient/mri/bullseye_parcellation.nii.gz ]]; then
    echo -e "${RED} XXXXX Patient ${patient} does not have bullseye file XXXXX ${NC}"
    echo -e "${RED} XXXXX Stopping Iteration XXXXX ${NC}"
    continue
  fi 
  
  patient_name=$( echo $patient | cut -d '_' -f1)
  
  if [[ -d $root_dir_target1/$patient_name ]]; then 
    cp $root_dir/$patient/mri/bullseye_parcellation.nii.gz $root_dir_target1/$patient_name/new_wmh_bianca/
    cp $root_dir/$patient/mri/bullseye_parcellation_bis.nii.gz $root_dir_target1/$patient_name/new_wmh_bianca/
    
    target_dir=$root_dir_target1/$patient_name/new_wmh_bianca
    echo -e "**** ${GREEN} Patient ${patient_name} found in ${target_dir} ****${NC}"
  
  elif [[ -d $root_dir_target2/$patient_name ]]; then 
    cp $root_dir/$patient/mri/bullseye_parcellation.nii.gz $root_dir_target2/$patient_name/new_wmh_bianca/
    cp $root_dir/$patient/mri/bullseye_parcellation_bis.nii.gz $root_dir_target2/$patient_name/new_wmh_bianca/
    
    target_dir=$root_dir_target2/$patient_name/new_wmh_bianca 
    echo -e "**** ${GREEN} Patient ${patient_name} found in ${target_dir} ****${NC}"
    
  elif [[ -d $root_dir_target3/$patient_name ]]; then 
    cp $root_dir/$patient/mri/bullseye_parcellation.nii.gz $root_dir_target3/$patient_name/new_wmh_bianca/
    cp $root_dir/$patient/mri/bullseye_parcellation_bis.nii.gz $root_dir_target3/$patient_name/new_wmh_bianca/
    
    target_dir=$root_dir_target3/$patient_name/new_wmh_bianca
    echo -e "**** ${GREEN} Patient ${patient_name} found in ${target_dir} ****${NC}"  
    
  else
    echo -e "${RED} XXXXX PATIENT ${patient_name} not found in FLAIRS directories XXXXX"
    echo -e "XXXXX STOPPING ITERATION XXXXX${NC}"
    continue
  fi
  
  cd $target_dir
  for file in *; do 
    if [[ $file =~ $pattern ]]; then 
      if [[ $file =~ "~" ]]; then 
        continue
      else
        echo -e "${CYAN} ---- FILE ${file} found in patient ${patient_name}  ----- ${NC}"
        wmh_bin=$file
        file_ok=1
        break
      fi 
    fi
  done 
  
  if [[ $file_ok -lt 1 ]]; then 
    echo -e "${RED} XXXXX PATIENT ${patient_name} HAS NO WMH BINARY FILE IN ${target_dir} XXXXX ${NC}" 
    echo -e "${RED} XXXXX STOPPING ITERATION XXXXX${NC}"
    continue
  fi 
  
   
  echo -e "${BLUE}###########################################"
  echo -e "##### Startin WMH Masking in ${patient_name}  #####"
  echo -e "###########################################${NC}"  
  
  python3 $source_dir/get_intersect_bullseye_wmh.py $target_dir/bullseye_parcellation.nii.gz $target_dir/bullseye_parcellation_bis.nii.gz $target_dir/$wmh_bin $target_dir/ bullseye_wmh_edited_mask.nii.gz bullseye_wmh_edited_mask_bis.nii.gz
  
  Rscript $source_dir/get_stats_table.R $target_dir/bullseye_wmh_edited_mask.nii.gz $patient_name $target_dir/bullseye_wmh_table.txt 
  Rscript $source_dir/get_stats_table.R $target_dir/bullseye_wmh_edited_mask_bis.nii.gz $patient_name $target_dir/bullseye_wmh_table_bis.txt 
  
  if [[ -f bullseye_wmh_table.txt ]]; then
    echo -e "${GREEN} **** Stats File Confirmed in patient ${patient_name} **** ${NC}"
    cat bullseye_wmh_table.txt
    cat bullseye_wmh_table_bis.txt
    cont=$(( $cont+1 )) 
  else 
    echo -e "${RED} XXXX Output could not be confirmed in patient ${patient_name} **** ${NC}"
  fi 
  
done 

echo -e "${BLUE}###############################################"
echo -e "${BLUE}########### PROGRAM FINISHED #################"
echo -e "${BLUE}#############################################"
echo -e "Number of tables: ${cont}"
     
