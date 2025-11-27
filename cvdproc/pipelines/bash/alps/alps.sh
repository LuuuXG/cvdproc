# ALPS

# REQUIRED INPUTS:
fa_img='' # Input FA image (NIfTI format): -fa_img <path>
output_dir='' # Output directory: -output_dir <path>
alps_dir='' # ALPS script directory: -alps_dir <path>
register_method='flirt' # Registration method: flirt or synthmorph (default: flirt): -register_method <method>

# OPTIONAL INPUTS:
xx_img='' # Input XX image (NIfTI format): -xx_img <path>
yy_img='' # Input YY image (NIfTI format): -yy_img <path>
zz_img='' # Input ZZ image (NIfTI format): -zz_img <path>
tensor_img='' # Input 4D tensor image (NIfTI format): -tensor_img <path> (if provided, will ignore XX, YY, ZZ images)
t1_img='' # Input T1w image (NIfTI format): -t1_img <path>
t1_to_mni_warp='' # Input T1 to MNI warp file (ANTs format): -t1_to_mni_warp <path>

# PARSE ARGUMENTS
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -fa_img)
      fa_img="$2"
      shift; shift
      ;;
    -output_dir)
      output_dir="$2"
      shift; shift
      ;;
    -alps_dir)
      alps_dir="$2"
      shift; shift
      ;;
    -register_method)
      register_method="$2"
      shift; shift
      ;;
    -xx_img)
      xx_img="$2"
      shift; shift
      ;;
    -yy_img)
      yy_img="$2"
      shift; shift
      ;;
    -zz_img)
      zz_img="$2"
      shift; shift
      ;;
    -tensor_img)
      tensor_img="$2"
      shift; shift
      ;;
    -t1_img)
      t1_img="$2"
      shift; shift
      ;;
    -t1_to_mni_warp)
      t1_to_mni_warp="$2"
      shift; shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

#################
# MAIN ANALYSIS #
#################

# make sure output directory exists
mkdir -p "$output_dir"

# make sure system variable $FSLDIR is set
if [ -z "$FSLDIR" ]; then
  echo "Error: FSLDIR environment variable is not set. Please set it to your FSL installation directory."
  exit 1
fi

if [ -n "$t1_img" ]; then
  template_img="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
else
  template_img="${FSLDIR}/data/atlases/JHU/JHU-ICBM-FA-1mm.nii.gz"
fi

# make sure 4 rois exist
rois="${alps_dir}/ROIs_JHU_ALPS/L_SCR.nii.gz,${alps_dir}/ROIs_JHU_ALPS/R_SCR.nii.gz,${alps_dir}/ROIs_JHU_ALPS/L_SLF.nii.gz,${alps_dir}/ROIs_JHU_ALPS/R_SLF.nii.gz"
proj_L=`echo "$rois" | cut -d "," -f1`
proj_R=`echo "$rois" | cut -d "," -f2`
assoc_L=`echo "$rois" | cut -d "," -f3`
assoc_R=`echo "$rois" | cut -d "," -f4`

for roi in $proj_L $proj_R $assoc_L $assoc_R; do
  if [ ! -f "$roi" ]; then
    echo "Error: ROI file not found: $roi"
    exit 1
  fi
done

# handle 4d tensor image input
if [ -n "$tensor_img" ]; then
  echo "Using provided tensor image: $tensor_img"
  fslroi "$tensor_img" "$output_dir/xx.nii.gz" 0 1
  fslroi "$tensor_img" "$output_dir/yy.nii.gz" 3 1
  fslroi "$tensor_img" "$output_dir/zz.nii.gz" 5 1
  xx_img="$output_dir/xx.nii.gz"
  yy_img="$output_dir/yy.nii.gz"
  zz_img="$output_dir/zz.nii.gz"
fi

# registration: provide T1w or use FA image
if [ -z "$t1_img" ]; then
  echo "No T1w image provided, using FA image for registration."
  if [ "$register_method" == "flirt" ]; then
    flirt -in "$fa_img" -ref "$template_img" -out "$output_dir/fa_to_template.nii.gz" -omat "$output_dir/fa_to_template.mat" -dof 12
    # apply same transform to xx, yy, zz images
    flirt -in "$xx_img" -ref "$template_img" -out "$output_dir/xx_to_template.nii.gz" -applyxfm -init "$output_dir/fa_to_template.mat"
    flirt -in "$yy_img" -ref "$template_img" -out "$output_dir/yy_to_template.nii.gz" -applyxfm -init "$output_dir/fa_to_template.mat"
    flirt -in "$zz_img" -ref "$template_img" -out "$output_dir/zz_to_template.nii.gz" -applyxfm -init "$output_dir/fa_to_template.mat"
  elif [ "$register_method" == "synthmorph" ]; then
    mri_synthmorph -t "$output_dir/fa_to_template_warp.nii.gz" "$fa_img" "$template_img" -g
    mri_convert -at "$output_dir/fa_to_template_warp.nii.gz" "$xx_img" "$output_dir/xx_to_template.nii.gz"
    mri_convert -at "$output_dir/fa_to_template_warp.nii.gz" "$yy_img" "$output_dir/yy_to_template.nii.gz"
    mri_convert -at "$output_dir/fa_to_template_warp.nii.gz" "$zz_img" "$output_dir/zz_to_template.nii.gz"
  else
    echo "Error: Unknown registration method: $register_method"
    exit 1
  fi
else
  # mri_synthstrip to skull-strip T1w
  t1_brain="${output_dir}/t1_brain.nii.gz"
  mri_synthstrip -i "$t1_img" -o "$t1_brain" --no-csf

  # T1w and FA registration
  flirt -in "$fa_img" -ref "$t1_brain" -out "$output_dir/fa_to_t1.nii.gz" -omat "$output_dir/fa_to_t1.mat" -dof 12
  flirt -in "$xx_img" -ref "$t1_brain" -out "$output_dir/xx_to_t1.nii.gz" -applyxfm -init "$output_dir/fa_to_t1.mat"
  flirt -in "$yy_img" -ref "$t1_brain" -out "$output_dir/yy_to_t1.nii.gz" -applyxfm -init "$output_dir/fa_to_t1.mat"
  flirt -in "$zz_img" -ref "$t1_brain" -out "$output_dir/zz_to_t1.nii.gz" -applyxfm -init "$output_dir/fa_to_t1.mat"
  # T1w to MNI registration
  if [ "$register_method" == "flirt" ]; then
    flirt -in "$t1_brain" -ref "$template_img" -out "$output_dir/t1_to_template.nii.gz" -omat "$output_dir/t1_to_template.mat" -dof 12
    # apply same transform to xx, yy, zz images
    flirt -in "$output_dir/xx_to_t1.nii.gz" -ref "$template_img" -out "$output_dir/xx_to_template.nii.gz" -applyxfm -init "$output_dir/t1_to_template.mat"
    flirt -in "$output_dir/yy_to_t1.nii.gz" -ref "$template_img" -out "$output_dir/yy_to_template.nii.gz" -applyxfm -init "$output_dir/t1_to_template.mat"
    flirt -in "$output_dir/zz_to_t1.nii.gz" -ref "$template_img" -out "$output_dir/zz_to_template.nii.gz" -applyxfm -init "$output_dir/t1_to_template.mat"
  elif [ "$register_method" == "synthmorph" ]; then
    if [ -n "$t1_to_mni_warp" ]; then
      mri_convert -at "$t1_to_mni_warp" "$output_dir/xx_to_t1.nii.gz" "$output_dir/xx_to_template.nii.gz"
      mri_convert -at "$t1_to_mni_warp" "$output_dir/yy_to_t1.nii.gz" "$output_dir/yy_to_template.nii.gz"
      mri_convert -at "$t1_to_mni_warp" "$output_dir/zz_to_t1.nii.gz" "$output_dir/zz_to_template.nii.gz"
    else
      mri_synthmorph -t "$output_dir/t1_to_template_warp.nii.gz" "$t1_brain" "$template_img" -g
      mri_convert -at "$output_dir/t1_to_template_warp.nii.gz" "$output_dir/xx_to_t1.nii.gz" "$output_dir/xx_to_template.nii.gz"
      mri_convert -at "$output_dir/t1_to_template_warp.nii.gz" "$output_dir/yy_to_t1.nii.gz" "$output_dir/yy_to_template.nii.gz"
      mri_convert -at "$output_dir/t1_to_template_warp.nii.gz" "$output_dir/zz_to_t1.nii.gz" "$output_dir/zz_to_template.nii.gz"
    fi
  else
    echo "Error: Unknown registration method: $register_method"
    exit 1
  fi
fi

xx_in_template_img="$output_dir/xx_to_template.nii.gz"
yy_in_template_img="$output_dir/yy_to_template.nii.gz"
zz_in_template_img="$output_dir/zz_to_template.nii.gz"

# statistics calculation
mkdir -p "${output_dir}/alps.stat"
echo "id,scanner,x_proj_L,x_assoc_L,y_proj_L,z_assoc_L,x_proj_R,x_assoc_R,y_proj_R,z_assoc_R,alps_L,alps_R,alps" > "${output_dir}/alps.stat/alps.csv"

# id: filename of FA image without path and extension (.nii or .nii.gz)
id=$(basename "$fa_img")
id="${id%.nii.gz}"
id="${id%.nii}"
scanner=""

x_proj_L="$(fslstats "${xx_in_template_img}" -k "${proj_L}" -m)"
x_assoc_L="$(fslstats "${xx_in_template_img}" -k "${assoc_L}" -m)"
y_proj_L="$(fslstats "${yy_in_template_img}" -k "${proj_L}" -m)"
z_assoc_L="$(fslstats "${zz_in_template_img}" -k "${assoc_L}" -m)"
x_proj_R="$(fslstats "${xx_in_template_img}" -k "${proj_R}" -m)"
x_assoc_R="$(fslstats "${xx_in_template_img}" -k "${assoc_R}" -m)"
y_proj_R="$(fslstats "${yy_in_template_img}" -k "${proj_R}" -m)"
z_assoc_R="$(fslstats "${zz_in_template_img}" -k "${assoc_R}" -m)"
alps_L=`echo "(($x_proj_L+$x_assoc_L)/2)/(($y_proj_L+$z_assoc_L)/2)" | bc -l` #proj1 and assoc1 are left side, bc -l needed for decimal printing results
alps_R=`echo "(($x_proj_R+$x_assoc_R)/2)/(($y_proj_R+$z_assoc_R)/2)" | bc -l` #proj2 and assoc2 are right side, bc -l needed for decimal printing results
alps=`echo "($alps_R+$alps_L)/2" | bc -l`

echo "${id},${scanner},${x_proj_L},${x_assoc_L},${y_proj_L},${z_assoc_L},${x_proj_R},${x_assoc_R},${y_proj_R},${z_assoc_R},${alps_L},${alps_R},${alps}" >> "${output_dir}/alps.stat/alps.csv"

echo "ALPS calculation completed. Results saved to ${output_dir}/alps.stat/alps.csv"