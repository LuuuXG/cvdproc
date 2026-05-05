# ALPS

# REQUIRED INPUTS:
fa_img='' # Input FA image (NIfTI format): -fa_img <path>
output_dir='' # Output directory: -output_dir <path>
alps_dir='' # ALPS script directory: -alps_dir <path>
register_method='flirt' # Registration method: flirt or synthmorph: -register_method <method>

# OPTIONAL INPUTS:
xx_img='' # Input XX image (NIfTI format): -xx_img <path>
yy_img='' # Input YY image (NIfTI format): -yy_img <path>
zz_img='' # Input ZZ image (NIfTI format): -zz_img <path>
tensor_img='' # Input 4D tensor image (NIfTI format): -tensor_img <path>
t1_img='' # Input T1w image (NIfTI format): -t1_img <path>
fa_to_t1w_affine='' # Input FA to T1w affine matrix: -fa_to_t1w_affine <path>
t1_to_mni_warp='' # Input T1 to MNI warp file: -t1_to_mni_warp <path>

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
    -fa_to_t1w_affine)
      fa_to_t1w_affine="$2"
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

mkdir -p "$output_dir"

if [ -z "$FSLDIR" ]; then
  echo "Error: FSLDIR environment variable is not set. Please set it to your FSL installation directory."
  exit 1
fi

if [ -z "$fa_img" ]; then
  echo "Error: -fa_img is required."
  exit 1
fi

if [ -z "$output_dir" ]; then
  echo "Error: -output_dir is required."
  exit 1
fi

if [ -z "$alps_dir" ]; then
  echo "Error: -alps_dir is required."
  exit 1
fi

if [ -n "$tensor_img" ]; then
  echo "Using provided tensor image: $tensor_img"
  fslroi "$tensor_img" "$output_dir/xx.nii.gz" 0 1
  fslroi "$tensor_img" "$output_dir/yy.nii.gz" 3 1
  fslroi "$tensor_img" "$output_dir/zz.nii.gz" 5 1
  xx_img="$output_dir/xx.nii.gz"
  yy_img="$output_dir/yy.nii.gz"
  zz_img="$output_dir/zz.nii.gz"
fi

if [ -z "$xx_img" ] || [ -z "$yy_img" ] || [ -z "$zz_img" ]; then
  echo "Error: provide either -tensor_img or all of -xx_img, -yy_img, and -zz_img."
  exit 1
fi

if [ -n "$t1_img" ]; then
  template_img="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
else
  template_img="${FSLDIR}/data/atlases/JHU/JHU-ICBM-FA-1mm.nii.gz"
fi

rois="${alps_dir}/ROIs_JHU_ALPS/L_SCR.nii.gz,${alps_dir}/ROIs_JHU_ALPS/R_SCR.nii.gz,${alps_dir}/ROIs_JHU_ALPS/L_SLF.nii.gz,${alps_dir}/ROIs_JHU_ALPS/R_SLF.nii.gz"
proj_L="$(echo "$rois" | cut -d "," -f1)"
proj_R="$(echo "$rois" | cut -d "," -f2)"
assoc_L="$(echo "$rois" | cut -d "," -f3)"
assoc_R="$(echo "$rois" | cut -d "," -f4)"

for roi in "$proj_L" "$proj_R" "$assoc_L" "$assoc_R"; do
  if [ ! -f "$roi" ]; then
    echo "Error: ROI file not found: $roi"
    exit 1
  fi
done

if [ -z "$t1_img" ]; then
  echo "No T1w image provided, using FA image for registration."

  if [ "$register_method" == "flirt" ]; then
    flirt -in "$fa_img" -ref "$template_img" -out "$output_dir/fa_to_template.nii.gz" -omat "$output_dir/fa_to_template.mat" -dof 12

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
  t1_brain="${output_dir}/t1_brain.nii.gz"
  mri_synthstrip -i "$t1_img" -o "$t1_brain" --no-csf

  if [ -n "$fa_to_t1w_affine" ]; then
    if [ ! -f "$fa_to_t1w_affine" ]; then
      echo "Error: FA to T1w affine matrix not found: $fa_to_t1w_affine"
      exit 1
    fi

    echo "Using provided FA to T1w affine matrix: $fa_to_t1w_affine"
    cp "$fa_to_t1w_affine" "$output_dir/fa_to_t1.mat"

    flirt -in "$fa_img" -ref "$t1_brain" -out "$output_dir/fa_to_t1.nii.gz" -applyxfm -init "$output_dir/fa_to_t1.mat"
  else
    echo "No FA to T1w affine matrix provided, running FLIRT FA to T1w registration."
    flirt -in "$fa_img" -ref "$t1_brain" -out "$output_dir/fa_to_t1.nii.gz" -omat "$output_dir/fa_to_t1.mat" -dof 12
  fi

  flirt -in "$xx_img" -ref "$t1_brain" -out "$output_dir/xx_to_t1.nii.gz" -applyxfm -init "$output_dir/fa_to_t1.mat"
  flirt -in "$yy_img" -ref "$t1_brain" -out "$output_dir/yy_to_t1.nii.gz" -applyxfm -init "$output_dir/fa_to_t1.mat"
  flirt -in "$zz_img" -ref "$t1_brain" -out "$output_dir/zz_to_t1.nii.gz" -applyxfm -init "$output_dir/fa_to_t1.mat"

  if [ "$register_method" == "flirt" ]; then
    flirt -in "$t1_brain" -ref "$template_img" -out "$output_dir/t1_to_template.nii.gz" -omat "$output_dir/t1_to_template.mat" -dof 12

    flirt -in "$output_dir/xx_to_t1.nii.gz" -ref "$template_img" -out "$output_dir/xx_to_template.nii.gz" -applyxfm -init "$output_dir/t1_to_template.mat"
    flirt -in "$output_dir/yy_to_t1.nii.gz" -ref "$template_img" -out "$output_dir/yy_to_template.nii.gz" -applyxfm -init "$output_dir/t1_to_template.mat"
    flirt -in "$output_dir/zz_to_t1.nii.gz" -ref "$template_img" -out "$output_dir/zz_to_template.nii.gz" -applyxfm -init "$output_dir/t1_to_template.mat"

  elif [ "$register_method" == "synthmorph" ]; then
    if [ -n "$t1_to_mni_warp" ]; then
      if [ ! -f "$t1_to_mni_warp" ]; then
        echo "Error: T1w to MNI warp file not found: $t1_to_mni_warp"
        exit 1
      fi

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
echo "id,scanner,x_proj_L,x_assoc_L,y_proj_L,z_assoc_L,x_proj_R,x_assoc_R,y_proj_R,z_assoc_R,alps_L,alps_R,alps,g_alps_L,g_alps_R,g_alps_avg,g_alps_sum" > "${output_dir}/alps.stat/alps.csv"

id="$(basename "$fa_img")"
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

alps_L="$(echo "(($x_proj_L+$x_assoc_L)/2)/(($y_proj_L+$z_assoc_L)/2)" | bc -l)"
alps_R="$(echo "(($x_proj_R+$x_assoc_R)/2)/(($y_proj_R+$z_assoc_R)/2)" | bc -l)"
alps="$(echo "($alps_R+$alps_L)/2" | bc -l)"

g_alps_L="$(echo "(($x_proj_L+$x_assoc_L)-$y_proj_L)/$z_assoc_L" | bc -l)"
g_alps_R="$(echo "(($x_proj_R+$x_assoc_R)-$y_proj_R)/$z_assoc_R" | bc -l)"
g_alps_avg="$(echo "($g_alps_L+$g_alps_R)/2" | bc -l)"

x_proj_LR="$(echo "$x_proj_L+$x_proj_R" | bc -l)"
x_assoc_LR="$(echo "$x_assoc_L+$x_assoc_R" | bc -l)"
y_proj_LR="$(echo "$y_proj_L+$y_proj_R" | bc -l)"
z_assoc_LR="$(echo "$z_assoc_L+$z_assoc_R" | bc -l)"

g_alps_sum="$(echo "(($x_proj_LR+$x_assoc_LR)-$y_proj_LR)/$z_assoc_LR" | bc -l)"

echo "${id},${scanner},${x_proj_L},${x_assoc_L},${y_proj_L},${z_assoc_L},${x_proj_R},${x_assoc_R},${y_proj_R},${z_assoc_R},${alps_L},${alps_R},${alps},${g_alps_L},${g_alps_R},${g_alps_avg},${g_alps_sum}" >> "${output_dir}/alps.stat/alps.csv"

echo "ALPS and G-ALPS calculation completed. Results saved to ${output_dir}/alps.stat/alps.csv"