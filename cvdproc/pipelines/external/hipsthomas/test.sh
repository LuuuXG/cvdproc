#!/bin/bash

# Check for required external environment variable pointing to app home directory
if [ -d "$THOMAS_HOME" ]; then
    RES_PATH=${THOMAS_HOME}/resources
else
    echo "ERROR: The environment variable THOMAS_HOME must be set before this script is run."
    exit 1
fi

function Usage () {
    echo ""
    echo "Usage: $0 -i inputImage [-xf fixedImageMask] [-xm movingImageMask] [-t1] [-co] [-dm] [-sc] [-um] [-v]"
    echo "  where:"
    echo "    -i inputImage = a T1 or WMn image filename (in the current directory) or a filepath."
    echo "    -t1 = use standard T1 image. Triggers the HIPS white matter null synthesis."
    echo "    -oldt1 = use T1 image but don't do HIPS white matter null synthesis. Uses majority voting."
    echo "    -co = crop only"
    echo "    -d  = turn on debugging mode (forces serial processing, retains temp directories)"
    echo "    -dm = use the denoise mask"
    echo "    -sc = Use smaller crop. WARNING: this will not produce reliable results for non-Thalamic structures."
    echo "    -um = use mask"
    echo "    -v = run in Verbose mode"
    echo ""
    echo "Examples:"
    echo "  $0 -i T1.nii.gz -t1"
    echo "  $0 -i my_WMn.nii.gz -v"
    echo ""
}

# Must be at least 1 command line argument
if [ $# -lt 1 ]; then
    Usage
    exit 2
fi

# Parse the command line arguments
fixedImageArg=""
movingImageArg=""

for arg in "$@"; do
    case "$1" in
        -h|--help)
            Usage
            exit 3
            ;;
        -i|--inputImage)
            inputImage="$2"
            shift
            shift
            ;;
        -xf|--fixedImageMask)
            fixedImageArg="-xf $2"
            shift
            shift
            ;;
        -xm|--movingImageMask)
            movingImageArg="-xm $2"
            shift
            shift
            ;;
        -t1|-T1|--t1|--T1)
            t1="-c"
            echo "Input is standard T1; doing HIPS synthesis."
            shift
            ;;
        -oldt1|-oldT1|--oldt1|--oldT1)
            oldt1="--oldt1"
            # mvOnly="--mvOnly"
            echo "Input is standard T1; NOT doing HIPS synthesis."
            shift
            ;;
        -co|--cropOnly)
            cropOnly="-co"
            shift
            ;;
        -d|--debug)
            DEBUG=--debug
            shift
            ;;
        -dm|--denoisem)
            denoisem="-dm"
            shift
            ;;
        -sc|--smallCrop)
            smallCrop="-sc"
            shift
            ;;
        -um|--useMask)
            useMask="-um"
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -*)
            echo "ERROR: Unrecognized argument: '$1'"
            Usage
            exit 3
            ;;
    esac
done

# Test for bad or empty input image filename/path
if [ ! "$inputImage" ]; then
    echo "ERROR: A T1 or WMn image filename (in the current directory) or a file path must be specified."
    Usage
    exit 4
elif [ ! -f "$inputImage" ]; then
    echo "ERROR: Unable to find the specified input image file: '$inputImage'"
    exit 4
fi

if [ "$smallCrop" ]; then
    UNCROP_PADDING=
    CROP_TEMPLATE=templ_93x187x68.nii.gz
else
    UNCROP_PADDING=10
    CROP_TEMPLATE=p15_templ_93x187x68.nii.gz
fi

if [ "$VERBOSE" ]; then
    echo "THOMAS_HOME setting of $THOMAS_HOME"
fi

# Function to return the absolute path to a directory, given a filename or a file path
function absolute_dir_path () {
    local aDir=$(dirname $1)
    \cd $aDir
    echo $PWD
}

# All initial checks have passed, setup variables and functions to be used throughout

# indices of the nuclei to be processed
NUCLEI=(1 2 4 5 6 7 8 9 10 11 12 13 14 26 27 28 29 30 31 32 33 34)

# indices of the nuclei to be fused into a result label image
FUSE_NUCLEI=(2 4 5 6 7 8 9 10 11 12 13 14 26 27 28 29 30 31 32)

 # a few ROIs whose majority voting results are preferred over joint fusion
MV_PREFERRED_ROIS=('5-VLa' '29-GPe' '30-GPi' '32-RN' '33-GP')

# Parse pieces of the input image path for later use
IMAGE_FILENAME=$(basename $inputImage)
# get absolute path of directory containing input image:
IMAGE_DIR=$(absolute_dir_path $inputImage)
# construct absolute path to image file:
INPUT_IMAGE=${IMAGE_DIR}/${IMAGE_FILENAME}

IMAGE_BASENAME=$(basename $IMAGE_FILENAME ".nii.gz")    # this works for .nii.gz (will fail for .nii)
IMAGE_BASENAME=$(basename $IMAGE_BASENAME ".nii")       # this works for .nii (.nii.gz just handled above)

if [ "$VERBOSE" ]; then
    echo "IMAGE_DIR=$IMAGE_DIR, IMAGE_FILE=$IMAGE_FILENAME, IMAGE_BASENAME=$IMAGE_BASENAME"
fi

# These paths are relative to the input image directory but could, later, come from a command line argument.
LEFT_DIR=${IMAGE_DIR}/left
RIGHT_DIR=${IMAGE_DIR}/right
LEFT_TEMP_DIR=${IMAGE_DIR}/temp
RIGHT_TEMP_DIR=${IMAGE_DIR}/tempr
LEFT_EXTRAS_DIR=${LEFT_DIR}/EXTRAS
RIGHT_EXTRAS_DIR=${RIGHT_DIR}/EXTRAS

extraArgs="$t1 $oldt1 $cropOnly $denoisem $mvOnly $smallCrop $useMask $VERBOSE $DEBUG"
echo "Running THOMAS on '$INPUT_IMAGE', outputs to '$IMAGE_DIR', with arguments $extraArgs"


# Function to collect fslstats results for each nucleus.
function make_nucleus_stats () {
    for nucleus in ${NUCLEI[@]}; do
        for nucImage in ${nucleus}-*.nii.gz; do
            nucBase=$(basename $nucImage .nii.gz)
            nucVol=$(fslstats $nucImage -V | awk '{print $2}')
            echo $nucBase $nucVol >> nucleiVols.txt
            if [ "$VERBOSE" ]; then
                echo $nucBase $nucVol
            fi
        done

        for mnucImage in m${nucleus}-*.nii.gz; do
            mnucBase=$(basename $mnucImage .nii.gz)
            mnucVol=$(fslstats $mnucImage -V | awk '{print $2}')
            echo $mnucBase $mnucVol >> nucleiVolsMV.txt
            if [ "$VERBOSE" ]; then
                echo $mnucBase $mnucVol
            fi
        done
    done
}

# Function to move auxiliary results from the left or right directory, saving them in a sub-directory.
function move_extras () {
    local side=$1            # path to source directory (e.g., left, right)
    local extras_dir=$2      # path to target directory (e.g., left/EXTRAS, right/EXTRAS)
    \mv ${side}/mask_inp.nii.gz ${extras_dir}
    \mv ${side}/*Affine.mat ${extras_dir}
    \mv ${side}/san*.nii.gz ${extras_dir}
    \mv ${side}/sthomas_?.nii.gz ${extras_dir}
    \mv ${side}/thomas_?.nii.gz ${extras_dir}
    \mv ${side}/*Warp.nii.gz ${extras_dir}
    if shopt -s nullglob; jfs=(${side}/jf*.nii.gz) && [[ ${#jfs[@]} -gt 0 ]]; then
        \mv ${side}/jf*.nii.gz ${extras_dir}
    fi
    \mv ${side}/*_cropped.nii.gz ${extras_dir}
    \mv ${side}/MV ${extras_dir}
}

# Function to reduce the bit depth for the output nuclei masks
function reduce_nuclei_bit_depth () {
    for nucImage in [0-9]*.nii.gz; do
        fslmaths $nucImage $nucImage -odt char
    done

    for mnucImage in m[0-9]*.nii.gz; do
        fslmaths $mnucImage $mnucImage -odt char
    done
}

# Function to cleanup any leftover temp directory before running THOMAS
function remove_temp_directory () {
    local temp_dir=$1        # path to temp directory to remove
    if [ -d "$temp_dir" ]; then
        if [ "$VERBOSE" ]; then
            echo "Deleting temp directory ${temp_dir}"
        fi
        \rm -rf $temp_dir
    fi
}

# Function to replace joint fusion results with majority voting results, for a few ROIs only
function replace_JF_with_MV () {
    for roi in ${MV_PREFERRED_ROIS[@]}; do
        echo "Using m${roi}.nii.gz instead of jf${roi}.nii.gz for ${roi}"
        \cp ${roi}.nii.gz jf${roi}.nii.gz
        \cp m${roi}.nii.gz ${roi}.nii.gz
    done
}

# Function to move majority voting results from the current directory into their own subdirectory
function save_mv_results () {
    local toDir=${1:-MV}
    mkdir -p $toDir
    \mv m*-* $toDir
    \mv nucleiVolsMV* $toDir
}

# Function to copy THOMAS.py results from the output directory and the
# specified temp directory to the given hemisphere (side) directory.
function save_results () {
    local side=$1            # path to side directory (e.g., left, right)
    local temp_dir=$2        # path to side temp directory (e.g., temp, tempr)
    local output_dir=${3}    # path to output directory
    \cp -rf ${temp_dir}/crop_* ${output_dir}
    \mv ${output_dir}/?-*.nii.gz ${side}
    \mv ${output_dir}/??-*.nii.gz ${side}
    \mv ${output_dir}/m??-*.nii.gz ${side}
    \mv ${output_dir}/*4567* ${side}
    \mv ${output_dir}/6_* ${side}
    \mv ${output_dir}/san* ${side}
    \mv ${output_dir}/crop_* ${side}
    \mv ${temp_dir}/*Warp* ${temp_dir}/*Aff* ${side}
    # The -um option uses an existing mask and, for that to work, the
    # the affine* and mask* should remain in the main directory.
    \cp ${output_dir}/mask* ${side}
    \cp ${output_dir}/rigid* ${side}
}

# Function to uncrop each nucleus to produce a full size image
function uncrop_nuclei () {
    for nucImage in [0-9]*.nii.gz; do
        nucBase=$(basename $nucImage .nii.gz)
        nucCrop="${nucBase}_cropped.nii.gz"
        \mv $nucImage $nucCrop
        uncrop.py $nucCrop $nucImage mask_inp.nii.gz $UNCROP_PADDING
    done

    for mnucImage in m[0-9]*.nii.gz; do
        mnucBase=$(basename $mnucImage .nii.gz)
        mnucCrop="${mnucBase}_cropped.nii.gz"
        \mv $mnucImage $mnucCrop
        uncrop.py $mnucCrop $mnucImage mask_inp.nii.gz $UNCROP_PADDING
    done
}

# Change directory to the directory of the input image to begin running THOMAS
\cd $IMAGE_DIR
echo "CDed to $PWD"

# Cleanup any previous temp directory before running THOMAS
remove_temp_directory $LEFT_TEMP_DIR

# Run THOMAS to segment thalamic nuclei on the LEFT side
THOMAS.py -v $fixedImageArg $movingImageArg -a v2 $extraArgs --tempdir $LEFT_TEMP_DIR $IMAGE_FILENAME ALL

if [ "$cropOnly" ]; then
    echo "Exiting the $0 script after cropping"
    exit 0
fi

# Work in the LEFT hemisphere directory
mkdir -p $LEFT_DIR $LEFT_EXTRAS_DIR
save_results $LEFT_DIR $LEFT_TEMP_DIR $IMAGE_DIR
cd $LEFT_DIR

# For a few ROIs, replace joint fusion with majority voting as it is more robust
if [ "$t1" ]; then
    replace_JF_with_MV
fi

fuselabels.sh "left" ${FUSE_NUCLEI[@]}
# fuselabelsc 1     # TODO: MISSING SCRIPT!

uncrop.py thomas_L.nii.gz thomasfull_L.nii.gz mask_inp.nii.gz ${UNCROP_PADDING}
# uncrop.py basalg_L.nii.gz basalgfull_L.nii.gz mask_inp.nii.gz ${UNCROP_PADDING}

antsApplyTransforms -d 3 -i ${RES_PATH}/${CROP_TEMPLATE} -r crop_${IMAGE_BASENAME}.nii.gz -o regn_L.nii.gz -t \[${IMAGE_BASENAME}0GenericAffine.mat, 1\] -t ${IMAGE_BASENAME}1InverseWarp.nii.gz
swapdimlike.py regn_L.nii.gz ${INPUT_IMAGE} regn_L.nii.gz
swapdimlike.py crop_${IMAGE_BASENAME}.nii.gz ${INPUT_IMAGE} crop_${IMAGE_BASENAME}.nii.gz

# Handle the CL nucleus (directly warp from Marseille space to WMn template to native space by chaining warps)
antsApplyTransforms -d 3 -i ${RES_PATH}/AMICL.nii.gz -r crop_${IMAGE_BASENAME}.nii.gz -o CL_L_cropped.nii.gz -t \[${IMAGE_BASENAME}0GenericAffine.mat, 1\] -t ${IMAGE_BASENAME}1InverseWarp.nii.gz  -t ${RES_PATH}/AMI2WTWarp.nii.gz  -t ${RES_PATH}/AMI2WT0GenericAffine.mat -n NearestNeighbor
swapdimlike.py CL_L_cropped.nii.gz ${INPUT_IMAGE} CL_L_cropped.nii.gz
uncrop.py CL_L_cropped.nii.gz CL_L.nii.gz mask_inp.nii.gz ${UNCROP_PADDING}

if [ "$t1" ]; then
    \cp ${LEFT_TEMP_DIR}/bcrop_${IMAGE_BASENAME}.nii.gz ocrop_${IMAGE_BASENAME}.nii.gz
fi

# Run fslstats on each nucleus and save them
make_nucleus_stats

# Uncrop each nucleus mask to produce a full size mask
uncrop_nuclei

# Reduce the bit depth for all nucleus masks
reduce_nuclei_bit_depth

# Move majority voting results into a subdirectory
save_mv_results

# Return to input image directory
cd $IMAGE_DIR
echo "Done; LEFT segmentation results in directory left"

#
# Now do the RIGHT side
#

# Cleanup any previous temp directory before running THOMAS
remove_temp_directory $RIGHT_TEMP_DIR

# Skip cropping the right side
useMask="-um"
extraArgs="$t1 $oldt1 $cropOnly $denoisem $mvOnly $smallCrop $useMask $VERBOSE $DEBUG"
echo "Skipping crop for right side: using left crop mask"

# Run THOMAS to segment thalamic nuclei on the RIGHT side
THOMAS.py -v $fixedImageArg $movingImageArg -a v2 $extraArgs --right --tempdir tempr $inputImage ALL

# Work in the RIGHT hemisphere directory
mkdir -p $RIGHT_DIR $RIGHT_EXTRAS_DIR
save_results $RIGHT_DIR $RIGHT_TEMP_DIR $IMAGE_DIR
cd $RIGHT_DIR

# For a few ROIs, replace joint fusion with majority voting as it is more robust
if [ "$t1" ]; then
    replace_JF_with_MV
fi

fuselabels.sh "right" ${FUSE_NUCLEI[@]}
# fuselabelsc 2     # TODO: MISSING SCRIPT!

uncrop.py thomas_R.nii.gz thomasfull_R.nii.gz mask_inp.nii.gz ${UNCROP_PADDING}
# uncrop.py basalg_R.nii.gz basalgfull_R.nii.gz mask_inp.nii.gz ${UNCROP_PADDING}

antsApplyTransforms -d 3 -i ${RES_PATH}/${CROP_TEMPLATE} -r ${RIGHT_TEMP_DIR}/crop_${IMAGE_BASENAME}.nii.gz -o regn_R.nii.gz -t \[${IMAGE_BASENAME}0GenericAffine.mat, 1\] -t ${IMAGE_BASENAME}1InverseWarp.nii.gz

# Handle CL separately
antsApplyTransforms -d 3 -i ${RES_PATH}/AMICL.nii.gz -r ${RIGHT_TEMP_DIR}/crop_${IMAGE_BASENAME}.nii.gz -o CL_R_cropped.nii.gz -t \[${IMAGE_BASENAME}0GenericAffine.mat, 1\] -t ${IMAGE_BASENAME}1InverseWarp.nii.gz  -t ${RES_PATH}/AMI2WTWarp.nii.gz  -t ${RES_PATH}/AMI2WT0GenericAffine.mat -n NearestNeighbor
fslswapdim CL_R_cropped.nii.gz -x y z CL_R_cropped.nii.gz
swapdimlike.py CL_R_cropped.nii.gz ${INPUT_IMAGE} CL_R_cropped.nii.gz

uncrop.py CL_R_cropped.nii.gz CL_R.nii.gz mask_inp.nii.gz ${UNCROP_PADDING}

fslswapdim regn_R.nii.gz -x y z regn_R.nii.gz
swapdimlike.py regn_R.nii.gz ${INPUT_IMAGE} regn_R.nii.gz
\cp ${RIGHT_TEMP_DIR}/crop_${IMAGE_BASENAME}.nii.gz rcrop_${IMAGE_BASENAME}.nii.gz
fslswapdim rcrop_${IMAGE_BASENAME}.nii.gz -x y z rcrop_${IMAGE_BASENAME}.nii.gz
swapdimlike.py rcrop_${IMAGE_BASENAME}.nii.gz ${INPUT_IMAGE} rcrop_${IMAGE_BASENAME}.nii.gz

# match the left side convention
if [ "$t1" ]; then
    \mv crop_${IMAGE_BASENAME}.nii.gz ocrop_${IMAGE_BASENAME}.nii.gz
fi
\mv rcrop_${IMAGE_BASENAME}.nii.gz crop_${IMAGE_BASENAME}.nii.gz

# run fslstats on each nucleus and save them
make_nucleus_stats

# Uncrop each nucleus mask to produce a full size mask
uncrop_nuclei

# Reduce the bit depth for all nucleus masks
reduce_nuclei_bit_depth

# Move majority voting results into a subdirectory
save_mv_results

echo "Done; RIGHT segmentation results in directory right"

# Return to input image directory
cd $IMAGE_DIR

echo "Creating sthomas_LR_labels.nii.gz and sthomas_LR_labels.png"

# Combine left and right labels into a single label set
fslmaths left/sthomas_L.nii.gz -add right/sthomas_R.nii.gz sthomas_LR_labels.nii.gz

# Create the cool qc png file
qcplot.py

echo "Done; Combined labels and QC files in top-level output directory"

# Remove temp directories and extra files
if [ -z "$DEBUG" ]; then
    remove_temp_directory $LEFT_TEMP_DIR
    remove_temp_directory $RIGHT_TEMP_DIR
    \rm -f rigid0GenericAffine.mat mask_inp.nii.gz
fi

# Move processing by-products to an extras directory, leaving only the most important results
move_extras $LEFT_DIR $LEFT_EXTRAS_DIR
move_extras $RIGHT_DIR $RIGHT_EXTRAS_DIR