#!/bin/bash

#   Copyright (C) 2004 University of Oxford
#
#   Part of FSL - FMRIB's Software Library
#   http://www.fmrib.ox.ac.uk/fsl
#   fsl@fmrib.ox.ac.uk
#
#   Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
#   Imaging of the Brain), Department of Clinical Neurology, Oxford
#   University, Oxford, UK
#
#
#   LICENCE
#
#   FMRIB Software Library, Release 6.0 (c) 2018, The University of
#   Oxford (the "Software")
#
#   The Software remains the property of the Oxford University Innovation
#   ("the University").
#
#   The Software is distributed "AS IS" under this Licence solely for
#   non-commercial use in the hope that it will be useful, but in order
#   that the University as a charitable foundation protects its assets for
#   the benefit of its educational and research purposes, the University
#   makes clear that no condition is made or to be implied, nor is any
#   warranty given or to be implied, as to the accuracy of the Software,
#   or that it will be suitable for any particular purpose or for use
#   under any specific conditions. Furthermore, the University disclaims
#   all responsibility for the use which is made of the Software. It
#   further disclaims any liability for the outcomes arising from using
#   the Software.
#
#   The Licensee agrees to indemnify the University and hold the
#   University harmless from and against any and all claims, damages and
#   liabilities asserted by third parties (including claims for
#   negligence) which arise directly or indirectly from the use of the
#   Software or the sale of any products based on the Software.
#
#   No part of the Software may be reproduced, modified, transmitted or
#   transferred in any form or by any means, electronic or mechanical,
#   without the express permission of the University. The permission of
#   the University is not required if the said reproduction, modification,
#   transmission or transference is done without financial return, the
#   conditions of this Licence are imposed upon the receiver of the
#   product, and all original and amended source code is included in any
#   transmitted product. You may be held legally responsible for any
#   copyright infringement that is caused or encouraged by your failure to
#   abide by these terms and conditions.
#
#   You are not permitted under this Licence to use this Software
#   commercially. Use for which any financial return is received shall be
#   defined as commercial use, and includes (1) integration of all or part
#   of the source code or the Software into a product for sale or license
#   by or on behalf of Licensee to third parties or (2) use of the
#   Software or any derivative of it for research with the final aim of
#   developing software products for sale or license to a third party or
#   (3) use of the Software or any derivative of it for research with the
#   final aim of developing non-software products for sale or license to a
#   third party, or (4) use of the Software to provide any service to an
#   external organisation for which payment is received. If you are
#   interested in using the Software commercially, please contact Oxford
#   University Innovation ("OUI"), the technology transfer company of the
#   University, to negotiate a licence. Contact details are:
#   fsl@innovation.ox.ac.uk quoting Reference Project 9564, FSL.
export LC_ALL=C

Usage() {
    echo ""
    echo "Usage: bedpostx_gpu <subject_directory> [options]"
    echo ""
    echo "expects to find bvals and bvecs in subject directory"
    echo "expects to find data and nodif_brain_mask in subject directory"
    echo "expects to find grad_dev in subject directory, if -g is set"
    echo ""
    echo "<options>:"
    echo "-NJOBS   (number of jobs/parts, default 4)"
    echo "-n       (number of fibres per voxel, default 3)"
    echo "-w       (ARD weight, default 1)"
    echo "-b       (burnin, default 1000)"
    echo "-j       (number of jumps, default 1250)"
    echo "-s       (sample every, default 25)"
    echo "-model   (1 sticks, 2 sticks+range (default), 3 zeppelins)"
    echo "-g       (use gradient nonlinearity file grad_dev, default off)"
    echo ""
    echo "You may also pass xfibres options directly (e.g. --noard --cnonlinear)."
    echo "Note: This version runs sequentially without fsl_sub/queue system."
    exit 1
}

make_absolute(){
    dir=$1
    if [ -d "${dir}" ]; then
        OLDWD=$(pwd)
        cd "${dir}"
        dir_all=$(pwd)
        cd "$OLDWD"
    else
        dir_all="${dir}"
    fi
    echo "${dir_all}"
}

[ "$1" = "" ] && Usage

subjdir=$(make_absolute "$1")
subjdir=$(echo "$subjdir" | sed 's/\/$/$/g')

echo "---------------------------------------------"
echo "------------ BedpostX GPU (no fsl_sub) ------"
echo "---------------------------------------------"
echo "subjectdir is $subjdir"

# -------------------------
# Parse options (no queues)
# -------------------------
njobs=4
nfibres=3
fudge=1
burnin=1000
njumps=1250
sampleevery=25
model=2
gflag=0
other=()

shift
while [ -n "$1" ]; do
  case "$1" in
      -NJOBS) njobs=$2; shift;;
      -n)     nfibres=$2; shift;;
      -w)     fudge=$2; shift;;
      -b)     burnin=$2; shift;;
      -j)     njumps=$2; shift;;
      -s)     sampleevery=$2; shift;;
      -model) model=$2; shift;;
      -g)     gflag=1;;
      *)      other+=("$1");;
  esac
  shift
done

# Make xfibres options
opts=("--nf=$nfibres" "--fudge=$fudge" "--bi=$burnin" "--nj=$njumps" "--se=$sampleevery" "--model=$model")
defopts=("--cnonlinear")
opts=("${opts[@]}" "${defopts[@]}" "${other[@]}")

# -------------------------
# Sanity checks
# -------------------------
if [ ! -d "$subjdir" ]; then
    echo "subject directory not found: $subjdir" >&2
    exit 1
fi

if [ ! -e "${subjdir}/bvecs" ]; then
    if [ -e "${subjdir}/bvecs.txt" ]; then
        mv "${subjdir}/bvecs.txt" "${subjdir}/bvecs"
    else
        echo "${subjdir}/bvecs not found" >&2
        exit 1
    fi
fi

if [ ! -e "${subjdir}/bvals" ]; then
    if [ -e "${subjdir}/bvals.txt" ]; then
        mv "${subjdir}/bvals.txt" "${subjdir}/bvals"
    else
        echo "${subjdir}/bvals not found" >&2
        exit 1
    fi
fi

if [ $("${FSLDIR}/bin/imtest" "${subjdir}/data") -eq 0 ]; then
    echo "${subjdir}/data not found" >&2
    exit 1
fi

if [ ${gflag} -eq 1 ]; then
    if [ $("${FSLDIR}/bin/imtest" "${subjdir}/grad_dev") -eq 0 ]; then
        echo "${subjdir}/grad_dev not found (required by -g)" >&2
        exit 1
    fi
fi

if [ $("${FSLDIR}/bin/imtest" "${subjdir}/nodif_brain_mask") -eq 0 ]; then
    echo "${subjdir}/nodif_brain_mask not found" >&2
    exit 1
fi

if [ -e "${subjdir}.bedpostX/xfms/eye.mat" ]; then
    echo "${subjdir} has already been processed: ${subjdir}.bedpostX." >&2
    echo "Delete or rename ${subjdir}.bedpostX before repeating." >&2
    exit 1
fi

# -------------------------
# Prepare directories
# -------------------------
echo "Making bedpostx directory structure"
mkdir -p "${subjdir}.bedpostX/diff_parts" \
         "${subjdir}.bedpostX/logs/logs_gpu" \
         "${subjdir}.bedpostX/logs/monitor" \
         "${subjdir}.bedpostX/xfms"
rm -f "${subjdir}.bedpostX/logs/monitor/"*

echo "Copying files to bedpost directory"
cp "${subjdir}/bvecs" "${subjdir}/bvals" "${subjdir}.bedpostX"
"${FSLDIR}/bin/imcp" "${subjdir}/nodif_brain_mask" "${subjdir}.bedpostX"

if [ $("${FSLDIR}/bin/imtest" "${subjdir}/nodif") = "1" ] ; then
    "${FSLDIR}/bin/fslmaths" "${subjdir}/nodif" -mas "${subjdir}/nodif_brain_mask" "${subjdir}.bedpostX/nodif_brain"
fi

# -------------------------
# Split dataset into parts
# -------------------------
echo "Pre-processing stage"
if [ ${gflag} -eq 1 ]; then
    "${FSLDIR}/bin/split_parts_gpu" \
        "${subjdir}/data" \
        "${subjdir}/nodif_brain_mask" \
        "${subjdir}.bedpostX/bvals" \
        "${subjdir}.bedpostX/bvecs" \
        "${subjdir}/grad_dev" 1 "$njobs" "${subjdir}.bedpostX"
else
    "${FSLDIR}/bin/split_parts_gpu" \
        "${subjdir}/data" \
        "${subjdir}/nodif_brain_mask" \
        "${subjdir}.bedpostX/bvals" \
        "${subjdir}.bedpostX/bvecs" \
        NULL 0 "$njobs" "${subjdir}.bedpostX"
fi
split_status=$?
if [ $split_status -ne 0 ]; then
    echo "split_parts_gpu failed (exit $split_status)." >&2
    exit $split_status
fi

# Count the number of voxels in the nodif_brain_mask
nvox=$("${FSLDIR}/bin/fslstats" "$subjdir.bedpostX/nodif_brain_mask" -V | cut -d ' ' -f1)

# -------------------------
# Sequential run (no fsl_sub)
# -------------------------
echo "Queuing parallel processing stage (sequential run; no fsl_sub)"
part=0
while [ $part -lt "$njobs" ]; do
    partzp=$("$FSLDIR/bin/zeropad" $part 4)

    if [ ${gflag} -eq 1 ]; then
        gopts=("${opts[@]}" "--gradnonlin=${subjdir}.bedpostX/grad_dev_$part")
    else
        gopts=("${opts[@]}")
    fi

    echo "Running xfibres_gpu for part $part (log: ${subjdir}.bedpostX/diff_parts/data_part_${partzp})"
    "${FSLDIR}/bin/xfibres_gpu" \
        --data="${subjdir}.bedpostX/data_${part}" \
        --mask="${subjdir}.bedpostX/nodif_brain_mask" \
        -b "${subjdir}.bedpostX/bvals" \
        -r "${subjdir}.bedpostX/bvecs" \
        --forcedir \
        --logdir="${subjdir}.bedpostX/diff_parts/data_part_${partzp}" \
        "${gopts[@]}" "${subjdir}" "$part" "$njobs" "$nvox"
    status=$?
    if [ $status -ne 0 ]; then
        echo "xfibres_gpu failed on part $part (exit $status)." >&2
        exit $status
    fi

    echo "$((part+1)) parts processed out of $njobs"

    touch "${subjdir}.bedpostX/logs/monitor/${part}"

    part=$((part + 1))
done

# -------------------------
# Post-processing (sequential run)
# -------------------------
echo "All parts finished. Running post-processing stage (no fsl_sub)"
echo "Log directory is: ${subjdir}.bedpostX/diff_parts"
echo "${subjdir}.bedpostX/nodif_brain_mask"

post_gopts=("${opts[@]}") 
"${FSLDIR}/bin/bedpostx_postproc_gpu.sh" \
    "--data=${subjdir}/data" \
    "--mask=${subjdir}.bedpostX/nodif_brain_mask" \
    -b "${subjdir}.bedpostX/bvals" \
    -r "${subjdir}.bedpostX/bvecs" \
    --forcedir \
    "--logdir=${subjdir}.bedpostX/diff_parts" \
    "${post_gopts[@]}" \
    "$nvox" "$njobs" "${subjdir}" "${FSLDIR}"
status=$?
if [ $status -ne 0 ]; then
    echo "bedpostx_postproc_gpu.sh failed (exit $status)." >&2
    exit $status
fi

echo "Removing intermediate files"
echo "Creating identity xfm"
mkdir -p "${subjdir}.bedpostX/xfms"
echo "1 0 0 0" > "${subjdir}.bedpostX/xfms/eye.mat"
echo "0 1 0 0" >> "${subjdir}.bedpostX/xfms/eye.mat"
echo "0 0 1 0" >> "${subjdir}.bedpostX/xfms/eye.mat"
echo "Done"
echo "Done. (no fsl_sub)"
