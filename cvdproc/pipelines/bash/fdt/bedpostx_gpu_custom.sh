#!/bin/bash
export LC_ALL=C

Usage() {
    echo ""
    echo "Usage:"
    echo "  bedpostx_gpu_custom.sh --dwi <data.nii[.gz]> --bvec <bvecs> --bval <bvals> --mask <mask.nii[.gz]> --out-dir <output_dir> [options] [xfibres options...]"
    echo ""
    echo "Required:"
    echo "  --dwi       Path to DWI image (4D)"
    echo "  --bvec      Path to bvecs"
    echo "  --bval      Path to bvals"
    echo "  --mask      Path to nodif_brain_mask image"
    echo "  --out-dir   Output directory (final results will be written here; no .bedpostX suffix)"
    echo ""
    echo "Optional (bedpostx-like):"
    echo "  -NJOBS      Number of jobs/parts (default 4)"
    echo "  -n          Number of fibres per voxel (default 3)"
    echo "  -w          ARD weight/fudge (default 1)"
    echo "  -b          Burnin (default 1000)"
    echo "  -j          Number of jumps (default 1250)"
    echo "  -s          Sample every (default 25)"
    echo "  -model      1 sticks, 2 sticks+range (default), 3 zeppelins"
    echo "  --grad-dev  Path to gradient nonlinearity file (optional; enables grad nonlin)"
    echo ""
    echo "You may also pass xfibres options directly (e.g. --noard --cnonlinear)."
    echo "Note: This version runs sequentially without fsl_sub/queue system."
    echo ""
    exit 1
}

make_absolute() {
    local p="$1"
    if [ -z "$p" ]; then
        echo ""
        return
    fi
    if [ -d "$p" ]; then
        (cd "$p" && pwd)
    else
        if [[ "$p" == /* ]]; then
            echo "$p"
        else
            echo "$(pwd)/$p"
        fi
    fi
}

im_exists() {
    "${FSLDIR}/bin/imtest" "$1" 2>/dev/null
}

[ "$1" = "" ] && Usage

# -------------------------
# Defaults
# -------------------------
njobs=4
nfibres=3
fudge=1
burnin=1000
njumps=1250
sampleevery=25
model=2
gflag=0
grad_dev=""
other=()

dwi=""
bvec=""
bval=""
mask=""
out_dir=""

# -------------------------
# Parse args
# -------------------------
while [ -n "$1" ]; do
  case "$1" in
      --dwi)      dwi="$2"; shift 2;;
      --bvec)     bvec="$2"; shift 2;;
      --bval)     bval="$2"; shift 2;;
      --mask)     mask="$2"; shift 2;;
      --out-dir)  out_dir="$2"; shift 2;;

      --grad-dev) grad_dev="$2"; gflag=1; shift 2;;

      -NJOBS) njobs="$2"; shift 2;;
      -n)     nfibres="$2"; shift 2;;
      -w)     fudge="$2"; shift 2;;
      -b)     burnin="$2"; shift 2;;
      -j)     njumps="$2"; shift 2;;
      -s)     sampleevery="$2"; shift 2;;
      -model) model="$2"; shift 2;;

      -h|--help) Usage;;

      *) other+=("$1"); shift;;
  esac
done

# -------------------------
# Validate required
# -------------------------
[ -z "$dwi" ] && echo "ERROR: --dwi is required" >&2 && Usage
[ -z "$bvec" ] && echo "ERROR: --bvec is required" >&2 && Usage
[ -z "$bval" ] && echo "ERROR: --bval is required" >&2 && Usage
[ -z "$mask" ] && echo "ERROR: --mask is required" >&2 && Usage
[ -z "$out_dir" ] && echo "ERROR: --out-dir is required" >&2 && Usage

dwi=$(make_absolute "$dwi")
bvec=$(make_absolute "$bvec")
bval=$(make_absolute "$bval")
mask=$(make_absolute "$mask")
out_dir=$(make_absolute "$out_dir")

if [ ! -e "$dwi" ]; then
    echo "ERROR: DWI not found: $dwi" >&2
    exit 1
fi
if [ ! -e "$bvec" ]; then
    echo "ERROR: bvec not found: $bvec" >&2
    exit 1
fi
if [ ! -e "$bval" ]; then
    echo "ERROR: bval not found: $bval" >&2
    exit 1
fi
if [ "$(im_exists "$mask")" -eq 0 ]; then
    echo "ERROR: mask not found or not an image: $mask" >&2
    exit 1
fi

if [ $gflag -eq 1 ]; then
    grad_dev=$(make_absolute "$grad_dev")
    if [ "$(im_exists "$grad_dev")" -eq 0 ]; then
        echo "ERROR: grad_dev not found or not an image: $grad_dev" >&2
        exit 1
    fi
fi

# -------------------------
# Build xfibres options
# -------------------------
opts=("--nf=$nfibres" "--fudge=$fudge" "--bi=$burnin" "--nj=$njumps" "--se=$sampleevery" "--model=$model")
defopts=("--cnonlinear")
opts=("${opts[@]}" "${defopts[@]}" "${other[@]}")

echo "---------------------------------------------"
echo "------------ BedpostX GPU (no fsl_sub) ------"
echo "---------------------------------------------"
echo "DWI:      $dwi"
echo "bvecs:    $bvec"
echo "bvals:    $bval"
echo "mask:     $mask"
if [ $gflag -eq 1 ]; then
  echo "grad_dev: $grad_dev"
fi
echo "out_dir:  $out_dir"

# -------------------------
# Output checks
# -------------------------
if [ -e "${out_dir}/xfms/eye.mat" ]; then
    echo "ERROR: Output directory appears already processed: ${out_dir}" >&2
    echo "Delete or rename ${out_dir} before repeating." >&2
    exit 1
fi

# -------------------------
# Prepare directories
# -------------------------
echo "Making output directory structure"
mkdir -p "${out_dir}/diff_parts" \
         "${out_dir}/logs/logs_gpu" \
         "${out_dir}/logs/monitor" \
         "${out_dir}/xfms"
rm -f "${out_dir}/logs/monitor/"* 2>/dev/null || true

echo "Staging bvecs/bvals and mask into output directory"
cp "$bvec" "${out_dir}/bvecs"
cp "$bval" "${out_dir}/bvals"
"${FSLDIR}/bin/imcp" "$mask" "${out_dir}/nodif_brain_mask"

# Stage DWI as out_dir/data (FSL image prefix without extension)
# This avoids relying on any "subject directory" convention.
"${FSLDIR}/bin/imcp" "$dwi" "${out_dir}/data"

# If nodif exists in the same space, user can provide it as xfibres option later.
# We keep original behavior for nodif only if user already staged it manually.
if [ "$(im_exists "${out_dir}/nodif")" = "1" ] ; then
    "${FSLDIR}/bin/fslmaths" "${out_dir}/nodif" -mas "${out_dir}/nodif_brain_mask" "${out_dir}/nodif_brain"
fi

# -------------------------
# Split dataset into parts
# -------------------------
echo "Pre-processing stage"
if [ $gflag -eq 1 ]; then
    "${FSLDIR}/bin/split_parts_gpu" \
        "${out_dir}/data" \
        "${out_dir}/nodif_brain_mask" \
        "${out_dir}/bvals" \
        "${out_dir}/bvecs" \
        "$grad_dev" 1 "$njobs" "${out_dir}"
else
    "${FSLDIR}/bin/split_parts_gpu" \
        "${out_dir}/data" \
        "${out_dir}/nodif_brain_mask" \
        "${out_dir}/bvals" \
        "${out_dir}/bvecs" \
        NULL 0 "$njobs" "${out_dir}"
fi
split_status=$?
if [ $split_status -ne 0 ]; then
    echo "ERROR: split_parts_gpu failed (exit $split_status)." >&2
    exit $split_status
fi

# Count voxels in mask
nvox=$("${FSLDIR}/bin/fslstats" "${out_dir}/nodif_brain_mask" -V | cut -d ' ' -f1)

# -------------------------
# Sequential run (no fsl_sub)
# -------------------------
echo "Queuing parallel processing stage (sequential run; no fsl_sub)"
part=0
while [ $part -lt "$njobs" ]; do
    partzp=$("$FSLDIR/bin/zeropad" $part 4)

    if [ $gflag -eq 1 ]; then
        gopts=("${opts[@]}" "--gradnonlin=${out_dir}/grad_dev_${part}")
    else
        gopts=("${opts[@]}")
    fi

    echo "Running xfibres_gpu for part $part (log: ${out_dir}/diff_parts/data_part_${partzp})"
    "${FSLDIR}/bin/xfibres_gpu" \
        --data="${out_dir}/data_${part}" \
        --mask="${out_dir}/nodif_brain_mask" \
        -b "${out_dir}/bvals" \
        -r "${out_dir}/bvecs" \
        --forcedir \
        --logdir="${out_dir}/diff_parts/data_part_${partzp}" \
        "${gopts[@]}" "${out_dir}" "$part" "$njobs" "$nvox"
    status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: xfibres_gpu failed on part $part (exit $status)." >&2
        exit $status
    fi

    echo "$((part+1)) parts processed out of $njobs"
    touch "${out_dir}/logs/monitor/${part}"

    part=$((part + 1))
done

# -------------------------
# Post-processing (sequential run)
# -------------------------
echo "All parts finished. Running post-processing stage (no fsl_sub)"
echo "Log directory is: ${out_dir}/diff_parts"
echo "${out_dir}/nodif_brain_mask"

post_gopts=("${opts[@]}")
"${FSLDIR}/bin/bedpostx_postproc_gpu.sh" \
    "--data=${out_dir}/data" \
    "--mask=${out_dir}/nodif_brain_mask" \
    -b "${out_dir}/bvals" \
    -r "${out_dir}/bvecs" \
    --forcedir \
    "--logdir=${out_dir}/diff_parts" \
    "${post_gopts[@]}" \
    "$nvox" "$njobs" "${out_dir}" "${FSLDIR}"
status=$?
if [ $status -ne 0 ]; then
    echo "ERROR: bedpostx_postproc_gpu.sh failed (exit $status)." >&2
    exit $status
fi

echo "Creating identity xfm"
mkdir -p "${out_dir}/xfms"
{
  echo "1 0 0 0"
  echo "0 1 0 0"
  echo "0 0 1 0"
} > "${out_dir}/xfms/eye.mat"

echo "Done"
