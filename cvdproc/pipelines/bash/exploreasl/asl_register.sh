#!/bin/bash

set -euo pipefail

asl_space_img=$1
asl_space_t1w_img=$2
target_t1w_img=$3
# output
asl_in_t1w_img=$4

output_dir=$(dirname $target_t1w_img)
mkdir -p $output_dir

flirt \
    -in $asl_space_img \
    -ref $asl_space_t1w_img \
    -out $asl_in_t1w_img \
    -applyxfm \
    -usesqform \
    -interp trilinear

fslcpgeom $target_t1w_img $asl_in_t1w_img