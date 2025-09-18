#!/usr/bin/env bash

set -e

input_image=$1
output_image=$2
output_bias_field=$3

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <input_image> <output_image> <output_bias_field>"
  exit 1
fi

N4BiasFieldCorrection -d 3 \
                    -v 1 \
                    -s 4 \
                    -b [180] \
                    -c [50x50x50x50,0.0] \
                    -i "$input_image" \
                    -o ["$output_image","$output_bias_field"]

echo "N4 bias field correction completed."