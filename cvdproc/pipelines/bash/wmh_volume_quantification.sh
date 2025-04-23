#!/bin/bash

# This function processes a single subject
process_subject() {
    TWMH_file=$1
    PWMH_file=$2
    DWMH_file=$3
    OUTPUT_DIR=$4

    # Set MNI template path
    echo "Processing subject with TWMH: $TWMH_file, PWMH: $PWMH_file, DWMH: $DWMH_file, output will be saved to: $OUTPUT_DIR"

    resultsFile="$OUTPUT_DIR/WMH_PWMH&DWMH_volume_thr5voxels.txt"

    # 0在此没有意义，因为已经二值化
    echo -e "TWMH" > "$resultsFile" # because we want to overwrite the file
    TWMH=$(bianca_cluster_stats $TWMH_file 0 5)
    echo "$TWMH" >> "$resultsFile"

    echo -e "\nPWMH" >> "$resultsFile"
    PWMH=$(bianca_cluster_stats $PWMH_file 0 5)
    echo "$PWMH" >> "$resultsFile"

    echo -e "\nDWMH" >> "$resultsFile"
    DWMH=$(bianca_cluster_stats $DWMH_file 0 5)
    echo "$DWMH" >> "$resultsFile"

    echo "Quantification completed!"
}

# Call the function with the provided paths
process_subject $1 $2 $3 $4
