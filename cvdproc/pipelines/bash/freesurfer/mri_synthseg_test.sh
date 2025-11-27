export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mri_synthseg \
  --i  /mnt/f/BIDS/demo_BIDS/sub-AFib0241/ses-baseline/anat/sub-AFib0241_ses-baseline_acq-highres_T1w.nii.gz \
  --o  /mnt/f/BIDS/demo_BIDS/derivatives/xfm/sub-AFib0241/ses-baseline/synthseg_test.nii.gz \
  --vol /mnt/f/BIDS/demo_BIDS/derivatives/xfm/sub-AFib0241/ses-baseline/synthseg_test.csv \
  --cpu \
  --threads 1
