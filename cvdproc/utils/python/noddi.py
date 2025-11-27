import amico
amico.setup()

ae = amico.Evaluation()

# dwi = '/mnt/f/BIDS/demo_BIDS/derivatives/qsirecon-3dSHORE/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-extrapolated_dwi.nii.gz'
# dwi_bval = '/mnt/f/BIDS/demo_BIDS/derivatives/qsirecon-3dSHORE/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-extrapolated_dwi.bval'
# dwi_bvec = '/mnt/f/BIDS/demo_BIDS/derivatives/qsirecon-3dSHORE/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-extrapolated_dwi.bvec'

dwi = '/mnt/f/BIDS/demo_BIDS/derivatives/qsiprep/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi.nii.gz'
dwi_bval = '/mnt/f/BIDS/demo_BIDS/derivatives/qsiprep/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi.bval'
dwi_bvec = '/mnt/f/BIDS/demo_BIDS/derivatives/qsiprep/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi.bvec'
dwi_mask = '/mnt/f/BIDS/demo_BIDS/derivatives/qsiprep/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-brain_mask.nii.gz'

amico.util.fsl2scheme(dwi_bval, dwi_bvec)

ae.load_data(dwi, '/mnt/f/BIDS/demo_BIDS/derivatives/qsiprep/sub-TAOHC0261/ses-baseline/dwi/sub-TAOHC0261_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_desc-preproc_dwi.scheme', mask_filename=dwi_mask, b0_thr=0)

ae.set_model('NODDI')
ae.generate_kernels(regenerate=True)

ae.load_kernels()

ae.fit()

ae.save_results()