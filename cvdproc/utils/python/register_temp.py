import os
import subprocess
import shutil

bids_root_dir = '/mnt/f/BIDS/SVD_BIDS'

subject_id = 'SVD0033'

session_id_dti = '03'
session_id_lesion = '01'

dwi_path = os.path.join(bids_root_dir, f'sub-{subject_id}', f'ses-{session_id_lesion}', 'dwi', f'sub-{subject_id}_ses-{session_id_lesion}_acq-DWIb1000_dir-PA_dwi.nii.gz')
lesion_mask_path = os.path.join(bids_root_dir, 'derivatives', 'lesion_mask', f'sub-{subject_id}', f'ses-{session_id_lesion}', 'dwi_infarction.nii.gz')

# put output in
output_dir = os.path.join(bids_root_dir, 'derivatives', 'lesion_mask', f'sub-{subject_id}', f'ses-{session_id_dti}')
os.makedirs(output_dir, exist_ok=True)

# get dwi b0
subprocess.run(['fslroi', dwi_path, os.path.join(output_dir, 'dwi_b0.nii.gz'), '0', '1'])
#shutil.copy(dwi_path, os.path.join(output_dir, 'dwi_b0.nii.gz'))
# skull strip using mri_synthstrip
subprocess.run(['mri_synthstrip', '-i', os.path.join(output_dir, 'dwi_b0.nii.gz'), '-o', os.path.join(output_dir, 'dwi_b0_brain.nii.gz')])

# get dti b0
dti_path = os.path.join(bids_root_dir, 'derivatives', 'fdt', f'sub-{subject_id}', f'ses-{session_id_dti}', 'eddy_corrected_data.nii.gz')
subprocess.run(['fslroi', dti_path, os.path.join(output_dir, 'dti_b0.nii.gz'), '0', '1'])
# skull strip using mri_synthstrip
subprocess.run(['mri_synthstrip', '-i', os.path.join(output_dir, 'dti_b0.nii.gz'), '-o', os.path.join(output_dir, 'dti_b0_brain.nii.gz')])

# get the transformation matrix
subprocess.run(['flirt', '-in', os.path.join(output_dir, 'dwi_b0_brain.nii.gz'), '-ref', os.path.join(output_dir, 'dti_b0_brain.nii.gz'), '-omat', os.path.join(output_dir, 'dwi2dti.mat')])
# apply the transformation matrix to the lesion mask
subprocess.run(['flirt', '-in', lesion_mask_path, '-ref', os.path.join(output_dir, 'dti_b0_brain.nii.gz'), '-out', os.path.join(output_dir, 'dti_infarction.nii.gz'), '-applyxfm', '-init', os.path.join(output_dir, 'dwi2dti.mat'), '-interp', 'nearestneighbour'])

# delete intermediate files
os.remove(os.path.join(output_dir, 'dwi_b0.nii.gz'))
os.remove(os.path.join(output_dir, 'dwi_b0_brain.nii.gz'))
os.remove(os.path.join(output_dir, 'dti_b0.nii.gz'))
os.remove(os.path.join(output_dir, 'dti_b0_brain.nii.gz'))
os.remove(os.path.join(output_dir, 'dwi2dti.mat'))