import os
import subprocess
import nibabel as nib

subject_id = 'sub-SVD0035'
session_id = 'ses-02'

bids_root_dir = '/mnt/f/BIDS/SVD_BIDS'
output_root_dir = '/mnt/f/BIDS/SVD_BIDS/derivatives/fdt_paths_mni'
mni_ref = '/mnt/e/codes/cvdproc/cvdproc/data/standard/MNI152/MNI152_T1_1mm.nii.gz'

os.makedirs(output_root_dir, exist_ok=True)

fdt_path_native = os.path.join(bids_root_dir, 'derivatives', 'dwi_pipeline', subject_id, session_id, 'probtrackx_output', 'fdt_paths.nii.gz')
waytotal = os.path.join(bids_root_dir, 'derivatives', 'dwi_pipeline', subject_id, session_id, 'probtrackx_output', 'waytotal')

t1w = os.path.join(bids_root_dir, subject_id, session_id, 'anat', f'{subject_id}_{session_id}_acq-highres_T1w.nii.gz')
t1w_roi = os.path.join(bids_root_dir, 'derivatives', 'fsl_anat', subject_id, session_id, 'fsl.anat', 'T1_biascorr.nii.gz')
t1_to_mni_warp = os.path.join(bids_root_dir, 'derivatives', 'fsl_anat', subject_id, session_id, 'fsl.anat', 'T1_to_MNI_nonlin_field.nii.gz')
orig_to_roi = os.path.join(bids_root_dir, 'derivatives', 'fsl_anat', subject_id, session_id, 'fsl.anat', 'T1_orig2roi.mat')

# reslice fdt_path to have same res with t1w (flirt -applyxfm use qform)
subprocess.run([
    'flirt',
    '-in', fdt_path_native,
    '-ref', t1w ,
    '-out', os.path.join(output_root_dir, f'{subject_id}_{session_id}_fdt_paths_resliced.nii.gz'),
    '-applyxfm', '-usesqform'
], check=True)

# # resliced to t1w_roi
# subprocess.run([
#     'flirt',
#     '-in', os.path.join(output_root_dir, f'{subject_id}_{session_id}_fdt_paths_resliced.nii.gz'),
#     '-ref', t1w_roi,
#     '-out', os.path.join(output_root_dir, f'{subject_id}_{session_id}_fdt_paths_resliced_roi.nii.gz'),
#     '-applyxfm', '-init', orig_to_roi
# ], check=True)
#
# # use applywarp to transform fdt_paths from native space to MNI space
# subprocess.run([
#     'applywarp',
#     '-i', os.path.join(output_root_dir, f'{subject_id}_{session_id}_fdt_paths_resliced_roi.nii.gz'),
#     '-o', os.path.join(output_root_dir, f'{subject_id}_{session_id}_fdt_paths_mni.nii.gz'),
#     '-r', mni_ref,
#     '-w', t1_to_mni_warp
# ], check=True)
#

# get new t1w_to_mni_warp use synthmorph
subprocess.run([
    'mri_synthmorph',
    '-t', os.path.join(output_root_dir, f'{subject_id}_{session_id}_t1w_to_mni_warp.nii.gz'),
    t1w, mni_ref, '-g'
], check=True)

# transform use mri_convert
subprocess.run([
    'at', os.path.join(output_root_dir, f'{subject_id}_{session_id}_t1w_to_mni_warp.nii.gz'),
    os.path.join(output_root_dir, f'{subject_id}_{session_id}_fdt_paths_resliced.nii.gz'),
    os.path.join(output_root_dir, f'{subject_id}_{session_id}_fdt_paths_mni.nii.gz'),
], check=True)

# scale the output by waytotal (there will be only one number in file 'waytotal')
fdt_path_mni = os.path.join(output_root_dir, f'{subject_id}_{session_id}_fdt_paths_mni.nii.gz')

img = nib.load(fdt_path_mni)
data = img.get_fdata()

waytotal_number = float(open(waytotal, 'r').read().strip())

data /= waytotal_number
scaled_img = nib.Nifti1Image(data, img.affine, img.header)
nib.save(scaled_img, fdt_path_mni)