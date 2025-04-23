from neuromaps.datasets import fetch_annotation
from neuromaps import transforms
neurosynth = fetch_annotation(source='neurosynth')
fslr = transforms.mni152_to_fslr(neurosynth, '32k')

fslr_lh, fslr_rh = fslr
print(fslr_lh.agg_data().shape)

import nibabel as nib

# Save the left hemisphere data as a GIFTI file
nib.save(fslr_lh, r'E:\Codes\Basic_Imaging_Process\data\left_hemi.func.gii')

# Save the right hemisphere data as a GIFTI file
nib.save(fslr_rh, r'E:\Codes\Basic_Imaging_Process\data\right_hemi.func.gii')