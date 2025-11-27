#%%
# Reconstruct with GQI in DIPY

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

from dipy.core.gradients import gradient_table
from dipy.core.ndindex import ndindex
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.odf import gfa

from dipy.direction.peaks import peaks_from_model, reshape_peaks_for_visualization

#%% 01 Load DSI data

# Use sample DSI data if desired
fraw, fbval, fbvec = get_fnames('taiwan_ntu_dsi')
brain_mask = None

# Preprocessed DWI, bval, bvec, mask paths
# fraw = r'E:\Neuroimage\TestDataSet\BIDS_TestDataSet\sub-Patient238\dwi\sub-Patient238_acq-b4000_dwi.nii.gz'
# fbval = r'E:\Neuroimage\TestDataSet\BIDS_TestDataSet\sub-Patient238\dwi\sub-Patient238_acq-b4000_dwi.bval'
# fbvec = r'E:\Neuroimage\TestDataSet\BIDS_TestDataSet\sub-Patient238\dwi\sub-Patient238_acq-b4000_dwi.bvec'
# brain_mask = r'E:\Neuroimage\TestDataSet\BIDS_TestDataSet\sub-Patient238\dwi\sub-Patient238_acq-b4000_dwi_b0_mask.nii.gz'

# Extract folder path for saving results
file_folder = os.path.dirname(fraw)

data, affine, voxel_size = load_nifti(fraw, return_voxsize=True)

# If no mask is specified, use an all-ones mask
if brain_mask is None:
    data_mask = np.ones(data.shape[:-1])
else:
    data_mask, _, _ = load_nifti(brain_mask, return_voxsize=True)

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
bvecs[1:] = (bvecs[1:] /
             np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None])
gtab = gradient_table(bvals, bvecs)
print('data.shape (%d, %d, %d, %d)' % data.shape)


#%% 02 Set reconstruction model

#dsi_model = DiffusionSpectrumModel(gtab)
gqi_model = GeneralizedQSamplingModel(gtab, sampling_length=3)

sphere = get_sphere('symmetric642')

# Reconstruct slice by slice along the z-axis
# peak: x*y*z*15 (peaks_from_model defaults to 5 peaks)
total_peaks = np.zeros((data.shape[0], data.shape[1], data.shape[2], 15))
# total_gfa = np.zeros((data.shape[0], data.shape[1], data.shape[2]))

for i in range(data.shape[2]):
    print('Reconstructing slice %d/%d' % (i + 1, data.shape[2]))
    dataslice = data[:, :, i]
    maskslice = data_mask[:, :, i]

    peaks = peaks_from_model(model=gqi_model,
                             data=dataslice,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             mask=maskslice,)

    new_peaks = reshape_peaks_for_visualization(peaks)
    total_peaks[:, :, i, :] = new_peaks

    # gfa = peaks.gfa
    # total_gfa[:, :, i] = gfa

# save results as 4D .nii.gz
total_peaks_nii = nib.Nifti1Image(total_peaks, affine)
nib.save(total_peaks_nii, os.path.join(file_folder, 'peaks.nii.gz'))

# total_gfa_nii = nib.Nifti1Image(total_gfa, affine)
# nib.save(total_gfa_nii, os.path.join(file_folder, 'gfa.nii.gz'))
