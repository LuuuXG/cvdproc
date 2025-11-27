#!/usr/bin/python3

import os 
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
import sys

source = sys.argv[1]
target = sys.argv[2]

ndist_filename = os.path.join(data_path, source)
ndist = nib.load(ndist_filename)
ndist_data = ndist.get_fdata()

print(ndist)

out = np.zeros(ndist.shape)
limits = np.linspace(0., 1., 5)

for i in np.arange(4)+1:
    mask = np.logical_and(ndist_data >= limits[i-1], ndist_data < limits[i])
    out[mask] = i

out[np.isclose(ndist_data, 0.)] = 0 

out_img = nib.Nifti1Image(out, affine=ndist.affine)
nib.save(out_img, os.path.join(target, 'dist_map_grouped.nii.gz'))
