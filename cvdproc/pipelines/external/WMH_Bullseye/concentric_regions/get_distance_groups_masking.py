#!/usr/bin/python3

import os 
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
import sys

dist_group = sys.argv[1]
aseg = sys.argv[2]
target = sys.argv[3]

dist_filename = os.path.join(data_path, dist_group)
dist = nib.load(dist_filename)
dist_data = dist.get_fdata()

aseg_filename = os.path.join(data_path, aseg)
aseg = nib.load(aseg_filename)
aseg_data = aseg.get_fdata()

dist_data[np.where(aseg_data == 0)] = 0
dist_data_bis = np.copy(dist_data)

dist_data_bis[np.where(aseg_data == 42)] = 4
dist_data_bis[np.where(aseg_data == 3)] = 4

new_dist_img = nib.Nifti1Image(dist_data, affine=dist.affine, header = aseg.header)
new_dist_img_bis = nib.Nifti1Image(dist_data_bis, affine=dist.affine, header = aseg.header)

nib.save(new_dist_img, os.path.join(target, 'dist_map_grouped_masked.nii.gz'))
nib.save(new_dist_img_bis, os.path.join(target, 'dist_map_grouped_masked_bis.nii.gz'))
