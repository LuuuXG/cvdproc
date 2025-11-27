#!/usr/bin/python3

import os 
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import sys

# Load Bullseye Parcellation
bullseye_filename = os.path.join(data_path, sys.argv[1])
bullseye_filename_bis = os.path.join(data_path, sys.argv[2])

bullseye = nib.load(bullseye_filename)
bullseye_bis = nib.load(bullseye_filename_bis)

bullseye_data = bullseye.get_fdata()
bullseye_data_bis = bullseye_bis.get_fdata()

# Load WMH
wmh_filename = os.path.join(data_path, sys.argv[3])
wmh = nib.load(wmh_filename)
wmh_data = wmh.get_fdata()

# Output Matrix
out = np.zeros(wmh.shape, dtype = np.int32)
out_bis = np.zeros(wmh.shape, dtype = np.int32)

# Creation sets

u1_set = np.unique(wmh_data.ravel())
u2_set = np.unique(bullseye_data.ravel())
u2_set_bis = np.unique(bullseye_data_bis.ravel())

# Principal Intersect
for u1 in u1_set:
    if u1 == 0: continue
    mask1 = wmh_data == u1
    for u2 in u2_set:
        mask2 = bullseye_data == u2
        mask3 = np.logical_and(mask1, mask2)
        if not np.any(mask3): continue
        out[mask3] = int(str(int(round(u1))) + str(int(round(u2))))  

# Intersect Bis 
for u1 in u1_set:
    if u1 == 0: continue
    mask1 = wmh_data == u1
    for u2 in u2_set_bis:
        mask2 = bullseye_data_bis == u2
        mask3 = np.logical_and(mask1, mask2)
        if not np.any(mask3): continue
        out_bis[mask3] = int(str(int(round(u1))) + str(int(round(u2))))  

print(np.unique(out.ravel()))

# Save
out_img = nib.Nifti1Image(out, affine = bullseye.affine, header = bullseye.header)
out_img_bis = nib.Nifti1Image(out_bis, affine = bullseye.affine, header = bullseye.header)

nib.save(out_img, os.path.join(sys.argv[4], sys.argv[5]))
nib.save(out_img_bis, os.path.join(sys.argv[4], sys.argv[6]))
