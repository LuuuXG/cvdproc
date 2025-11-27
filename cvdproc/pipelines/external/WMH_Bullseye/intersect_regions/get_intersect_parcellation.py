#!/usr/bin/python3

import os 
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import sys

lobar_map_filename = os.path.join(data_path, sys.argv[1])
concentric_map_filename = os.path.join(data_path, sys.argv[2])

lobar_map = nib.load(lobar_map_filename)
concentric_map = nib.load(concentric_map_filename)

lobar_map_data = lobar_map.get_fdata()
concentric_map_data = concentric_map.get_fdata()

out = np.zeros(lobar_map.shape, dtype=np.int32)

u1_set = np.unique(lobar_map_data.ravel())
u2_set = np.unique(concentric_map_data.ravel())

for u1 in u1_set:
    if u1 == 0: continue
    mask1 = lobar_map_data == u1
    for u2 in u2_set:
        if u2 == 0: continue
        mask2 = concentric_map_data == u2
        mask3 = np.logical_and(mask1, mask2)
        if not np.any(mask3): continue
        out[mask3] = int(str(int(round(u1))) + str(int(round(u2))))  

out_img = nib.Nifti1Image(out, affine = lobar_map.affine, header = lobar_map.header)

nib.save(out_img, os.path.join(sys.argv[3], sys.argv[4]))
