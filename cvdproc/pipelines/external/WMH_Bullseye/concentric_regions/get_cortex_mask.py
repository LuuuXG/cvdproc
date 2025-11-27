#!/usr/bin/python3

import os 
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import sys

source = sys.argv[1]
target = sys.argv[2]

aseg_filename = os.path.join(data_path, source)
A = nib.load(aseg_filename)
A_data = A.get_fdata()

print(A)

A_data[np.where(A_data == 42)] = 1
A_data[np.where(A_data == 3)] = 1
A_data[np.where(A_data != 1)] = 0

B = nib.Nifti1Image(A_data, affine=A.affine, header = A.header)
nib.save(B, os.path.join(target, 'cortex_mask.nii.gz'))
