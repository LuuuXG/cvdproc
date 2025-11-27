#!/usr/bin/python3

import os 
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import sys

lobar_aseg_filename = os.path.join(data_path, sys.argv[1])

A = nib.load(lobar_aseg_filename)
A_data = A.get_fdata()

A_data[(A_data == 7) | (A_data == 8) | (A_data == 47) | (A_data == 46) | (A_data == 16) | (A_data == 15) | (A_data == 60) | (A_data == 28) | (A_data == 14) | (A_data == 43) | (A_data == 4) | (A_data == 44) | (A_data == 5) | (A_data == 24) | (A_data == 31) | (A_data == 63) | (A_data == 85) | (A_data == 72) | (A_data == 80)] = 0
A_data[(A_data == 54) | (A_data == 53)] = 4004
A_data[(A_data == 17) | (A_data == 18)] = 3004
A_data[(A_data == 49) | (A_data == 10) | (A_data == 11) | (A_data == 12) | (A_data == 51) | (A_data == 13) | (A_data == 50) | (A_data == 58) | (A_data == 26) | (A_data == 30) | (A_data == 62)] = 52
A_data[(A_data == 251) | (A_data == 252) | (A_data == 253) | (A_data == 254) | (A_data == 255)] = 251

A_data_bis = np.copy(A_data)

A_data_bis[(A_data_bis == 2002)] = 4002 
A_data_bis[(A_data_bis == 1002)] = 3002
A_data_bis[(A_data_bis == 2001)] = 4001
A_data_bis[(A_data_bis == 1001)] = 3001
A_data_bis[(A_data_bis == 2004)] = 4004
A_data_bis[(A_data_bis == 1004)] = 3004
A_data_bis[(A_data_bis == 2003)] = 4003
A_data_bis[(A_data_bis == 1003)] = 3003

A_data[(A_data == 2002) | (A_data == 1002) | (A_data == 2001) | (A_data == 1001) | (A_data == 2004) | (A_data == 1004) | (A_data == 2003) | (A_data == 1003)] = 0

B = nib.Nifti1Image(A_data, affine=A.affine, header = A.header)
B_bis = nib.Nifti1Image(A_data_bis, affine=A.affine, header = A.header)

nib.save(B, os.path.join(sys.argv[2], 'lobar_aseg_masked.nii.gz'))
nib.save(B_bis, os.path.join(sys.argv[2], 'lobar_aseg_masked_bis.nii.gz'))
