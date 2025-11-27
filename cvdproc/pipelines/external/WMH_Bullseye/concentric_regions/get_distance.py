#!/usr/bin/python3
import os 
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import sys

source_ventricles = sys.argv[1]
source_cortex = sys.argv[2]
target = sys.argv[3]

ventricles_filename = os.path.join(data_path, source_ventricles)
cortex_filename = os.path.join(data_path, source_cortex)

ventricles = nib.load(ventricles_filename)
print(ventricles)
cortex = nib.load(cortex_filename)
print(cortex)

ventricles_mask = ventricles.get_fdata()
cortex_mask = cortex.get_fdata()

dist_orig = distance_transform_edt(np.logical_not(ventricles_mask))
dist_dest = distance_transform_edt(np.logical_not(cortex_mask))
ndist = dist_orig / (dist_orig + dist_dest)

dist = nib.Nifti1Image(ndist, affine = cortex.affine)
nib.save(dist, os.path.join(target, 'dist_map.nii.gz'))
