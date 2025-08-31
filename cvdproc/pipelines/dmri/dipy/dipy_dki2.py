import os
import numpy as np
import nibabel as nib
from dipy.reconst.dki import DiffusionKurtosisModel, Wcons
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.denoise.localpca import mppca
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
from dipy.segment.mask import median_otsu
from dipy.viz.plotting import compare_maps
import pickle
import json

fraw, fbval, fbvec, t1_fname = get_fnames(name="cfin_multib")

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs=bvecs)

bval_sel = np.zeros_like(gtab.bvals)
bval_sel[bvals == 0] = 1
bval_sel[bvals == 600] = 1
bval_sel[bvals == 1000] = 1
bval_sel[bvals == 2000] = 1

data = data[..., bval_sel == 1]
gtab = gradient_table(bvals[bval_sel == 1], bvecs=bvecs[bval_sel == 1])

datamask, mask = median_otsu(
    data, vol_idx=[0, 1], median_radius=4, numpass=2, autocrop=False, dilate=1
)

data = mppca(data, patch_radius=[3, 3, 3])

dkimodel = dki.DiffusionKurtosisModel(gtab)
dkifit = dkimodel.fit(data[:, :, 9:10], mask=mask[:, :, 9:10])

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data[:, :, 9:10], mask=mask[:, :, 9:10])

fits = [tenfit, dkifit]
maps = ["fa", "md", "rd", "ad"]
fit_labels = ["DTI", "DKI"]
map_kwargs = [{"vmax": 0.7}, {"vmax": 2e-3}, {"vmax": 2e-3}, {"vmax": 2e-3}]
compare_maps(
    fits,
    maps,
    fit_labels=fit_labels,
    map_kwargs=map_kwargs,
    filename="Diffusion_tensor_measures_from_DTI_and_DKI.png",
)