import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.direction import peaks_from_model

fraw, fbval, fbvec = get_fnames('taiwan_ntu_dsi')

data, affine, voxel_size = load_nifti(fraw, return_voxsize=True)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
bvecs[1:] = (bvecs[1:] /
                 np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None])
gtab = gradient_table(bvals, bvecs)
print('data.shape (%d, %d, %d, %d)' % data.shape)

gqmodel = GeneralizedQSamplingModel(gtab, sampling_length=3)

dataslice = data[:, :, data.shape[2] // 2]

mask = dataslice[..., 0] > 50

gqfit = gqmodel.fit(dataslice, mask=mask)

sphere = get_sphere('repulsion724')

ODF = gqfit.odf(sphere)

print('ODF.shape (%d, %d, %d)' % ODF.shape)

gqpeaks = peaks_from_model(model=gqmodel,
                           data=dataslice,
                           sphere=sphere,
                           relative_peak_threshold=.5,
                           min_separation_angle=25,
                           mask=mask,
                           return_odf=False,
                           normalize_peaks=True)

gqpeak_values = gqpeaks.peak_values