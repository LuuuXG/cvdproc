import commit
import os

#commit.setup()

# 更换工作目录
os.chdir(r'E:\Neuroimage\TestDataSet\COMMIT_example\data')

from commit import trk2dictionary
trk2dictionary.run(
    filename_tractogram = 'demo01_fibers.tck',
    filename_peaks      = 'peaks.nii.gz',
    filename_mask       = 'WM.nii.gz',
    fiber_shift         = 0.5,
    peaks_use_affine    = True
)

import amico
amico.util.fsl2scheme( 'bvals.txt', 'bvecs.txt', 'DWI.scheme' )

mit = commit.Evaluation()
mit.set_verbose(4)
mit.load_data( 'DWI.nii.gz', 'DWI.scheme' )

mit.set_model( 'StickZeppelinBall' )
d_par       = 1.7E-3             # Parallel diffusivity [mm^2/s]
d_perps_zep = [ 0.51E-3 ]        # Perpendicular diffusivity(s) [mm^2/s]
d_isos      = [ 1.7E-3, 3.0E-3 ] # Isotropic diffusivity(s) [mm^2/s]
mit.model.set( d_par, d_perps_zep, d_isos )
mit.generate_kernels( regenerate=True )
mit.load_kernels()

mit.load_dictionary( 'COMMIT' )

mit.set_threads()
mit.build_operator()

mit.fit( tol_fun=1e-3, max_iter=1000 )

mit.save_results()