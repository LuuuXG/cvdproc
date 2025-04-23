import numpy as np
import os
import commit
from commit import trk2dictionary

os.chdir(r'E:\Neuroimage\TestDataSet\COMMIT_example\data')

os.system( 'dice_connectome_build fibers_assignment.txt connectome.csv -t demo01_fibers.tck -a GM_1x1x1.nii.gz -f')
os.system( 'dice_tractogram_sort demo01_fibers.tck GM_1x1x1.nii.gz demo01_fibers_connecting.tck -f')

trk2dictionary.run(
    filename_tractogram = 'demo01_fibers_connecting.tck',
    filename_mask       = 'WM.nii.gz',
    fiber_shift         = 0.5
)

# convert the bvals/bvecs pair to a single scheme file
import amico
amico.util.fsl2scheme( 'bvals.txt', 'bvecs.txt', 'DWI.scheme' )

# load the data
mit = commit.Evaluation( '.', '.' )
mit.load_data( 'DWI.nii.gz', 'DWI.scheme' )

# use a forward-model with 1 Stick for the streamlines and 2 Balls for all the rest
mit.set_model( 'StickZeppelinBall' )
d_par       = 1.7E-3             # Parallel diffusivity [mm^2/s]
d_perps_zep = []                 # Perpendicular diffusivity(s) [mm^2/s]
d_isos      = [ 1.7E-3, 3.0E-3 ] # Isotropic diffusivity(s) [mm^2/s]
mit.model.set( d_par, d_perps_zep, d_isos )

mit.generate_kernels( regenerate=True )
mit.load_kernels()

# create the sparse data structures to handle the matrix A
mit.load_dictionary( 'COMMIT' )
mit.set_threads()
mit.build_operator()

# perform the fit
mit.fit( tol_fun=1e-3, max_iter=1000 )
mit.save_results( path_suffix="_COMMIT1" )

C = np.loadtxt( 'connectome.csv', delimiter=',' )
C = np.triu( C ) # be sure to get only the upper-triangular part of the matrix
group_size = C[C>0].astype(np.int32)

tmp = np.insert( np.cumsum(group_size), 0, 0 )
group_idx = np.fromiter( [np.arange(tmp[i],tmp[i+1]) for i in range(len(tmp)-1)], dtype=np.object_ )

params_IC = {}
params_IC['group_idx'] = group_idx
params_IC['group_weights_cardinality'] = True
params_IC['group_weights_adaptive'] = True

perc_lambda = 0.00025 # change to suit your needs

# set the regularisation
mit.set_regularisation(
    regularisers   = ('group_lasso', None, None),
    is_nonnegative = (True, True, True),
    lambdas        = (perc_lambda, None, None),
    params         = (params_IC, None, None)
)

mit.fit( tol_fun=1e-3, max_iter=1000 )
mit.save_results( path_suffix="_COMMIT2" )