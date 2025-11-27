import os
import subprocess
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str
import amico

from cvdproc.config.paths import get_package_path

# fs_output_dir=$1 # have /mri, /surf...
# fs_to_orig_mat=$2
# orig_to_dwi_mat=$3
# t1w_ref=$4
# dwi_ref=$5
# output_dir=$6
# dwi_final_ref=$7

class PrepareVPROIInputSpec(CommandLineInputSpec):
    fs_output_dir = Directory(exists=True, desc="FreeSurfer output directory", mandatory=True, position=0, argstr='%s')
    fs_to_orig_mat = File(exists=True, desc="FreeSurfer to original T1w affine matrix", mandatory=True, position=1, argstr='%s')
    t1w_ref = File(exists=True, desc="Reference T1w image", mandatory=True, position=2, argstr='%s')
    dwi_ref = File(exists=True, desc="Reference DWI image", mandatory=True, position=3, argstr='%s')
    output_dir = Directory(desc="Output directory", mandatory=True, position=4, argstr='%s')
    dwi_final_ref = File(exists=True, desc="Final DWI reference image", mandatory=True, position=5, argstr='%s')

class PrepareVPROIOutputSpec(TraitedSpec):
    lh_lgn_roi = File(desc="Left LGN ROI in DWI space")
    lh_lgn_dil1_roi = File(desc="Left LGN ROI dilated by 1 voxel in DWI space")
    rh_lgn_roi = File(desc="Right LGN ROI in DWI space")
    rh_lgn_dil1_roi = File(desc="Right LGN ROI dilated by 1 voxel in DWI space")
    lh_v1_roi = File(desc="Left V1 ROI in DWI space")
    rh_v1_roi = File(desc="Right V1 ROI in DWI space")
    optic_chiasm_roi = File(desc="Optic chiasm ROI in DWI space")
    optic_chiasm_dil1_roi = File(desc="Optic chiasm ROI dilated by 1 voxel in DWI space")

class PrepareVPROI(CommandLine):
    input_spec = PrepareVPROIInputSpec
    output_spec = PrepareVPROIOutputSpec
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'visual_pathway_project', 'prepare_roi.sh')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        output_dir = os.path.abspath(self.inputs.output_dir)

        outputs['lh_lgn_roi'] = os.path.join(output_dir, 'lh_LGN_in_DWI.nii.gz')
        outputs['lh_lgn_dil1_roi'] = os.path.join(output_dir, 'lh_LGN_in_DWI_dil1.nii.gz')
        outputs['rh_lgn_roi'] = os.path.join(output_dir, 'rh_LGN_in_DWI.nii.gz')
        outputs['rh_lgn_dil1_roi'] = os.path.join(output_dir, 'rh_LGN_in_DWI_dil1.nii.gz')
        outputs['lh_v1_roi'] = os.path.join(output_dir, 'lh_V1_exvivo_in_DWI.nii.gz')
        outputs['rh_v1_roi'] = os.path.join(output_dir, 'rh_V1_exvivo_in_DWI.nii.gz')
        outputs['optic_chiasm_roi'] = os.path.join(output_dir, 'optic_chiasm_in_DWI.nii.gz')
        outputs['optic_chiasm_dil1_roi'] = os.path.join(output_dir, 'optic_chiasm_in_DWI_dil1.nii.gz')

        return outputs
