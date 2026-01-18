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
# subject_id=$2
# session_id=$3
# fs_to_dwi_xfm=$4
# output_dir=$5
# dwi_ref=$6
# space_entity=$7 # e.g., ACPC

class PrepareVPROIInputSpec(CommandLineInputSpec):
    fs_output_dir = Directory(exists=True, desc="FreeSurfer output directory", mandatory=True, position=0, argstr='%s')
    subject_id = Str(desc="Subject ID", mandatory=True, position=1, argstr='%s')
    session_id = Str(desc="Session ID", mandatory=True, position=2, argstr='%s')
    fs_to_dwi_xfm = File(exists=True, desc="FreeSurfer to DWI space transform", mandatory=True, position=3, argstr='%s')
    output_dir = Directory(desc="Output directory for prepared ROIs", mandatory=True, position=4, argstr='%s')
    dwi_ref = File(exists=True, desc="DWI reference image", mandatory=True, position=5, argstr='%s')
    space_entity = Str(desc="Space entity for output files", mandatory=True, position=6, argstr='%s')

class PrepareVPROIOutputSpec(TraitedSpec):
    lh_lgn_roi = File(desc="Left LGN ROI in DWI space")
    lh_lgn_dil1_roi = File(desc="Left LGN ROI dilated by 1 voxel in DWI space")
    rh_lgn_roi = File(desc="Right LGN ROI in DWI space")
    rh_lgn_dil1_roi = File(desc="Right LGN ROI dilated by 1 voxel in DWI space")
    lh_v1_roi = File(desc="Left V1 ROI in DWI space")
    lh_v1_ext2_roi = File(desc="Left V1 ROI extended by 2mm in DWI space")
    rh_v1_roi = File(desc="Right V1 ROI in DWI space")
    rh_v1_ext2_roi = File(desc="Right V1 ROI extended by 2mm in DWI space")
    optic_chiasm_roi = File(desc="Optic chiasm ROI in DWI space")
    optic_chiasm_dil1_roi = File(desc="Optic chiasm ROI dilated by 1 voxel in DWI space")

class PrepareVPROI(CommandLine):
    input_spec = PrepareVPROIInputSpec
    output_spec = PrepareVPROIOutputSpec
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'visual_pathway_project', 'prepare_roi2.sh')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        output_dir = os.path.abspath(self.inputs.output_dir)

        outputs['lh_lgn_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_hemi-L_space-{self.inputs.space_entity}_label-LGN_mask.nii.gz")
        outputs['lh_lgn_dil1_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_hemi-L_space-{self.inputs.space_entity}_label-LGN_desc-dilate1_mask.nii.gz")
        outputs['rh_lgn_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_hemi-R_space-{self.inputs.space_entity}_label-LGN_mask.nii.gz")
        outputs['rh_lgn_dil1_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_hemi-R_space-{self.inputs.space_entity}_label-LGN_desc-dilate1_mask.nii.gz")
        outputs['lh_v1_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_hemi-L_space-{self.inputs.space_entity}_label-V1exvivo_mask.nii.gz")
        outputs['lh_v1_ext2_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_hemi-L_space-{self.inputs.space_entity}_label-V1exvivo_desc-extend2mm_mask.nii.gz")
        outputs['rh_v1_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_hemi-R_space-{self.inputs.space_entity}_label-V1exvivo_mask.nii.gz")
        outputs['rh_v1_ext2_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_hemi-R_space-{self.inputs.space_entity}_label-V1exvivo_desc-extend2mm_mask.nii.gz")
        outputs['optic_chiasm_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_space-{self.inputs.space_entity}_label-opticchiasm_mask.nii.gz")
        outputs['optic_chiasm_dil1_roi'] = os.path.join(output_dir, f"sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_space-{self.inputs.space_entity}_label-opticchiasm_desc-dilate1_mask.nii.gz")

        return outputs