import os
import nibabel as nib
from nipype.interfaces.base import (
    TraitedSpec, CommandLineInputSpec, CommandLine,
    File, traits, Str, Directory
)

class MarkVCIDFreeWaterInputSpec(CommandLineInputSpec):
    in_dwi = File(exists=True, mandatory=True, desc='Input DWI file', argstr='%s', position=0)
    in_dwi_mask = File(exists=True, mandatory=True, desc='Input DWI mask file', argstr='%s', position=1)
    in_dwi_bval = File(exists=True, mandatory=True, desc='Input bval file', argstr='%s', position=2)
    in_dwi_bvec = File(exists=True, mandatory=True, desc='Input bvec file', argstr='%s', position=3)
    output_dir = Str(mandatory=True, desc='Output directory', argstr='%s', position=4)
    script_dir = Str(mandatory=True, desc='Directory containing the script', argstr='%s', position=5)

class MarkVCIDFreeWaterOutputSpec(TraitedSpec):
    out_fw = File(desc='Output freewater image')
    out_fw_fa = File(desc='Output freewater-corrected FA image')

class MarkVCIDFreeWater(CommandLine):
    """
    A nipype interface to run the MarkVCID freewater processing script.
    """
    input_spec = MarkVCIDFreeWaterInputSpec
    output_spec = MarkVCIDFreeWaterOutputSpec
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'MarkVCID2', 'scripts_FW_CONSORTIUM', 'MAIN_script_FW_custom.sh'))

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_fw'] = os.path.join(self.inputs.output_dir, 'wls_dti_FW.nii.gz')
        outputs['out_fw_fa'] = os.path.join(self.inputs.output_dir, 'wls_dti_FA.nii.gz')
        return outputs