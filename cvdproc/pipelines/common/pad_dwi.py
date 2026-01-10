# # a nipype interface to pad dwi

import os
import nibabel as nib
from nipype.interfaces.base import (
    TraitedSpec, CommandLineInputSpec, CommandLine,
    File, traits, Str
)

class PadDWIInputSpec(CommandLineInputSpec):
    in_dwi = File(exists=True, mandatory=True, desc='Input DWI file', argstr='%s', position=0)
    in_bvec = File(exists=True, mandatory=True, desc='Input bvec file', argstr='%s', position=1)
    in_bval = File(exists=True, mandatory=True, desc='Input bval file', argstr='%s', position=2)
    in_json = File(exists=True, mandatory=True, desc='Input JSON file', argstr='%s', position=3)
    out_file = File(mandatory=True, desc='Output padded DWI file', argstr='%s', position=4)

class PadDWIOutputSpec(TraitedSpec):
    out_file = File(desc='Output padded DWI file')
    out_bvec = File(desc='Output padded bvec file')
    out_bval = File(desc='Output padded bval file')
    out_json = File(desc='Output padded JSON file')

class PadDWI(CommandLine):
    """
    A nipype interface to pad DWI images.
    """
    input_spec = PadDWIInputSpec
    output_spec = PadDWIOutputSpec
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bash', 'fdt', 'pad_dwi.sh'))

    def _list_outputs(self):
        outputs = self.output_spec().get()
        # If the z dimension of in_dwi is even, the outputs match the inputs.
        # If the z dimension of in_dwi is odd, the outputs use the padded files (swap .nii.gz to .bvec/.bval/.json).
        dwi_img = nib.load(self.inputs.in_dwi)
        Nz = dwi_img.shape[2]
        if Nz % 2 == 0:
            outputs['out_file'] = self.inputs.in_dwi
            outputs['out_bvec'] = self.inputs.in_bvec
            outputs['out_bval'] = self.inputs.in_bval
            outputs['out_json'] = self.inputs.in_json
        else:
            outputs['out_file'] = self.inputs.out_file
            outputs['out_bvec'] = self.inputs.out_file.replace('.nii.gz', '.bvec')
            outputs['out_bval'] = self.inputs.out_file.replace('.nii.gz', '.bval')
            outputs['out_json'] = self.inputs.out_file.replace('.nii.gz', '.json')

        return outputs
