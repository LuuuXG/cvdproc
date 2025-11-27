import nibabel as nib
import os
import numpy as np
from dipy.denoise.gibbs import gibbs_removal
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory, Str

class DipyDegibbsInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc="Path to the DWI file to be degibbsed")
    output_dwi = Str(mandatory=True, desc="Path to save the degibbsed DWI file")

class DipyDegibbsOutputSpec(TraitedSpec):
    output_dwi_img = Str(desc="Path to the degibbsed DWI file")

class DipyDegibbs(BaseInterface):
    input_spec = DipyDegibbsInputSpec
    output_spec = DipyDegibbsOutputSpec

    def _run_interface(self, runtime):
        # If self.inputs.output_dwi already exists, skip processing
        if os.path.exists(self.inputs.output_dwi):
            print(f"[DipyDegibbs] Output file {self.inputs.output_dwi} already exists. Skipping degibbsing.")
            return runtime
        # Load the DWI data
        dwi_img = nib.load(self.inputs.dwi_file)
        dwi_data = dwi_img.get_fdata()

        # Apply Gibbs ringing correction
        corrected_data = gibbs_removal(dwi_data, num_processes=8)

        # Save the corrected data
        corrected_img = nib.Nifti1Image(corrected_data, dwi_img.affine, dwi_img.header)
        nib.save(corrected_img, self.inputs.output_dwi)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_dwi_img'] = self.inputs.output_dwi
        return outputs