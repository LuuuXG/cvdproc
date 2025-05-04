import os
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import fetch_hbn
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory

class FreeWaterTensorInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc="Path to the DWI file")
    bval_file = File(exists=True, mandatory=True, desc="Path to the bval file")
    bvec_file = File(exists=True, mandatory=True, desc="Path to the bvec file")
    mask_file = File(exists=True, mandatory=True, desc="Path to the brain mask file")
    output_dir = Directory(exists=True, mandatory=True, desc="Output directory for the free water map")

class FreeWaterTensorOutputSpec(TraitedSpec):
    freewater_file = File(desc="Path to the output free water map")
    fwfa_file = File(desc="Path to the output free water fraction map")
    fwmd_file = File(desc="Path to the output free water mean diffusivity map")

class FreeWaterTensor(BaseInterface):
    input_spec = FreeWaterTensorInputSpec
    output_spec = FreeWaterTensorOutputSpec

    def _run_interface(self, runtime):
        # Load the DWI data
        img = nib.load(self.inputs.dwi_file)
        data = np.asarray(img.dataobj)

        # Load the gradient table
        gtab = gradient_table(self.inputs.bval_file, self.inputs.bvec_file)

        # Load the mask
        mask_img = nib.load(self.inputs.mask_file)
        mask = mask_img.get_fdata()

        # Fit the Free Water Tensor model
        fwdtimodel = fwdti.FreeWaterTensorModel(gtab)
        fwdtifit = fwdtimodel.fit(data, mask=mask)

        # Extract the free water map
        freewater = fwdtifit.f
        fwfa = fwdtifit.fa
        fwmd = fwdtifit.md

        # Save the free water map
        freewater_img = nib.Nifti1Image(freewater, img.affine, img.header)
        freewater_path = os.path.join(self.inputs.output_dir, "freewater.nii.gz")
        freewater_img.to_filename(freewater_path)

        # Save the free water fraction map
        fwfa_img = nib.Nifti1Image(fwfa, img.affine, img.header)
        fwfa_path = os.path.join(self.inputs.output_dir, "freewater_fa.nii.gz")
        fwfa_img.to_filename(fwfa_path)

        # Save the free water mean diffusivity map
        fwmd_img = nib.Nifti1Image(fwmd, img.affine, img.header)
        fwmd_path = os.path.join(self.inputs.output_dir, "freewater_md.nii.gz")
        fwmd_img.to_filename(fwmd_path)

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["freewater_file"] = os.path.abspath(os.path.join(self.inputs.output_dir, "freewater.nii.gz"))
        outputs["fwfa_file"] = os.path.abspath(os.path.join(self.inputs.output_dir, "freewater_fa.nii.gz"))
        outputs["fwmd_file"] = os.path.abspath(os.path.join(self.inputs.output_dir, "freewater_md.nii.gz"))
        return outputs