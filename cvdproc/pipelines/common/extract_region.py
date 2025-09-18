import os
import subprocess
import shutil
import nibabel as nib
import time
import numpy as np
import pandas as pd
from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str

class ExtractRegionInputSpec(BaseInterfaceInputSpec):
    in_nii = File(exists=True, mandatory=True, desc='Input NIfTI file path')
    roi_list = traits.List(traits.Int(), mandatory=True, desc='List of ROI values to extract')
    binarize = Bool(True, desc='Whether to binarize the extracted region')
    output_nii = Str(mandatory=True, desc='Output NIfTI file path')

class ExtractRegionOutputSpec(TraitedSpec):
    out_nii = File(desc='Output NIfTI file path with extracted region')

class ExtractRegion(BaseInterface):
    """
    Extract specified ROI regions from a NIfTI file and save to a new NIfTI file.

    Parameters:
        in_nii (str): Input NIfTI file path.
        roi_list (list of int): List of ROI values to extract (e.g., [1, 2, 3]).
        binarize (bool): Whether to binarize the extracted region. If True, all selected
                         region values are set to 1; otherwise, original region values are retained.
        output_nii (str): Output NIfTI file path.
    """
    input_spec = ExtractRegionInputSpec
    output_spec = ExtractRegionOutputSpec

    def _run_interface(self, runtime):
        in_nii = self.inputs.in_nii
        roi_list = self.inputs.roi_list
        binarize = self.inputs.binarize
        output_nii = self.inputs.output_nii

        # Load input NIfTI file
        img = nib.load(in_nii)
        data = img.get_fdata()

        # Create ROI mask for specified values
        roi_mask = np.isin(data, roi_list)

        # Initialize new data array with zeros
        new_data = np.zeros(data.shape, dtype=data.dtype)

        if binarize:
            new_data[roi_mask] = 1
        else:
            new_data[roi_mask] = data[roi_mask]

        # Create new NIfTI image and save to output path
        new_img = nib.Nifti1Image(new_data, affine=img.affine, header=img.header)
        nib.save(new_img, output_nii)

        self.out_nii = output_nii
        print(f"Extracted ROI saved to {output_nii}")

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_nii'] = getattr(self, 'out_nii', None)
        return outputs