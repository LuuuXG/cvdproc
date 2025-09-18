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
import csv

class CalculateROIVolumeInputSpec(BaseInterfaceInputSpec):
    in_nii = File(exists=True, mandatory=True, desc='Input NIfTI file path (binary mask)')
    roi_nii = File(exists=True, mandatory=True, desc='ROI NIfTI file path (integer labels)')
    output_csv = Str(desc='Output CSV file path. If not provided, will be saved in the same directory as input with prefix "roi_volume_"')
class CalculateROIVolumeOutputSpec(TraitedSpec):
    output_csv = File(desc='Output CSV file path with ROI volume calculations')
class CalculateROIVolume(BaseInterface):
    input_spec = CalculateROIVolumeInputSpec
    output_spec = CalculateROIVolumeOutputSpec

    def _run_interface(self, runtime):
        in_nii = self.inputs.in_nii
        roi_nii = self.inputs.roi_nii
        output_csv = self.inputs.output_csv

        # Load input NIfTI files
        img = nib.load(in_nii)
        roi = nib.load(roi_nii)

        if img.shape != roi.shape:
            raise ValueError("Input image and ROI mask must have the same dimensions.")

        img_data = img.get_fdata()
        roi_data = roi.get_fdata().astype(int)
        unique_labels = np.unique(roi_data)

        voxel_size = np.abs(np.linalg.det(img.affine[:3, :3]))

        results = []
        for label in unique_labels:
            if label == 0:
                continue  # skip background if 0

            mask = roi_data == label
            voxel_count = np.sum(img_data[mask] > 0)  # only count where input_image == 1
            volume = voxel_count * voxel_size

            results.append((label, voxel_count, voxel_size, volume))

        if not output_csv:
            base_name, ext = os.path.splitext(os.path.basename(in_nii))
            if base_name.endswith('.nii'):
                base_name = os.path.splitext(base_name)[0]  # for .nii.gz
            folder = os.path.dirname(in_nii) or "."
            output_csv = os.path.join(folder, f"roi_volume_{base_name}.csv")

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Label', 'Voxel Count', 'Voxel Size (mm^3)', 'Volume (mm^3)'])
            for label, voxel_count, voxel_size, volume in results:
                writer.writerow([label, voxel_count, voxel_size, volume])

        self._output_csv = output_csv
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_csv'] = os.path.abspath(self._output_csv)
        return outputs