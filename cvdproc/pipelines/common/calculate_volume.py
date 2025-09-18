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

class CalculateVolumeInputSpec(BaseInterfaceInputSpec):
    in_nii = File(exists=True, mandatory=True, desc='Input NIfTI file path')
    output_csv = Str(desc='Output CSV file path. If not provided, will be saved in the same directory as input with prefix "volume_"')
class CalculateVolumeOutputSpec(TraitedSpec):
    output_csv = File(desc='Output CSV file path with volume calculations')
class CalculateVolume(BaseInterface):
    input_spec = CalculateVolumeInputSpec
    output_spec = CalculateVolumeOutputSpec

    def _run_interface(self, runtime):
        in_nii = self.inputs.in_nii
        output_csv = self.inputs.output_csv

        # Load input NIfTI file
        img = nib.load(in_nii)
        data = img.get_fdata()
        unique_labels = np.unique(data)

        # voxel size in mm^3
        voxel_size = np.abs(np.linalg.det(img.affine[:3, :3]))

        # calculate voxel count and volume per label
        results = []
        for label in unique_labels:
            voxel_count = np.sum(data == label)
            volume = voxel_count * voxel_size
            results.append((label, voxel_count, voxel_size, volume))

        # determine output path
        if not output_csv:
            base_name, ext = os.path.splitext(os.path.basename(in_nii))
            if base_name.endswith('.nii'):
                base_name = os.path.splitext(base_name)[0]  # for .nii.gz
            folder = os.path.dirname(in_nii) or "."
            output_csv = os.path.join(folder, f"volume_{base_name}.csv")

        # write to CSV
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