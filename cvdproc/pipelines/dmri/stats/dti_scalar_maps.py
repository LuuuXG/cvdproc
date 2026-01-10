import os
import subprocess
import shutil
import nibabel as nib
import time
import csv
import numpy as np
import pandas as pd
from nipype import Node, Workflow, MapNode
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface, Merge
from traits.api import Bool, Int, Str, Either, Float

# Or we already have a WM mask in DWI space
class GenerateNAWMMaskInputSpec(BaseInterfaceInputSpec):
    wm_mask = File(exists=True, desc="White matter mask file", mandatory=True)
    exclude_masks = traits.List(desc="List of exclude mask files", mandatory=True)
    output_mask = File(desc="Output NAWM mask file", mandatory=True)
    erode_mm = Float(desc="Erosion size in mm", mandatory=False, default_value=2.0)
class GenerateNAWMMaskOutputSpec(TraitedSpec):
    nawm_mask = File(exists=True, desc="Normal-appearing white matter mask file")
class GenerateNAWMMask(BaseInterface):
    input_spec = GenerateNAWMMaskInputSpec
    output_spec = GenerateNAWMMaskOutputSpec

    def _run_interface(self, runtime):
        wm_mask_img = nib.load(self.inputs.wm_mask)
        wm_mask_data = wm_mask_img.get_fdata().astype(bool)

        for exclude_mask_file in self.inputs.exclude_masks:
            exclude_mask_img = nib.load(exclude_mask_file)
            exclude_mask_data = exclude_mask_img.get_fdata().astype(bool)
            wm_mask_data[exclude_mask_data] = False

        # Erode the mask
        if self.inputs.erode_mm > 0:
            from scipy.ndimage import binary_erosion, generate_binary_structure
            voxel_sizes = wm_mask_img.header.get_zooms()
            erode_voxels = [int(np.ceil(self.inputs.erode_mm / vs)) for vs in voxel_sizes]
            struct = generate_binary_structure(3, 1)
            for _ in range(max(erode_voxels)):
                wm_mask_data = binary_erosion(wm_mask_data, structure=struct)

        nawm_mask_img = nib.Nifti1Image(wm_mask_data.astype(np.uint8), wm_mask_img.affine, wm_mask_img.header)
        nib.save(nawm_mask_img, self.inputs.output_mask)

        self._nawm_mask = self.inputs.output_mask
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["nawm_mask"] = self._nawm_mask
        return outputs


#########################
# Calculate scalar maps #
#########################
from cvdproc.utils.python.basic_image_processor import extract_roi_means

class CalculateScalarMapsInputSpec(BaseInterfaceInputSpec):
    data_files = traits.List(traits.Str, desc="List of scalar map files to process", mandatory=True)
    mask_file = File(exists=True, desc="Mask/ROI file", mandatory=True)

    roi_label = traits.Int(desc="ROI label to compute mean value", mandatory=True)

    colnames = traits.List(traits.Str, desc="Column names for the output CSV", mandatory=True)

    output_csv = traits.File(desc="Output CSV file", mandatory=True)


class CalculateScalarMapsOutputSpec(TraitedSpec):
    output_csv = File(exists=True, desc="Output CSV file")


class CalculateScalarMaps(BaseInterface):
    input_spec = CalculateScalarMapsInputSpec
    output_spec = CalculateScalarMapsOutputSpec

    def _safe_exists(self, p):
        if p is None:
            return False
        p = str(p).strip()
        if p == "":
            return False
        return os.path.exists(p)

    def _run_interface(self, runtime):
        data_files = list(self.inputs.data_files)
        colnames = list(self.inputs.colnames)

        if len(data_files) != len(colnames):
            raise ValueError(f"Length mismatch: data_files={len(data_files)} vs colnames={len(colnames)}")

        mask_file = self.inputs.mask_file
        roi_label = int(self.inputs.roi_label)
        out_csv = str(self.inputs.output_csv)

        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        values = []
        for f in data_files:
            if not self._safe_exists(f):
                values.append(float("nan"))
                continue

            # extract_roi_means returns (labels, means) in the revised version
            labels, means = extract_roi_means(
                input_image=f,
                roi_image=mask_file,
                ignore_background=True,
                roi_label=[roi_label],
                output_csv=None,
            )
            # means should be length 1
            if means is None or len(means) == 0:
                values.append(float("nan"))
            else:
                values.append(float(means[0]))

        # write a single-row CSV: header + one row
        with open(out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(colnames)
            w.writerow(values)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_csv"] = str(self.inputs.output_csv)
        return outputs