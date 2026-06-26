import os
import subprocess
import nibabel as nib
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File
from traits.api import Bool, Int, Str, List, Float, Undefined, Either

from cvdproc.utils.python.basic_image_processor import extract_roi_means

# ----------------------------------------------
# calculate mean value given a ROI mask with many regions
# ----------------------------------------------
class CalcMeanInROIMaskInputSpec(BaseInterfaceInputSpec):
    image_file = File(exists=True, desc="Input image file", mandatory=True)
    roi_mask_file = File(exists=True, desc="ROI mask file with multiple regions", mandatory=True)

    ignore_background = Bool(True, usedefault=True)

    roi_label = List(Int, desc="ROI labels to compute (e.g. [1, 2])")

    # Optional CSV output
    output_csv = Either(
        File(),
        Undefined,
        desc="Optional output CSV file. If Undefined, CSV is not written.",
        usedefault=True,
    )

class CalcMeanInROIMaskOutputSpec(TraitedSpec):
    roi_label = List(Int, desc="ROI labels actually used")
    roi_mean_value = List(Float, desc="Mean values corresponding to roi_label")

    output_csv = Either(
        File(exists=True),
        Undefined,
        desc="Output CSV file (only if written)",
    )

class CalcMeanInROIMask(BaseInterface):
    input_spec = CalcMeanInROIMaskInputSpec
    output_spec = CalcMeanInROIMaskOutputSpec

    def _run_interface(self, runtime):
        labels, means = extract_roi_means(
            input_image=self.inputs.image_file,
            roi_image=self.inputs.roi_mask_file,
            ignore_background=self.inputs.ignore_background,
            roi_label=list(self.inputs.roi_label) if self.inputs.roi_label else None,
            output_csv=None if self.inputs.output_csv is Undefined else self.inputs.output_csv,
        )

        self._roi_labels = labels
        self._roi_means = means
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["roi_label"] = getattr(self, "_roi_labels", [])
        outputs["roi_mean_value"] = getattr(self, "_roi_means", [])

        if self.inputs.output_csv is not Undefined:
            outputs["output_csv"] = self.inputs.output_csv
        else:
            outputs["output_csv"] = Undefined

        return outputs

# ----------------------------------------------
# Merge colnames and data into a csv
# ----------------------------------------------
class MergeDataToCSVInputSpec(BaseInterfaceInputSpec):
    output_csv = File(desc="Output CSV file", mandatory=True)
    colnames = List(Str, desc="Column names", mandatory=True)
    data = List(Either(Str, Float, Int), desc="Data corresponding to column names", mandatory=True)

class MergeDataToCSVOutputSpec(TraitedSpec):
    output_csv = File(exists=True, desc="Output CSV file")

class MergeDataToCSV(BaseInterface):
    input_spec = MergeDataToCSVInputSpec
    output_spec = MergeDataToCSVOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        if len(self.inputs.colnames) != len(self.inputs.data):
            raise ValueError("Length of colnames and data must be the same.")

        df = pd.DataFrame([self.inputs.data], columns=self.inputs.colnames)
        df.to_csv(self.inputs.output_csv, index=False)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_csv"] = self.inputs.output_csv
        return outputs

# -----------------------------
# Weighted mean calculation
# -----------------------------
def weighted_mean_from_nifti(
    scalar_nii,
    weight_nii,
    out_txt,
    ignore_background=False,
):
    if not os.path.isfile(scalar_nii):
        raise FileNotFoundError(f"Scalar NIfTI not found: {scalar_nii}")
    if not os.path.isfile(weight_nii):
        raise FileNotFoundError(f"Weight NIfTI not found: {weight_nii}")

    scalar_img = nib.load(scalar_nii)
    weight_img = nib.load(weight_nii)

    scalar = scalar_img.get_fdata(dtype=np.float64)
    weight = weight_img.get_fdata(dtype=np.float64)

    if scalar.shape[:3] != weight.shape[:3]:
        raise RuntimeError(
            f"Shape mismatch: scalar {scalar.shape} vs weight {weight.shape}"
        )

    mask = (
        (weight > 0) &
        np.isfinite(weight) &
        np.isfinite(scalar)
    )

    if ignore_background:
        mask &= (scalar != 0)

    if not np.any(mask):
        raise RuntimeError("No valid voxels after masking.")

    w = weight[mask]
    x = scalar[mask]

    w_sum = float(np.sum(w))
    if w_sum <= 0:
        raise RuntimeError("Sum of weights is zero.")

    weighted_mean = float(np.sum(x * w) / w_sum)

    out_dir = os.path.dirname(out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_txt, "w") as f:
        f.write(f"{weighted_mean:.8f}\n")

    return out_txt, weighted_mean, int(np.sum(mask)), w_sum

class TDWeightedMeanInputSpec(BaseInterfaceInputSpec):
    scalar_nii = File(
        exists=True,
        mandatory=True,
        desc="Scalar NIfTI image (e.g., ICVF)",
    )
    weight_nii = File(
        exists=True,
        mandatory=True,
        desc="Weight NIfTI image (e.g., TDI, values in [0,1])",
    )
    out_txt = File(
        mandatory=True,
        desc="Output text file containing weighted mean value",
    )
    ignore_background = Bool(
        False,
        usedefault=True,
        desc="If True, ignore voxels where scalar value equals zero",
    )

class TDWeightedMeanOutputSpec(TraitedSpec):
    out_txt = File(exists=True, desc="Output text file with weighted mean")

class TDWeightedMean(BaseInterface):
    input_spec = TDWeightedMeanInputSpec
    output_spec = TDWeightedMeanOutputSpec

    def _run_interface(self, runtime):
        out_txt = os.path.abspath(self.inputs.out_txt)

        weighted_mean_from_nifti(
            scalar_nii=self.inputs.scalar_nii,
            weight_nii=self.inputs.weight_nii,
            out_txt=out_txt,
            ignore_background=self.inputs.ignore_background,
        )

        self._out_txt = out_txt
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_txt"] = getattr(
            self, "_out_txt", os.path.abspath(self.inputs.out_txt)
        )
        return outputs

# ------------------------
# Combine Two Binary Masks
# ------------------------
class CombineMasksInputSpec(BaseInterfaceInputSpec):
    mask1 = File(exists=True, mandatory=True, desc="First binary mask NIfTI file")
    mask2 = File(exists=True, mandatory=True, desc="Second binary mask NIfTI file")
    output_mask = File(mandatory=True, desc="Output combined binary mask NIfTI file")

class CombineMasksOutputSpec(TraitedSpec):
    output_mask = File(exists=True, desc="Output combined binary mask NIfTI file")

class CombineMasks(BaseInterface):
    input_spec = CombineMasksInputSpec
    output_spec = CombineMasksOutputSpec

    def _run_interface(self, runtime):
        # Load the two binary masks
        mask1_img = nib.load(self.inputs.mask1)
        mask2_img = nib.load(self.inputs.mask2)

        # Combine the masks using logical OR
        combined_data = np.logical_or(mask1_img.get_fdata(), mask2_img.get_fdata()).astype(np.uint8)

        # Save the combined mask
        combined_img = nib.Nifti1Image(combined_data, mask1_img.affine, mask1_img.header)
        nib.save(combined_img, self.inputs.output_mask)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_mask"] = os.path.abspath(self.inputs.output_mask)
        return outputs

# -----------------------------
# Remove mask region from NIfTI
# -----------------------------
class RemoveMaskRegionInputSpec(BaseInterfaceInputSpec):
    input_image = File(exists=True, mandatory=True, desc="Input NIfTI image")
    mask_image = File(exists=True, mandatory=True, desc="Mask NIfTI image")
    output_image = File(mandatory=True, desc="Output NIfTI image")
    mask_threshold = Float(0.5, usedefault=True, desc="Mask threshold")


class RemoveMaskRegionOutputSpec(TraitedSpec):
    output_image = File(exists=True, desc="Output NIfTI image")


class RemoveMaskRegion(BaseInterface):
    input_spec = RemoveMaskRegionInputSpec
    output_spec = RemoveMaskRegionOutputSpec

    def _run_interface(self, runtime):
        input_img = nib.load(self.inputs.input_image)
        mask_img = nib.load(self.inputs.mask_image)

        input_data = input_img.get_fdata(dtype=np.float64)
        mask_data = mask_img.get_fdata(dtype=np.float64)

        if input_data.shape != mask_data.shape:
            raise RuntimeError(
                f"Shape mismatch: input image {input_data.shape} vs mask image {mask_data.shape}"
            )

        # if not np.allclose(input_img.affine, mask_img.affine, atol=1e-4):
        #     raise RuntimeError("Affine mismatch between input image and mask image.")

        output_data = input_data.copy()
        output_data[mask_data > self.inputs.mask_threshold] = 0

        output_image = os.path.abspath(self.inputs.output_image)
        os.makedirs(os.path.dirname(output_image), exist_ok=True)

        output_img = nib.Nifti1Image(output_data, input_img.affine, input_img.header)
        output_img.set_data_dtype(input_img.get_data_dtype())
        nib.save(output_img, output_image)

        self._output_image = output_image
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_image"] = getattr(
            self, "_output_image", os.path.abspath(self.inputs.output_image)
        )
        return outputs