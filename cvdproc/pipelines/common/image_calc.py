import os
import subprocess
import nibabel as nib
import numpy as np
import csv
import os as _os_image_calc

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits
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
    output_csv = Either(File(), Undefined, desc="Optional output CSV file. If Undefined, CSV is not written.", usedefault=True)

class CalcMeanInROIMaskOutputSpec(TraitedSpec):
    roi_label = List(Int, desc="ROI labels actually used")
    roi_mean_value = List(Float, desc="Mean values corresponding to roi_label")
    output_csv = Either(File(exists=True), Undefined, desc="Output CSV file (only if written)")

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
        raise RuntimeError(f"Shape mismatch: scalar {scalar.shape} vs weight {weight.shape}")

    mask = (weight > 0) & np.isfinite(weight) & np.isfinite(scalar)

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
    scalar_nii = File(exists=True, mandatory=True, desc="Scalar NIfTI image (e.g., ICVF)")
    weight_nii = File(exists=True, mandatory=True, desc="Weight NIfTI image (e.g., TDI, values in [0,1])")
    out_txt = File(mandatory=True, desc="Output text file containing weighted mean value")
    ignore_background = Bool(False, usedefault=True, desc="If True, ignore voxels where scalar value equals zero")

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
        outputs["out_txt"] = getattr(self, "_out_txt", os.path.abspath(self.inputs.out_txt))
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
        outputs["output_image"] = getattr(self, "_output_image", os.path.abspath(self.inputs.output_image))
        return outputs


# ----------------------------------------------
# Calculate scalar map statistics within ROI(s)
# ----------------------------------------------
class CalculateScalarMapsInputSpec(BaseInterfaceInputSpec):
    data_files = traits.List(traits.Str, desc="List of scalar map files to process", mandatory=True)
    mask_file = File(exists=True, desc="Binary mask or multi-label ROI file", mandatory=True)
    roi_label = traits.Either(traits.Int, traits.Any, desc="Single ROI label. If Undefined, roi_labels or all nonzero labels are used.")
    roi_labels = traits.List(traits.Int, desc="Selected ROI labels. If empty, roi_label or all nonzero labels are used.")
    colnames = traits.List(traits.Str, desc="Column names for scalar maps", mandatory=True)
    output_csv = traits.File(desc="Output CSV file", mandatory=True)
    ignore_background = traits.Bool(True, usedefault=True)
    statistic = traits.Enum("mean", "median", desc="Statistic to extract from ROI", usedefault=True)
    combine_rois = traits.Bool(False, usedefault=True, desc="If True, selected labels are combined into one ROI.")


class CalculateScalarMapsOutputSpec(TraitedSpec):
    output_csv = File(exists=True, desc="Output CSV file")


class CalculateScalarMaps(BaseInterface):
    input_spec = CalculateScalarMapsInputSpec
    output_spec = CalculateScalarMapsOutputSpec

    @staticmethod
    def _safe_exists(path):
        if path is None:
            return False
        path = str(path).strip()
        return path != "" and os.path.exists(path)

    @staticmethod
    def _same_grid(img, roi_img):
        return img.shape[:3] == roi_img.shape[:3] and np.allclose(img.affine, roi_img.affine, atol=1e-4)

    @staticmethod
    def _load_3d_data(image_file, roi_img):
        img = nib.load(image_file)
        if not CalculateScalarMaps._same_grid(img, roi_img):
            raise RuntimeError(f"Grid mismatch between scalar map and ROI mask: {image_file}")
        data = img.get_fdata(dtype=np.float64)
        if data.ndim == 4:
            if data.shape[3] != 1:
                raise RuntimeError(f"Scalar map must be 3D or 4D with one volume, got {data.shape}: {image_file}")
            data = data[..., 0]
        return data

    @staticmethod
    def _extract_value(data, roi_mask, ignore_background, statistic):
        valid = roi_mask & np.isfinite(data)
        if ignore_background:
            valid &= data != 0
        if not np.any(valid):
            return float("nan")
        values = data[valid]
        if statistic == "mean":
            return float(np.mean(values))
        if statistic == "median":
            return float(np.median(values))
        raise ValueError(f"Unsupported statistic: {statistic}")

    @staticmethod
    def _get_selected_labels(roi_data, roi_label, roi_labels):
        if roi_labels is not Undefined and len(roi_labels) > 0:
            return [int(x) for x in roi_labels]
        if roi_label is not Undefined:
            return [int(roi_label)]
        labels = np.unique(roi_data[np.isfinite(roi_data)])
        return [int(np.round(x)) for x in labels if x != 0]

    @staticmethod
    def _derive_colname_from_filename(filepath):
        """Extract metric name from BIDS filename: '_param-<value>' -> '<VALUE>'."""
        import re
        basename = os.path.basename(str(filepath))
        m = re.search(r'_param-([a-zA-Z0-9]+)', basename)
        if m:
            return m.group(1).upper()
        # fallback: extract suffix label before extension (e.g. '_Chidia.nii.gz' -> 'CHIDIA')
        m = re.search(r'_([a-zA-Z]+)\.(nii|nii\.gz|mgh|mgz)$', basename)
        if m:
            return m.group(1).upper()
        if basename.endswith('.nii.gz'):
            return basename[:-7].upper()
        return os.path.splitext(basename)[0].upper()

    def _run_interface(self, runtime):
        data_files = list(self.inputs.data_files)
        colnames = list(self.inputs.colnames)

        if len(data_files) != len(colnames):
            colnames = [self._derive_colname_from_filename(f) for f in data_files]

        mask_file = str(self.inputs.mask_file)
        out_csv = os.path.abspath(str(self.inputs.output_csv))
        ignore_background = bool(self.inputs.ignore_background)
        statistic = str(self.inputs.statistic)
        combine_rois = bool(self.inputs.combine_rois)

        roi_img = nib.load(mask_file)
        roi_data = np.round(roi_img.get_fdata(dtype=np.float64))

        if roi_data.ndim == 4:
            if roi_data.shape[3] != 1:
                raise RuntimeError(f"ROI mask must be 3D or 4D with one volume, got {roi_data.shape}: {mask_file}")
            roi_data = roi_data[..., 0]

        scalar_data = []
        for image_file in data_files:
            if self._safe_exists(image_file):
                scalar_data.append(self._load_3d_data(image_file, roi_img))
            else:
                scalar_data.append(None)

        selected_labels = self._get_selected_labels(roi_data, self.inputs.roi_label, self.inputs.roi_labels)

        if len(selected_labels) == 0:
            raise ValueError("No ROI labels were selected.")

        if combine_rois:
            roi_mask = np.isin(roi_data, selected_labels)
            row = []
            for data in scalar_data:
                if data is None:
                    row.append(float("nan"))
                else:
                    row.append(self._extract_value(data, roi_mask, ignore_background, statistic))
            header = colnames
            rows = [row]
        else:
            header = ["roi_label"] + colnames
            rows = []
            for roi_label in selected_labels:
                roi_mask = np.equal(roi_data, roi_label)
                row = [roi_label]
                for data in scalar_data:
                    if data is None:
                        row.append(float("nan"))
                    else:
                        row.append(self._extract_value(data, roi_mask, ignore_background, statistic))
                rows.append(row)

        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_csv, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            writer.writerows(rows)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_csv"] = os.path.abspath(str(self.inputs.output_csv))
        return outputs