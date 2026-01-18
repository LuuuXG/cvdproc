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
import tempfile

# Or we already have a WM mask in DWI space
class GenerateNAWMMaskInputSpec(BaseInterfaceInputSpec):
    wm_mask = File(exists=True, desc="White matter mask file", mandatory=True)
    exclude_mask = File(exists=True, desc="Exclude mask file", mandatory=False)
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

        if self.inputs.exclude_mask:
            exclude_mask_img = nib.load(self.inputs.exclude_mask)
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

def tdi_weighted_mean_with_background_value(scalar_nii, weight_nii, background_value=-1):
    """
    Compute TDI-weighted mean of a scalar image within the support of a TDI image.

    Inputs
    ------
    scalar_nii : str
        Scalar NIfTI (e.g., FA/MD/ICVF/ODI/ISO).
    weight_nii : str
        TDI NIfTI used as weights. Typically in [0,1] but any non-negative weights are acceptable.
    background_value : int
        Semantics:
          -1: automatically ignore scalar values {0, 1}
          >=0: ignore voxels where scalar == background_value

    Returns
    -------
    float
        Weighted mean = sum(scalar * weight) / sum(weight) over valid voxels.
        Returns np.nan if no valid voxels.
    """
    scalar_nii = str(scalar_nii)
    weight_nii = str(weight_nii)

    if not os.path.isfile(scalar_nii):
        raise FileNotFoundError(f"Scalar NIfTI not found: {scalar_nii}")
    if not os.path.isfile(weight_nii):
        raise FileNotFoundError(f"Weight NIfTI not found: {weight_nii}")

    s_img = nib.load(scalar_nii)
    w_img = nib.load(weight_nii)

    s = s_img.get_fdata(dtype=np.float64)
    w = w_img.get_fdata(dtype=np.float64)

    if s.shape[:3] != w.shape[:3]:
        raise ValueError(f"Shape mismatch: scalar {s.shape} vs weight {w.shape}")

    # Base validity: finite, weight > 0
    mask = np.isfinite(s) & np.isfinite(w) & (w > 0)

    # Background filtering on scalar values
    if int(background_value) == -1:
        mask &= (s != 0) & (s != 1)
    else:
        mask &= (s != int(background_value))

    if not np.any(mask):
        return float("nan")

    ww = w[mask]
    ss = s[mask]

    denom = float(np.sum(ww))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")

    num = float(np.sum(ss * ww))
    if not np.isfinite(num):
        return float("nan")

    return num / denom

class CalculateTDIWeightedScalarsInputSpec(BaseInterfaceInputSpec):
    data_files = traits.List(
        traits.Str,
        mandatory=True,
        desc="List of scalar NIfTI files (e.g., ICVF, ODI, ISO)",
    )

    weight_file = File(
        exists=True,
        mandatory=True,
        desc="TDI NIfTI file used as weights",
    )

    colnames = traits.List(
        traits.Str,
        mandatory=True,
        desc="Column names for the output CSV (same length as data_files)",
    )

    output_csv = File(
        mandatory=True,
        desc="Output CSV file (single-row)",
    )

    background_value = traits.Int(
        -1,
        usedefault=True,
        desc="Background value rule. "
             "Use -1 to automatically ignore {0,1}. "
             "Use >=0 to ignore that exact scalar value.",
    )


class CalculateTDIWeightedScalarsOutputSpec(TraitedSpec):
    output_csv = File(exists=True, desc="Output CSV file")


class CalculateTDIWeightedScalars(BaseInterface):
    input_spec = CalculateTDIWeightedScalarsInputSpec
    output_spec = CalculateTDIWeightedScalarsOutputSpec

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
            raise ValueError(
                f"Length mismatch: data_files={len(data_files)} vs colnames={len(colnames)}"
            )

        weight_file = os.path.abspath(str(self.inputs.weight_file))
        out_csv = os.path.abspath(str(self.inputs.output_csv))
        background_value = int(self.inputs.background_value)

        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        values = []
        for f in data_files:
            if not self._safe_exists(f):
                values.append(float("nan"))
                continue

            try:
                val = tdi_weighted_mean_with_background_value(
                    scalar_nii=f,
                    weight_nii=weight_file,
                    background_value=background_value,
                )
            except Exception as e:
                val = float("nan")
                raise RuntimeError(f"Failed on scalar {f}: {e}")

            values.append(float(val))

        with open(out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(colnames)
            w.writerow(["NA" if (isinstance(v, float) and np.isnan(v)) else f"{v:.6f}" for v in values])

        if not os.path.isfile(out_csv):
            raise RuntimeError(f"Failed to write output CSV: {out_csv}")

        self._out_csv = out_csv
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_csv"] = getattr(self, "_out_csv", os.path.abspath(str(self.inputs.output_csv)))
        return outputs

class TckSampleMultiScalarProfileInputSpec(BaseInterfaceInputSpec):
    tck_file = File(
        exists=True,
        mandatory=True,
        desc="Input TCK file (single bundle)",
    )

    scalar_files = traits.List(
        traits.Str,
        mandatory=True,
        desc="List of scalar NIfTI files",
    )

    scalar_names = traits.List(
        traits.Str,
        mandatory=True,
        desc="Names for each scalar (same length as scalar_files)",
    )

    output_csv = File(
        mandatory=True,
        desc="Output CSV: rows=points, cols=scalars",
    )

    precise = traits.Bool(
        False,
        usedefault=True,
        desc="Use tcksample -precise",
    )

    nointerp = traits.Bool(
        False,
        usedefault=True,
        desc="Use tcksample -nointerp",
    )


class TckSampleMultiScalarProfileOutputSpec(TraitedSpec):
    output_csv = File(exists=True)


class TckSampleMultiScalarProfile(BaseInterface):
    input_spec = TckSampleMultiScalarProfileInputSpec
    output_spec = TckSampleMultiScalarProfileOutputSpec

    def _run_interface(self, runtime):
        tck = os.path.abspath(self.inputs.tck_file)
        scalar_files = list(self.inputs.scalar_files)
        scalar_names = list(self.inputs.scalar_names)

        if len(scalar_files) != len(scalar_names):
            raise ValueError("scalar_files and scalar_names length mismatch")

        out_csv = os.path.abspath(self.inputs.output_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        profiles = []
        n_points_ref = None
        pending_nan_indices = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, (scalar, name) in enumerate(zip(scalar_files, scalar_names)):
                # Missing scalar -> fill later with NA
                if not os.path.exists(scalar):
                    profiles.append(None)
                    pending_nan_indices.append(idx)
                    continue

                tmp_txt = os.path.join(tmpdir, f"{name}.txt")

                cmd = ["tcksample", "-force"]
                if self.inputs.precise:
                    cmd.append("-precise")
                if self.inputs.nointerp:
                    cmd.append("-nointerp")

                cmd += [tck, scalar, tmp_txt]
                subprocess.run(cmd, check=True)

                data = np.loadtxt(tmp_txt)
                if data.ndim != 2:
                    raise RuntimeError(f"Invalid tcksample output for {name}: {data.shape}")

                if n_points_ref is None:
                    n_points_ref = data.shape[1]
                elif data.shape[1] != n_points_ref:
                    raise RuntimeError("Inconsistent number of points across scalars")

                mean_profile = np.nanmean(data, axis=0)
                profiles.append(mean_profile)

        if n_points_ref is None:
            raise RuntimeError("All scalar files are missing or failed; cannot determine number of points.")

        for idx in pending_nan_indices:
            profiles[idx] = np.full(n_points_ref, np.nan)

        mat = np.stack(profiles, axis=1)

        with open(out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(scalar_names)
            for row in mat:
                w.writerow(["NA" if np.isnan(v) else f"{v:.6f}" for v in row])

        if not os.path.isfile(out_csv):
            raise RuntimeError(f"Failed to write output CSV: {out_csv}")

        self._out_csv = out_csv
        return runtime


    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_csv"] = getattr(self, "_out_csv", os.path.abspath(self.inputs.output_csv))
        return outputs

if __name__ == "__main__":
    from nipype import Node
    tck_dwimap_node = Node(TckSampleMultiScalarProfile(), name="tck_dwimap")
    tck_dwimap_node.inputs.tck_file = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OR_streamlines.tck"
    tck_dwimap_node.inputs.scalar_files = [
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/NODDI/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_model-noddi_param-icvf_dwimap.nii.gz",
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/NODDI/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_model-noddi_param-isovf_dwimap.nii.gz"
    ]
    tck_dwimap_node.inputs.scalar_names = ["ICVF", "ISOVF"]
    tck_dwimap_node.inputs.output_csv = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/L_OR_ICVF_ISOVF.csv"
    tck_dwimap_node.run()