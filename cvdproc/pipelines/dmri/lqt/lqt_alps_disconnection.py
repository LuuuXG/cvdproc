from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits
import os
import csv
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

from cvdproc.config.paths import get_package_path


L_SCR = get_package_path("pipelines", "external", "alps", "ROIs_JHU_ALPS", "L_SCR.nii.gz")
L_SLF = get_package_path("pipelines", "external", "alps", "ROIs_JHU_ALPS", "L_SLF.nii.gz")
R_SCR = get_package_path("pipelines", "external", "alps", "ROIs_JHU_ALPS", "R_SCR.nii.gz")
R_SLF = get_package_path("pipelines", "external", "alps", "ROIs_JHU_ALPS", "R_SLF.nii.gz")


class LQTALPSDisconnectionInputSpec(BaseInterfaceInputSpec):
    tdi_file = File(exists=True, mandatory=True, desc="LQT disconnection TDI image")
    output_csv = File(mandatory=True, desc="Output CSV file")
    resample_rois = traits.Bool(True, usedefault=True, desc="Resample ROI masks to the TDI image grid if needed")
    roi_threshold = traits.Float(0.5, usedefault=True, desc="Threshold used after nearest-neighbor ROI resampling")


class LQTALPSDisconnectionOutputSpec(TraitedSpec):
    output_csv = File(desc="Output CSV file containing ALPS ROI disconnection metrics")


class LQTALPSDisconnection(BaseInterface):
    input_spec = LQTALPSDisconnectionInputSpec
    output_spec = LQTALPSDisconnectionOutputSpec

    def _run_interface(self, runtime):
        tdi_img = nib.load(self.inputs.tdi_file)
        tdi_data = tdi_img.get_fdata(dtype=np.float32)
        tdi_data = np.nan_to_num(tdi_data, nan=0.0, posinf=0.0, neginf=0.0)

        roi_masks = {
            "L_SCR": self._load_roi_mask(L_SCR, tdi_img),
            "L_SLF": self._load_roi_mask(L_SLF, tdi_img),
            "R_SCR": self._load_roi_mask(R_SCR, tdi_img),
            "R_SLF": self._load_roi_mask(R_SLF, tdi_img),
        }

        metrics = {
            "L_SCR_mean_disconnection": self._masked_mean(tdi_data, roi_masks["L_SCR"]),
            "L_SLF_mean_disconnection": self._masked_mean(tdi_data, roi_masks["L_SLF"]),
            "R_SCR_mean_disconnection": self._masked_mean(tdi_data, roi_masks["R_SCR"]),
            "R_SLF_mean_disconnection": self._masked_mean(tdi_data, roi_masks["R_SLF"]),
            "L_ALPS_mean_disconnection": self._masked_mean(
                tdi_data,
                np.logical_or(roi_masks["L_SCR"], roi_masks["L_SLF"])
            ),
            "R_ALPS_mean_disconnection": self._masked_mean(
                tdi_data,
                np.logical_or(roi_masks["R_SCR"], roi_masks["R_SLF"])
            ),
        }

        output_csv = os.path.abspath(self.inputs.output_csv)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)

        print(f"Saved ALPS ROI disconnection metrics: {output_csv}")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return runtime

    def _load_roi_mask(self, roi_file, tdi_img):
        roi_img = nib.load(roi_file)

        same_shape = roi_img.shape[:3] == tdi_img.shape[:3]
        same_affine = np.allclose(roi_img.affine, tdi_img.affine, atol=1e-5)

        if same_shape and same_affine:
            roi_data = roi_img.get_fdata(dtype=np.float32)
        else:
            if not self.inputs.resample_rois:
                raise ValueError(
                    f"ROI grid does not match TDI grid and resample_rois is False: {roi_file}"
                )

            roi_resampled = resample_from_to(
                roi_img,
                (tdi_img.shape[:3], tdi_img.affine),
                order=0
            )
            roi_data = roi_resampled.get_fdata(dtype=np.float32)

        roi_data = np.nan_to_num(roi_data, nan=0.0, posinf=0.0, neginf=0.0)
        roi_mask = roi_data > float(self.inputs.roi_threshold)

        if np.sum(roi_mask) == 0:
            raise ValueError(f"Empty ROI mask after loading or resampling: {roi_file}")

        return roi_mask

    def _masked_mean(self, data, mask):
        values = data[mask]
        values = values[np.isfinite(values)]

        if values.size == 0:
            return np.nan

        return float(np.mean(values))

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_csv"] = os.path.abspath(self.inputs.output_csv)
        return outputs