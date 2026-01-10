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