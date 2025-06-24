import os
import subprocess
import shutil
import nibabel as nib
import time
import numpy as np
import pandas as pd
from nipype import Node, Workflow, MapNode
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface, Merge
from traits.api import Bool, Int, Str, Either

######################
# WM mask generation #
######################
class GenerateWMMaskInputSpec(CommandLineInputSpec):
    fs_output = Directory(exists=True, desc="FreeSurfer output directory", argstr="--fs_output %s")
    fs_to_dwi_xfm = File(exists=True, desc="FreeSurfer to DWI transform file", argstr="--fs_to_dwi %s")
    dwi_data = File(exists=True, desc="DWI data file for flirt reference", argstr="--dwi %s")
    output_dir = Directory(desc="Output directory", argstr="--output_dir %s")
    exclude_masks = traits.List(desc="List of exclude mask files", argstr="--exclude %s")

class GenerateWMMaskOutputSpec(TraitedSpec):
    output_dir = Directory(exists=True, desc="Output directory")
    wm_mask = File(exists=True, desc="White matter mask file")
    wm_mask_raw = File(exists=True, desc="White matter mask file without excluding any other masks")

class GenerateWMMaskCommandLine(CommandLine):
    _cmd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bash", "generate_wm_from_fs.sh"))
    input_spec = GenerateWMMaskInputSpec
    output_spec = GenerateWMMaskOutputSpec
    #terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["wm_mask"] = os.path.join(self.inputs.output_dir, "WM_final.nii.gz")
        outputs["wm_mask_raw"] = os.path.join(self.inputs.output_dir, "WM_in_dwi.nii.gz")
        outputs["output_dir"] = self.inputs.output_dir
        return outputs

#########################
# Calculate scalar maps #
#########################
from cvdproc.utils.python.basic_image_processor import extract_roi_means

class CalculateScalarMapsInputSpec(BaseInterfaceInputSpec):
    dwi_data = traits.List(File(exists=True), desc="List of DWI data files", mandatory=True)
    wm_mask = File(exists=True, desc="White matter mask file", mandatory=True)
    output_dir = Directory(exists=True, desc="Output directory", mandatory=True)

class CalculateScalarMapsOutputSpec(TraitedSpec):
    stats_csv = traits.List(File(exists=True), desc="List of CSV files with statistics")

class CalculateScalarMaps(BaseInterface):
    input_spec = CalculateScalarMapsInputSpec
    output_spec = CalculateScalarMapsOutputSpec

    def _run_interface(self, runtime):
        dwi_data_list = self.inputs.dwi_data
        wm_mask = self.inputs.wm_mask
        output_dir = self.inputs.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        stats_csv_paths = []
        wm_mask_filename = os.path.basename(wm_mask)
        if wm_mask_filename.endswith(".nii"):
            wm_mask_filename = wm_mask_filename[:-4]
        elif wm_mask_filename.endswith(".nii.gz"):
            wm_mask_filename = wm_mask_filename[:-7]

        for dwi_data in dwi_data_list:
            dwi_data_filename = os.path.basename(dwi_data)
            if dwi_data_filename.endswith(".nii"):
                dwi_data_filename = dwi_data_filename[:-4]
            elif dwi_data_filename.endswith(".nii.gz"):
                dwi_data_filename = dwi_data_filename[:-7]

            output_csv = os.path.join(output_dir, f"{dwi_data_filename}_in_{wm_mask_filename}_stats.csv")
            stats_csv = extract_roi_means(
                dwi_data, wm_mask, ignore_background=True, output_path=output_csv
            )
            stats_csv_paths.append(stats_csv)

        self._stats_csv = stats_csv_paths
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["stats_csv"] = self._stats_csv
        return outputs

if __name__ == "__main__":
    test_wf = Workflow(name="test_wf")

    inputnode = Node(IdentityInterface(fields=["dti_fa", "dti_md", "dti_other"]), name="inputnode")
    inputnode.inputs.dti_fa = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/dti_FA.nii.gz"
    inputnode.inputs.dti_md = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/dti_MD.nii.gz"
    inputnode.inputs.dti_other = None

    merge_node = Node(Merge(3), name="merge_node")
    test_wf.connect([
        (inputnode, merge_node, [("dti_fa", "in1"), ("dti_md", "in2"), ("dti_other", "in3")]),
    ])

    wm_mask_node = Node(GenerateWMMaskCommandLine(), name="wm_mask_node")
    wm_mask_node.inputs.fs_output = "/mnt/f/BIDS/SVD_BIDS/derivatives/freesurfer/sub-SVD0035/ses-02"
    wm_mask_node.inputs.fs_to_dwi_xfm = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/fs_processing/freesurfer2fa.mat"
    wm_mask_node.inputs.dwi_data = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/dti_FA.nii.gz"
    wm_mask_node.inputs.output_dir = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/dwi_scalar_stats"
    wm_mask_node.inputs.exclude_masks = ['/mnt/f/BIDS/SVD_BIDS/derivatives/lesion_mask/sub-SVD0035/ses-02/dti_infarction.nii.gz']

    dwi_scalar_maps = MapNode(CalculateScalarMaps(), name="dwi_scalar_maps", iterfield=["dwi_data"])

    test_wf.connect([
        (wm_mask_node, dwi_scalar_maps, [("wm_mask", "wm_mask"), ("output_dir", "output_dir")]),
        (merge_node, dwi_scalar_maps, [("out", "dwi_data")]),
    ])

    test_wf.run()  # This will execute the node and print the output