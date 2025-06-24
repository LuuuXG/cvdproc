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

class Synb0InputSpec(CommandLineInputSpec):
    # T1W_IMG=$1
    # DWI_IMG=$2
    # OUTPUT_DIR=$3
    # DWI_JSON=$4
    # FMAP_DIR=$5

    # if [[ $# -ne 5 ]]; then
    # echo "Usage: $0 <T1w.nii.gz> <DWI.nii.gz> <synb0_output_dir> <dwi.json> <fmap_output_dir>"
    # exit 1
    # fi
    t1w_img = File(desc="Path to the T1-weighted image", exists=True, mandatory=True, argstr="%s", position=1)
    dwi_img = File(desc="Path to the DWI image. First volume is b0 image", exists=True, mandatory=True, argstr="%s", position=2)
    output_path_synb0 = Directory(desc="Output directory for the synb0 image", mandatory=True, argstr="%s", position=3)
    dwi_json = File(desc="Path to the DWI JSON file", exists=True, mandatory=True, argstr="%s", position=4)
    fmap_output_dir = Directory(desc="Output directory for the field map", mandatory=True, argstr="%s", position=5)

class Synb0OutputSpec(TraitedSpec):
    acqparam = File(desc="Path to the acqparam.txt file")
    b0_all = File(desc="Path to the b0_all image")
    b0_u = File(desc="Path to the b0_u image")

class Synb0(CommandLine):
    _cmd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bash", "run_synb0.sh"))
    input_spec = Synb0InputSpec
    output_spec = Synb0OutputSpec

    def _run_interface(self, runtime):
        import re
        # Run the command
        runtime = super(Synb0, self)._run_interface(runtime)

        # Parse output for fmap image path
        fmap_expr = re.compile(r'Fmap image:\s+(.*\.nii\.gz)')
        match = fmap_expr.search(runtime.stdout)
        if match:
            fmap_path = match.group(1).strip()
        else:
            fmap_path = None
            self.raise_exception("Cannot extract fmap path from output.")

        setattr(self, "_fmap_path", fmap_path)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["acqparam"] = os.path.join(self.inputs.output_path_synb0, "INPUTS", "acqparam.txt")
        outputs["b0_all"] = os.path.join(self.inputs.output_path_synb0, "b0_all.nii.gz")
        outputs["b0_u"] = getattr(self, "_fmap_path", None)
        return outputs

if __name__ == "__main__":
    # Example usage
    synb0 = Synb0()
    synb0.inputs.t1w_img = "/mnt/f/BIDS/WZdata/sub-WZMCI001/ses-01/anat/sub-WZMCI001_ses-01_acq-highres_T1w.nii.gz"
    synb0.inputs.dwi_img = "/mnt/f/BIDS/WZdata/sub-WZMCI001/ses-01/dwi/sub-WZMCI001_ses-01_acq-DTIb1000_dwi.nii.gz"
    synb0.inputs.output_path_synb0 = "/mnt/f/BIDS/WZdata/derivatives/synb0/sub-WZMCI001/ses-01"
    synb0.inputs.dwi_json = "/mnt/f/BIDS/WZdata/sub-WZMCI001/ses-01/dwi/sub-WZMCI001_ses-01_acq-DTIb1000_dwi.json"
    synb0.inputs.fmap_output_dir = "/mnt/f/BIDS/WZdata/sub-WZMCI001/ses-01/fmap"

    result = synb0.run()  # This will execute the interface
    print(result.outputs)