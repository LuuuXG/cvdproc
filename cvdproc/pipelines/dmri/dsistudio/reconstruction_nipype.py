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
from traits.api import Bool, Int, Str, Float, Either

from cvdproc.config.paths import get_package_path

dsi_studio_path = get_package_path('data', 'lqt', 'extdata', 'DSI_studio', 'dsi-studio', 'dsi_studio')

class DSIstudioReconstructionInputSpec(CommandLineInputSpec):
    source = Str(desc="SRC File", argstr="--source=%s", mandatory=True)
    method = Int(desc="Reconstruction Method. 1=DTI, 4=GQI, 7=QSDR", argstr="--method=%d", mandatory=True)
    param0 = Float(desc="Parameter 0", argstr="--param0=%f", mandatory=False, default_value=1.25)
    output = Str(desc="Output Fib File Name or Directory", argstr="--output=%s", mandatory=False)
    thread_count = Int(desc="Number of Threads", argstr="--thread_count=%d", mandatory=False)
    qsdr_reso = Float(desc="QSDR Resolution", argstr="--qsdr_reso=%f", mandatory=False, default_value=2.0)
    other_output = Str(desc="Other Output File", argstr="--other_output=%s", mandatory=False)

class DSIstudioReconstructionOutputSpec(TraitedSpec):
    out_file = File(desc="Output Fib File")

class DSIstudioReconstruction(CommandLine):
    _cmd = dsi_studio_path + " --action=rec --check_btable=1"
    input_spec = DSIstudioReconstructionInputSpec
    output_spec = DSIstudioReconstructionOutputSpec

    def _expected_out_file(self):
        """
        Determine the expected output fib file path.
        """
        if self.inputs.output:
            if os.path.isdir(self.inputs.output):
                base_name = os.path.basename(self.inputs.source)
                base_name_noext = os.path.splitext(os.path.splitext(base_name)[0])[0]
                return os.path.join(self.inputs.output, base_name_noext + ".fib.gz")
            else:
                return self.inputs.output
        else:
            base_name = os.path.basename(self.inputs.source)
            base_name_noext = os.path.splitext(os.path.splitext(base_name)[0])[0]
            return os.path.abspath(base_name_noext + ".fib.gz")

    def _run_interface(self, runtime):
        out_file = self._expected_out_file()

        if out_file and os.path.exists(out_file):
            runtime.stdout = f"DSI Studio reconstruction skipped (output exists): {out_file}\n"
            runtime.stderr = ""
            runtime.returncode = 0
            return runtime

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._expected_out_file()
        return outputs
