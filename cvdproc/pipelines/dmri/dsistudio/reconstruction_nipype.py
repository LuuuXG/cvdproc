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
    out_file = Str(desc="Output Fib File")

class DSIstudioReconstruction(CommandLine):
    _cmd = dsi_studio_path + " --action=rec --check_btable=1"
    input_spec = DSIstudioReconstructionInputSpec
    output_spec = DSIstudioReconstructionOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.output:
            if os.path.isdir(self.inputs.output):
                base_name = os.path.basename(self.inputs.source)
                # remove .src.gz
                base_name_noext = os.path.splitext(os.path.splitext(base_name)[0])[0]
                outputs['out_file'] = os.path.join(self.inputs.output, base_name_noext + '.fib.gz')
            else:
                outputs['out_file'] = self.inputs.output
        else:
            base_name = os.path.basename(self.inputs.source)
            base_name_noext = os.path.splitext(os.path.splitext(base_name)[0])[0]
            outputs['out_file'] = os.path.abspath(base_name_noext + '.fib.gz')
        return outputs

if __name__ == '__main__':
    # Example usage
    create_src = DSIstudioReconstruction()
    create_src.inputs.source = '/mnt/f/BIDS/ALL/derivatives/dwi_pipeline/sub-WZCU002/ses-01/dsistudio/sub-WZCU002_ses-01_acq-DTIb1000_space-preprocdwi_desc-preproc_dwi.src.gz'
    create_src.inputs.method = 7
    create_src.inputs.qsdr_reso = 2.0
    create_src.inputs.other_output = 'fa,ad,rd,md,iso,rdi,nrdi,tensor'
    create_src.inputs.output = '/mnt/f/BIDS/ALL/derivatives/dwi_pipeline/sub-WZCU002/ses-01/dsistudio/sub-WZCU002_ses-01_acq-DTIb1000_space-preprocdwi_model-qsdr_dwimap.fib.gz'
    create_src.run()