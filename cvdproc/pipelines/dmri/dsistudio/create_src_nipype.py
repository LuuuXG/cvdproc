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

# if this dsi_studio_path does not exit, assume it is in the PATH (dsi_studio XXX can work)
if not os.path.exists(dsi_studio_path):
    dsi_studio_path = 'dsi_studio'

class CreateSRCInputSpec(CommandLineInputSpec):
    source = Str(desc="DWI File", argstr = "--source=%s", mandatory=True)
    bval = Str(desc="bval File", argstr = "--bval=%s", mandatory=False)
    bvec = Str(desc="bvec File", argstr = "--bvec=%s", mandatory=False)
    output = Str(desc="Output SRC File Name or Directory", argstr = "--output=%s", mandatory=False)
    bids = Int(desc="BIDS Format", argstr="--bids=%d", mandatory=False, default_value=0)

class CreateSRCOutputSpec(TraitedSpec):
    out_file = Str(desc="Output SRC File")

class CreateSRC(CommandLine):
    _cmd = dsi_studio_path + " --action=src"
    input_spec = CreateSRCInputSpec
    output_spec = CreateSRCOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.output:
            if os.path.isdir(self.inputs.output):
                base_name = os.path.basename(self.inputs.source)
                # remove .nii.gz
                base_name_noext = os.path.splitext(os.path.splitext(base_name)[0])[0]
                outputs['out_file'] = os.path.join(self.inputs.output, base_name_noext + '.src.gz')
            else:
                outputs['out_file'] = self.inputs.output
        else:
            base_name = os.path.basename(self.inputs.source)
            base_name_noext = os.path.splitext(os.path.splitext(base_name)[0])[0]
            outputs['out_file'] = os.path.abspath(base_name_noext + '.src.gz')
        return outputs