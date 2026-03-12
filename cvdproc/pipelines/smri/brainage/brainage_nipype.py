import os
import subprocess
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str

from cvdproc.config.paths import get_package_path

class BrainAgeRInputSpec(CommandLineInputSpec):
    t1w_image = File(exists=True, mandatory=True, desc="Input T1-weighted image", argstr='%s', position=0)
    output_dir = Directory(mandatory=True, desc="Output directory", argstr='%s', position=1)
    output_csv_filename = Str(mandatory=True, desc="Output CSV filename", argstr='%s', position=2)

class BrainAgeROutputSpec(TraitedSpec):
    output_csv = File(exists=True, desc="Output CSV file")

class BrainAgeR(CommandLine):
    input_spec = BrainAgeRInputSpec
    output_spec = BrainAgeROutputSpec
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'brainage', 'brainageR_custom.sh')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_csv'] = os.path.join(self.inputs.output_dir, self.inputs.output_csv_filename)
        return outputs