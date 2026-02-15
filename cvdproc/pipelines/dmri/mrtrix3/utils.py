import os
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
import csv
import numpy as np

from cvdproc.config.paths import get_package_path

class RemoveTractRegionInputSpec(CommandLineInputSpec):
    wm_mask = File(desc="Path to the input white matter mask", mandatory=True, argstr="%s", position=0)
    tck_file = File(desc="Path to the input track file", mandatory=True, argstr="%s", position=1)
    out_tract_mask = File(desc="Path to the output tract mask", mandatory=True, argstr="%s", position=2)
    out_tdi_norm = File(desc="Path to the output TDI (normalized)", mandatory=True, argstr="%s", position=3)
    out_wm_mask = File(desc="Path to the output WM mask (excluding tract)", mandatory=True, argstr="%s", position=4)

class RemoveTractRegionOutputSpec(TraitedSpec):
    out_tract_mask = File(desc="Path to the output tract mask", mandatory=True)
    out_tdi_norm = File(desc="Path to the output TDI (normalized)", mandatory=True)
    out_wm_mask = File(desc="Path to the output WM mask (excluding tract)", mandatory=True)

class RemoveTractRegion(CommandLine):
    input_spec = RemoveTractRegionInputSpec
    output_spec = RemoveTractRegionOutputSpec
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'mrtrix3', 'remove_conn_region.sh')
    #terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs.update({
            'out_tract_mask': self.inputs.out_tract_mask,
            'out_tdi_norm': self.inputs.out_tdi_norm,
            'out_wm_mask': self.inputs.out_wm_mask,
        })
        return outputs