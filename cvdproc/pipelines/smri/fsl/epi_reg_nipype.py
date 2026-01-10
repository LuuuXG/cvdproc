import os
import shutil
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory, Bool, Either

from cvdproc.config.paths import get_package_path

class EPIRegNipypeInputSpec(CommandLineInputSpec):
    epi = Str(mandatory=True, argstr="%s", position=0, desc="EPI image")
    t1 = Str(mandatory=True, argstr="%s", position=1, desc="T1-weighted image")
    t1brain = Str(mandatory=True, argstr="%s", position=2, desc="Skull-stripped T1-weighted image")
    out = Str(mandatory=True, argstr="%s", position=3, desc="Output registered EPI image")
    wmseg = Str(mandatory=True, argstr="%s", position=4, desc="White matter segmentation image")
    epi2t1mat = Str(mandatory=True, argstr="%s", position=5, desc="Output EPI to T1 transformation matrix")
    t12epimat = Str(mandatory=True, argstr="%s", position=6, desc="Output T1 to EPI transformation matrix")

class EPIRegNipypeOutputSpec(TraitedSpec):
    out = Either(File, None, desc='Output registered EPI image file')
    epi2t1mat = Either(File, None, desc='Output EPI to T1 transformation matrix file')
    t12epimat = Either(File, None, desc='Output T1 to EPI transformation matrix file')

class EPIRegNipype(CommandLine):
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'fsl', 'epi_reg_xfm.sh')
    input_spec = EPIRegNipypeInputSpec
    output_spec = EPIRegNipypeOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out'] = self.inputs.out
        outputs['epi2t1mat'] = self.inputs.epi2t1mat
        outputs['t12epimat'] = self.inputs.t12epimat
        return outputs