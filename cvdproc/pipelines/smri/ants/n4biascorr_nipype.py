import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError, Either

class SimpleN4BiasFieldCorrectionInputSpec(CommandLineInputSpec):
    input_image = File(exists=True, mandatory=True, argstr='%s', position=0, desc='Input image to be corrected')
    output_image = Str(mandatory=True, argstr='%s', position=1, desc='Output corrected image')
    output_bias = Str(mandatory=True, argstr='%s', position=2, desc='Output bias field image')

class SimpleN4BiasFieldCorrectionOutputSpec(TraitedSpec):
    output_image = File(desc='Output corrected image')
    output_bias = File(desc='Output bias field image')

class SimpleN4BiasFieldCorrection(CommandLine):
    """
    Nipype interface for ANTs N4BiasFieldCorrection command.
    """
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bash', 'ants', 'simple_n4biascorr.sh'))
    input_spec = SimpleN4BiasFieldCorrectionInputSpec
    output_spec = SimpleN4BiasFieldCorrectionOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_image'] = os.path.abspath(self.inputs.output_image)
        outputs['output_bias'] = os.path.abspath(self.inputs.output_bias)
        return outputs