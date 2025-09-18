import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError, Either

# binarize an image under a threshold

class FSLMathsUnderThrInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0, desc='Input image file')
    threshold = Float(mandatory=True, argstr='-uthr %f', position=1, desc='Upper threshold value')
    binarize = Bool(True, argstr='-bin', position=2, desc='Binarize the output image')
    out_file = Str(mandatory=True, argstr='%s', position=3, desc='Output image file')

class FSLMathsUnderThrOutputSpec(TraitedSpec):
    out_file = File(desc='Output image file')

class FSLMathsUnderThr(CommandLine):
    """
    Nipype interface for FSL fslmaths command to threshold an image under a value and binarize it.
    """
    _cmd = 'fslmaths'
    input_spec = FSLMathsUnderThrInputSpec
    output_spec = FSLMathsUnderThrOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

# binarize an image above a threshold
class FSLMathsThrInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0, desc='Input image file')
    threshold = Float(mandatory=True, argstr='-thr %f', position=1, desc='Threshold value')
    binarize = Bool(True, argstr='-bin', position=2, desc='Binarize the output image')
    out_file = Str(mandatory=True, argstr='%s', position=3, desc='Output image file')
class FSLMathsThrOutputSpec(TraitedSpec):
    out_file = File(desc='Output image file')
class FSLMathsThr(CommandLine):
    """
    Nipype interface for FSL fslmaths command to threshold an image above a value and binarize it.
    """
    _cmd = 'fslmaths'
    input_spec = FSLMathsThrInputSpec
    output_spec = FSLMathsThrOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs