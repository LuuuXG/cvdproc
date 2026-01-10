import os
import shutil
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory, Bool, Either

class SynthMorphInputSpec(CommandLineInputSpec):
    moving_image = File(exists=True, desc="The moving image to be registered", mandatory=True, argstr='%s', position=-2)
    fixed_image = File(exists=True, desc="The fixed image to register to", mandatory=True, argstr='%s', position=-1)
    output_image = Str(desc="The output registered image", argstr='-o %s', position=0)
    fixed_to_moving_transform = Str(desc="Output transform from fixed to moving", argstr='-t %s', position=1)
    moving_to_fixed_transform = Str(desc="Output transform from moving to fixed", argstr='-T %s', position=2)
    use_gpu = Bool(False, desc="Use GPU acceleration", argstr='-g', position=3)

class SynthMorphOutputSpec(TraitedSpec):
    output_image = File(desc="The registered output image")
    fixed_to_moving_transform = Either(File, Str, desc="Transform from fixed to moving image")
    moving_to_fixed_transform = Either(File, Str, desc="Transform from moving to fixed image")

class SynthMorph(CommandLine):
    """SynthMorph registration interface for Nipype.
    
    This interface wraps the SynthMorph command line tool for image registration.
    """
    _cmd = 'synthmorph'
    input_spec = SynthMorphInputSpec
    output_spec = SynthMorphOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.output_image:
            outputs['output_image'] = os.path.abspath(self.inputs.output_image)
        if self.inputs.fixed_to_moving_transform:
            outputs['fixed_to_moving_transform'] = os.path.abspath(self.inputs.fixed_to_moving_transform)
        if self.inputs.moving_to_fixed_transform:
            outputs['moving_to_fixed_transform'] = os.path.abspath(self.inputs.moving_to_fixed_transform)
        return outputs