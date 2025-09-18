import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError, Either

####################
# Make BIANCA mask #
####################
class MakeBIANCAMaskInputSpec(CommandLineInputSpec):
    fsl_anat_output = Directory(exists=True, desc="Output directory from fsl_anat", mandatory=True, argstr="%s", position=0)
    bianca_mask_name = Str(mandatory=True, desc="Name of the BIANCA mask output path", argstr="%s", position=1)

class MakeBIANCAMaskOutputSpec(TraitedSpec):
    bianca_mask = Str(desc="Path to the BIANCA WM mask")

class MakeBIANCAMask(CommandLine):
    input_spec = MakeBIANCAMaskInputSpec
    output_spec = MakeBIANCAMaskOutputSpec
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bash", "fsl", "make_bianca_mask.sh"))

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['bianca_mask'] = os.path.abspath(self.inputs.bianca_mask_name)
        return outputs