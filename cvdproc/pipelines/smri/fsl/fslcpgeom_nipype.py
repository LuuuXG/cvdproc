# fslcpgeom

# Usage: fslcpgeom <source> <destination> [-d]
# -d : don't copy image dimensions

import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError

class FSLcpgeomInputSpec(CommandLineInputSpec):
    source = File(exists=True, mandatory=True, desc="Source file to copy", argstr="%s", position=0)
    destination = File(mandatory=True, desc="Destination file", argstr="%s", position=1)
    dont_copy_dims = Bool(False, desc="Don't copy image dimensions", argstr="-d", position=2)

class FSLcpgeomOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")

class FSLcpgeom(CommandLine):
    input_spec = FSLcpgeomInputSpec
    output_spec = FSLcpgeomOutputSpec
    _cmd = "fslcpgeom"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = self.inputs.destination
        return outputs