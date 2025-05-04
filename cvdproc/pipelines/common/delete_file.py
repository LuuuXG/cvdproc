# a nipype interface to delete files

import os
import subprocess
from nipype.interfaces.base import (
    CommandLine, CommandLineInputSpec, TraitedSpec,
    File, traits
)

class DeleteFileInputSpec(CommandLineInputSpec):
    file = File(exists=True, desc="File to delete", argstr="%s", position=0)

class DeleteFileOutputSpec(TraitedSpec):
    deleted_file = File(exists=False, desc="Deleted output file")

class DeleteFileCommandLine(CommandLine):
    _cmd = "rm"
    input_spec = DeleteFileInputSpec
    output_spec = DeleteFileOutputSpec
    terminal_output = "allatonce"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        
        # The deleted file is the input file
        outputs["deleted_file"] = os.path.abspath(self.inputs.file)

        return outputs