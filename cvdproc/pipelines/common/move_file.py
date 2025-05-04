# a nipype interface to move files

import os
import subprocess
from nipype.interfaces.base import (
    CommandLine, CommandLineInputSpec, TraitedSpec,
    File, traits
)

class MoveFileInputSpec(CommandLineInputSpec):
    source_file = File(exists=True, desc="Source file to move", argstr="%s", position=0)
    destination_file = File(desc="Destination file path", argstr="%s", position=1)

class MoveFileOutputSpec(TraitedSpec):
    moved_file = File(exists=True, desc="Moved output file")

class MoveFileCommandLine(CommandLine):
    _cmd = "mv"
    input_spec = MoveFileInputSpec
    output_spec = MoveFileOutputSpec
    terminal_output = "allatonce"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        
        # The moved file is the destination file
        outputs["moved_file"] = os.path.abspath(self.inputs.destination_file)

        return outputs
    
if __name__ == "__main__":
    # Example usage
    move_file = MoveFileCommandLine()
    move_file.inputs.source_file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/bedpostX_input/data.nii"
    move_file.inputs.destination_file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/data.nii"

    result = move_file.run()
    print(result.outputs.moved_file)