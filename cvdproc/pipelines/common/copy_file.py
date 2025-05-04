import os
import subprocess
from nipype.interfaces.base import (
    CommandLine, CommandLineInputSpec, BaseInterface, BaseTraitedSpec, BaseInterfaceInputSpec, TraitedSpec,
    File, traits
)

from traits.api import Either, Float

class CopyFileInputSpec(CommandLineInputSpec):
    input_file = File(exists=True, desc="Input file", argstr="%s", position=0)
    output_file = File(desc="Output file path", argstr="%s", position=1)

class CopyFileOutputSpec(TraitedSpec):
    output_file = Either(None, File(exists=True), desc="Output file")

class CopyFileCommandLine(CommandLine):
    _cmd = 'cp'
    input_spec = CopyFileInputSpec
    output_spec = CopyFileOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        
        # The output file is the destination file
        outputs["output_file"] = os.path.abspath(self.inputs.output_file) if self.inputs.output_file else None

        return outputs

if __name__ == "__main__":
    # Example usage
    copy_file = CopyFileCommandLine()
    copy_file.inputs.input_file = "/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-02/anat/lst_ai_output/space-flair_seg-lst.nii.gz"
    copy_file.inputs.output_file = "/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-02/anat/space-flair_seg-lst.nii.gz"

    result = copy_file.run()
    #print(result.outputs.output_file)