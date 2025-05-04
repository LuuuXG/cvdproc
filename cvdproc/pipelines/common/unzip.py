# a nipype interface to unzip files (.nii.gz -> .nii)

import os
import subprocess
from nipype.interfaces.base import (
    CommandLine, CommandLineInputSpec, TraitedSpec,
    File, traits
)

class UnzipInputSpec(CommandLineInputSpec):
    file = File(exists=True, desc="Input file to unzip", argstr="%s", position=0)
    decompress = traits.Bool(desc="Decompress the file", argstr="-d", default_value=True, mandatory=True)
    stdout = traits.Bool(desc="Output to stdout", argstr="-c")
    keep = traits.Bool(desc="Keep the original file", argstr="-k", default_value=True)
    force = traits.Bool(desc="Force overwrite", argstr="-f", default_value=True)

class UnzipOutputSpec(TraitedSpec):
    unzipped_file = File(exists=True, desc="Unzipped output file")

class UnzipCommandLine(CommandLine):
    _cmd = "gzip"
    input_spec = UnzipInputSpec
    output_spec = UnzipOutputSpec
    terminal_output = "allatonce"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        
        outputs["unzipped_file"] = os.path.abspath(self.inputs.file[:-3]) if self.inputs.file.endswith(".gz") else self.inputs.file

        return outputs
    
if __name__ == "__main__":
    # Example usage
    unzip = UnzipCommandLine()
    unzip.inputs.file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/bedpostX_input/data.nii.gz"
    unzip.inputs.decompress = True
    unzip.inputs.keep = True
    result = unzip.run()
    print(result.outputs.unzipped_file)