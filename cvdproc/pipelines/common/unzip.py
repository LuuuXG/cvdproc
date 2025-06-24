# # a nipype interface to unzip files (.nii.gz -> .nii)

# import os
# import subprocess
# from nipype.interfaces.base import (
#     CommandLine, CommandLineInputSpec, TraitedSpec,
#     File, traits, Str
# )

# class UnzipInputSpec(CommandLineInputSpec):
#     file = File(exists=True, desc="Input file to unzip", argstr="%s", position=0)
#     decompress = traits.Bool(desc="Decompress the file", argstr="-d", default_value=True, mandatory=True)
#     stdout = traits.Bool(desc="Output to stdout", argstr="-c")
#     keep = traits.Bool(desc="Keep the original file", argstr="-k", default_value=True)
#     force = traits.Bool(desc="Force overwrite", argstr="-f", default_value=True)

# class UnzipOutputSpec(TraitedSpec):
#     unzipped_file = Str(desc="Unzipped output file")

# class UnzipCommandLine(CommandLine):
#     _cmd = "gzip"
#     input_spec = UnzipInputSpec
#     output_spec = UnzipOutputSpec
#     #terminal_output = "allatonce"

#     def _list_outputs(self):
#         outputs = self.output_spec().get()
        
#         outputs["unzipped_file"] = os.path.abspath(self.inputs.file[:-3]) if self.inputs.file.endswith(".gz") else self.inputs.file

#         return outputs
    
# if __name__ == "__main__":
#     # Example usage
#     unzip = UnzipCommandLine()
#     unzip.inputs.file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/bedpostX_input/data.nii.gz"
#     unzip.inputs.decompress = True
#     unzip.inputs.keep = True
#     result = unzip.run()
#     print(result.outputs.unzipped_file)

import os
import nibabel as nib
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits, Str
)

class GunzipInputSpec(BaseInterfaceInputSpec):
    file = File(exists=True, mandatory=True, desc="Input .nii.gz file")
    out_dir = traits.Directory(desc="Output directory (default: same as input)")
    out_basename = Str(desc="Output file basename (default: same as input, without .gz extension)")
    keep = traits.Bool(True, usedefault=True, desc="Keep the original .gz file")

class GunzipOutputSpec(TraitedSpec):
    unzipped_file = File(desc="Unzipped .nii file")

class GunzipInterface(BaseInterface):
    input_spec = GunzipInputSpec
    output_spec = GunzipOutputSpec

    def _run_interface(self, runtime):
        os.makedirs(self.inputs.out_dir, exist_ok=True)

        in_file = os.path.abspath(self.inputs.file)
        out_dir = os.path.abspath(self.inputs.out_dir) if self.inputs.out_dir else os.path.dirname(in_file)
        out_path = os.path.join(out_dir, os.path.basename(in_file)[:-3])  # remove .gz

        if self.inputs.out_basename:
            out_path = os.path.join(out_dir, self.inputs.out_basename)
        else:
            out_path = os.path.join(out_dir, os.path.basename(in_file)[:-3])

        # Actually unzip using nibabel
        img = nib.load(in_file)
        nib.save(img, out_path)

        if not self.inputs.keep:
            os.remove(in_file)

        self._out_file = out_path
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['unzipped_file'] = os.path.abspath(self._out_file)

        return outputs
