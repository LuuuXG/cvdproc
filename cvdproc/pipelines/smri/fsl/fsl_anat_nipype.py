# fsl_anat

# import os
# from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
# from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError

# class FSLANATInputSpec(CommandLineInputSpec):
#     input_image = File(exists=True, mandatory=True, argstr='-i %s', position=0, desc='Input image to process')
#     output_directory = Str(argstr='-o %s', position=1, desc='Output directory')

# class FSLANATOutputSpec(TraitedSpec):
#     output_directory = Str(desc='Output directory')

# class FSLANAT(CommandLine):
#     _cmd = 'fsl_anat'
#     input_spec = FSLANATInputSpec
#     output_spec = FSLANATOutputSpec
#     terminal_output = 'allatonce'

#     def _list_outputs(self):
#         outputs = self.output_spec().get()
#         # output_directory: -o + '.anat'
#         # if -o was not specified, the output directory will be the [basename of inputimage].anat, and the same directory as the input image
#         if self.inputs.output_directory:
#             outputs['output_directory'] = f"{self.inputs.output_directory}.anat"
#         else:
#             outputs['output_directory'] = f"{os.path.splitext(self.inputs.input_image)[0]}.anat"
#         return outputs
    
import os
import subprocess
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Directory, traits
)

from traits.api import Str

class FSLANATInputSpec(BaseInterfaceInputSpec):
    input_image = File(exists=True, mandatory=True, desc="Input T1-weighted image")
    output_directory = Directory(desc="Output directory (without .anat suffix)")

class FSLANATOutputSpec(TraitedSpec):
    output_directory = Directory(desc="Path to the output .anat directory")
    t1w_to_mni_nonlin_field = Str(desc="Path to the T1w to MNI non-linear field file")
    mni_to_t1w_nonlin_field = Str(desc="Path to the MNI to T1w non-linear field file")

class FSLANAT(BaseInterface):
    input_spec = FSLANATInputSpec
    output_spec = FSLANATOutputSpec

    def _run_interface(self, runtime):
        input_image = self.inputs.input_image
        out_base = self.inputs.output_directory or os.path.splitext(os.path.basename(input_image))[0]
        output_anat_dir = os.path.abspath(f"{out_base}.anat")

        # Check if output exists
        if os.path.isdir(output_anat_dir):
            runtime.stdout = f"Skipping fsl_anat: {output_anat_dir} already exists."
            runtime.returncode = 0
            return runtime

        # Build command
        cmd = ['fsl_anat', '-i', input_image, '-o', out_base]
        result = subprocess.run(cmd, capture_output=True, text=True)

        runtime.returncode = result.returncode
        runtime.stdout = result.stdout
        runtime.stderr = result.stderr
        return runtime

    def _list_outputs(self):
        input_image = self.inputs.input_image
        out_base = self.inputs.output_directory or os.path.splitext(os.path.basename(input_image))[0]
        outputs = self._outputs().get()
        outputs["output_directory"] = os.path.abspath(f"{out_base}.anat")
        outputs["t1w_to_mni_nonlin_field"] = os.path.abspath(os.path.join(outputs["output_directory"], "T1_to_MNI_nonlin_field.nii.gz"))
        outputs["mni_to_t1w_nonlin_field"] = os.path.abspath(os.path.join(outputs["output_directory"], "MNI_to_T1_nonlin_field.nii.gz"))
        return outputs
