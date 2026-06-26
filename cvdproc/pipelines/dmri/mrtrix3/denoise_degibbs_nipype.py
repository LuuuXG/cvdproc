import os
from nipype.interfaces.base import TraitedSpec, File, Directory, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory

class DenoiseDegibbsInputSpec(CommandLineInputSpec):
    dwi_img = File(desc="Path to the DWI image", mandatory=True, argstr="%s", position=0)
    dwi_bvec = File(desc="Path to the DWI bvec file", mandatory=True, argstr="%s", position=1)
    dwi_bval = File(desc="Path to the DWI bval file", mandatory=True, argstr="%s", position=2)
    output_dir = Directory(desc="Output directory for the denoised and de-gibbsed DWI image", mandatory=True, argstr="%s", position=3)

class DenoiseDegibbsOutputSpec(TraitedSpec):
    output_dwi_img = File(desc="Path to the denoised and de-gibbsed DWI image")
    output_dwi_bvec = File(desc="Path to the denoised and de-gibbsed DWI bvec file")
    output_dwi_bval = File(desc="Path to the denoised and de-gibbsed DWI bval file")

class DenoiseDegibbs(CommandLine):
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bash", "mrtrix3", "mrtrix_denoise_degibbs.sh"))
    input_spec = DenoiseDegibbsInputSpec
    output_spec = DenoiseDegibbsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_dwi_img"] = os.path.abspath(os.path.join(self.inputs.output_dir, "dwi_denoise_degibbs.nii.gz"))
        outputs["output_dwi_bvec"] = os.path.abspath(os.path.join(self.inputs.output_dir, "dwi_denoise_degibbs.bvec"))
        outputs["output_dwi_bval"] = os.path.abspath(os.path.join(self.inputs.output_dir, "dwi_denoise_degibbs.bval"))
        return outputs

# Separate to Denoise and Degibbs
class MrtrixDenoiseInputSpec(CommandLineInputSpec):
    dwi_img = File(
        desc="Path to the DWI image",
        mandatory=True,
        exists=True,
        argstr="%s",
        position=0,
    )

    output_dwi = File(
        desc="Path to the denoised DWI image",
        mandatory=True,
        argstr="%s",
        position=1,
    )


class MrtrixDenoiseOutputSpec(TraitedSpec):
    output_dwi_img = File(desc="Path to the denoised DWI image", exists=True)


class MrtrixDenoise(CommandLine):
    _cmd = "dwidenoise -force"

    input_spec = MrtrixDenoiseInputSpec
    output_spec = MrtrixDenoiseOutputSpec

    def _run_interface(self, runtime):
        output_file = os.path.abspath(self.inputs.output_dwi)

        if os.path.exists(output_file):
            print(f"Output already exists, skipping dwidenoise: {output_file}")
            return runtime

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_dwi_img"] = os.path.abspath(self.inputs.output_dwi)
        return outputs

class MrtrixDegibbsInputSpec(CommandLineInputSpec):
    dwi_img = File(desc="Path to the DWI image", mandatory=True, argstr="%s", position=0)
    dwi_bvec = File(desc="Path to the DWI bvec file", mandatory=True, argstr="%s", position=1)
    dwi_bval = File(desc="Path to the DWI bval file", mandatory=True, argstr="%s", position=2)
    output_dir = Directory(desc="Output directory for the de-gibbsed DWI image", mandatory=True, argstr="%s", position=3)

class MrtrixDegibbsOutputSpec(TraitedSpec):
    output_dwi_img = File(desc="Path to the de-gibbsed DWI image")
    output_dwi_bvec = File(desc="Path to the de-gibbsed DWI bvec file")
    output_dwi_bval = File(desc="Path to the de-gibbsed DWI bval file")
    
class MrtrixDegibbs(CommandLine):
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bash", "mrtrix3", "mrtrix_degibbs.sh"))
    input_spec = MrtrixDegibbsInputSpec
    output_spec = MrtrixDegibbsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_dwi_img"] = os.path.abspath(os.path.join(self.inputs.output_dir, "dwi_degibbs.nii.gz"))
        outputs["output_dwi_bvec"] = os.path.abspath(os.path.join(self.inputs.output_dir, "dwi_degibbs.bvec"))
        outputs["output_dwi_bval"] = os.path.abspath(os.path.join(self.inputs.output_dir, "dwi_degibbs.bval"))
        return outputs
    