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
    dwi_img = File(desc="Path to the DWI image", mandatory=True, argstr="%s", position=0)
    dwi_bvec = File(desc="Path to the DWI bvec file", mandatory=True, argstr="%s", position=1)
    dwi_bval = File(desc="Path to the DWI bval file", mandatory=True, argstr="%s", position=2)
    output_dir = Directory(desc="Output directory for the denoised DWI image", mandatory=True, argstr="%s", position=3)

class MrtrixDenoiseOutputSpec(TraitedSpec):
    output_dwi_img = File(desc="Path to the denoised DWI image")
    output_dwi_bvec = File(desc="Path to the denoised DWI bvec file")
    output_dwi_bval = File(desc="Path to the denoised DWI bval file")

class MrtrixDenoise(CommandLine):
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bash", "mrtrix3", "mrtrix_denoise.sh"))
    input_spec = MrtrixDenoiseInputSpec
    output_spec = MrtrixDenoiseOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_dwi_img"] = os.path.abspath(os.path.join(self.inputs.output_dir, "dwi_denoise.nii.gz"))
        outputs["output_dwi_bvec"] = os.path.abspath(os.path.join(self.inputs.output_dir, "dwi_denoise.bvec"))
        outputs["output_dwi_bval"] = os.path.abspath(os.path.join(self.inputs.output_dir, "dwi_denoise.bval"))
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