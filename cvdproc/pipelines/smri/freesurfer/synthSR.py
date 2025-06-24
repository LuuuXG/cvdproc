import os
import shutil
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory, Bool

# usage: mri_synthsr [-h] [--i I] [--o O] [--ct] [--disable_sharpening] [--disable_flipping] [--lowfield] [--v1]
#                    [--threads THREADS] [--cpu] [--model MODEL]

# Implementation of SynthSR that generates a synthetic 1mm MP-RAGE from a scan of any contrast and resolution

# optional arguments:
#   -h, --help            show this help message and exit
#   --i I                 Image(s) to super-resolve. Can be a path to an image or to a folder.
#   --o O                 Output(s), i.e. synthetic 1mm MP-RAGE(s). Must be a folder if --i designates a folder.
#   --ct                  (optional) Use this flag for CT scans in Hounsfield scale, it clips intensities to [0,80].
#   --disable_sharpening  (optional) Use this flag to disable unsharp masking.
#   --disable_flipping    (optional) Use this flag to disable flipping augmentation at test time.
#   --lowfield            (optional) Use model for low-field scans (e.g., acquired with Hyperfine's Swoop scanner).
#   --v1                  (optional) Use version 1 model from July 2021.
#   --threads THREADS     (optional) Number of cores to be used. Default is 1.
#   --cpu                 (optional) Enforce running with CPU rather than GPU.
#   --model MODEL         (optional) Use a different model file.

class SynthSRInputSpec(CommandLineInputSpec):
    input = Str(mandatory=True, argstr="--i %s", position=0, desc="Image(s) to super-resolve. Can be a path to an image or to a folder.")
    output = Str(mandatory=True, argstr="--o %s", position=1, desc="Output(s), i.e. synthetic 1mm MP-RAGE(s). Must be a folder if --i designates a folder.")
    ct = Bool(False, usedefault=True, argstr="--ct", desc="Use this flag for CT scans in Hounsfield scale, it clips intensities to [0,80].")
    disable_sharpening = Bool(False, usedefault=True, argstr="--disable_sharpening", desc="Use this flag to disable unsharp masking.")
    disable_flipping = Bool(False, usedefault=True, argstr="--disable_flipping", desc="Use this flag to disable flipping augmentation at test time.")
    lowfield = Bool(False, usedefault=True, argstr="--lowfield", desc="Use model for low-field scans (e.g., acquired with Hyperfine's Swoop scanner).")
    v1 = Bool(False, usedefault=True, argstr="--v1", desc="Use version 1 model from July 2021.")
    threads = Int(1, usedefault=True, argstr="--threads %d", desc="Number of cores to be used. Default is 1.")
    cpu = Bool(False, usedefault=True, argstr="--cpu", desc="Enforce running with CPU rather than GPU.")
    model = Str(argstr="--model %s", desc="Use a different model file. If not specified, the default model will be used.")

class SynthSROutputSpec(TraitedSpec):
    output = Str(desc='Output directory containing the synthetic 1mm MP-RAGE image(s)')

class SynthSR(CommandLine):
    _cmd = 'mri_synthsr'
    input_spec = SynthSRInputSpec
    output_spec = SynthSROutputSpec
    #terminal_output = 'allatonce'
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output'] = self.inputs.output
        return outputs

if __name__ == '__main__':
    # Example usage
    synthsr = SynthSR()
    synthsr.inputs.input = '/mnt/f/BIDS/SVD_BIDS/sub-SVD0001/ses-01/anat/sub-SVD0001_ses-01_acq-lowres_T1w.nii.gz'
    synthsr.inputs.output = '/mnt/f/BIDS/SVD_BIDS/sub-SVD0001/ses-01/anat/sub-SVD0001_ses-01_acq-synthsr_T1w.nii.gz'
    result = synthsr.run()
    print(result.outputs.output)  # Output directory containing the synthetic image(s)