import os
import shutil
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory, Bool, Either

# usage: mri_synthstrip [-h] -i FILE [-o FILE] [-m FILE] [-d FILE] [-g] [-b BORDER] [-t THREADS] [--no-csf]
#                       [--model FILE]

# Robust, universal skull-stripping for brain images of any type.

# optional arguments:
#   -h, --help            show this help message and exit
#   -i FILE, --image FILE
#                         input image to skullstrip
#   -o FILE, --out FILE   save stripped image to file
#   -m FILE, --mask FILE  save binary brain mask to file
#   -d FILE, --sdt FILE   save distance transform to file
#   -g, --gpu             use the GPU
#   -b BORDER, --border BORDER
#                         mask border threshold in mm, defaults to 1
#   -t THREADS, --threads THREADS
#                         PyTorch CPU threads, PyTorch default if unset
#   --no-csf              exclude CSF from brain border
#   --model FILE          alternative model weights

# If you use SynthStrip in your analysis, please cite:
# ----------------------------------------------------
# SynthStrip: Skull-Stripping for Any Brain Image
# A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann
# NeuroImage 206 (2022), 119474
# https://doi.org/10.1016/j.neuroimage.2022.119474

# Website: https://synthstrip.io

class SynthStripInputSpec(CommandLineInputSpec):
    image = Str(mandatory=True, argstr="-i %s", position=0, desc="Input image to skullstrip")
    out_file = Str(argstr="-o %s", position=1, desc="Save stripped image to file")
    mask_file = Str(argstr="-m %s", position=2, desc="Save binary brain mask to file")
    sdt_file = Str(argstr="-d %s", position=3, desc="Save distance transform to file")
    gpu = Bool(False, argstr="-g", desc="Use the GPU")
    border = Int(1, argstr="-b %d", desc="Mask border threshold in mm, defaults to 1")
    threads = Int(0, argstr="-t %d", desc="PyTorch CPU threads, PyTorch default if unset")
    no_csf = Bool(False, argstr="--no-csf", desc="Exclude CSF from brain border")
    model = Str(argstr="--model %s", desc="Alternative model weights")
    four_d = Bool(False, argstr="-4d", desc="Process 4D images (default is 3D)")

class SynthStripOutputSpec(TraitedSpec):
    # all these outputs can be None if not specified
    out_file = Either(File, None, desc='Stripped image file')
    mask_file = Either(File, None, desc='Binary brain mask file')
    sdt_file = Either(File, None, desc='Distance transform file')

class SynthStrip(CommandLine):
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bash', 'freesurfer', 'mri_synthstrip_flexible.sh'))
    _cmd = f'bash {script_path}'
    input_spec = SynthStripInputSpec
    output_spec = SynthStripOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file) if self.inputs.out_file else None
        outputs['mask_file'] = os.path.abspath(self.inputs.mask_file) if self.inputs.mask_file else None
        outputs['sdt_file'] = os.path.abspath(self.inputs.sdt_file) if self.inputs.sdt_file else None
        return outputs