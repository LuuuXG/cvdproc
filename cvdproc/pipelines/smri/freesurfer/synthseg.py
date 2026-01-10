import os
import shutil
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory, Bool, Either

from cvdproc.config.paths import get_package_path

# usage: mri_synthseg [-h] [--i I] [--o O] [--parc] [--robust] [--fast] [--ct] [--vol VOL] [--qc QC] [--post POST]
#                     [--resample RESAMPLE] [--crop CROP [CROP ...]] [--autocrop] [--threads THREADS] [--cpu] [--v1]
#                     [--keepgeom] [--addctab] [--noaddctab] [--photo PHOTO] [--model MODEL]

# SynthSeg

# optional arguments:
#   -h, --help            show this help message and exit
#   --i I                 Image(s) to segment. Can be a path to an image or to a folder.
#   --o O                 Segmentation output(s). Must be a folder if --i designates a folder.
#   --parc                (optional) Whether to perform cortex parcellation.
#   --robust              (optional) Whether to use robust predictions (slower).
#   --fast                (optional) Bypass some processing for faster prediction.
#   --ct                  (optional) Clip CT scans in Hounsfield scale to [0, 80]
#   --vol VOL             (optional) Output CSV file with volumes for all structures and subjects.
#   --qc QC               (optional) Output CSV file with qc scores for all subjects.
#   --post POST           (optional) Posteriors output(s). Must be a folder if --i designates a folder.
#   --resample RESAMPLE   (optional) Resampled image(s). Must be a folder if --i is a folder.
#   --crop CROP [CROP ...]
#                         (optional) Only analyse an image patch of the given size.
#   --autocrop            (optional) Ignore background voxels in FOV.
#   --threads THREADS     (optional) Number of cores to be used. Default is 1.
#   --cpu                 (optional) Enforce running with CPU rather than GPU.
#   --v1                  (optional) Use SynthSeg 1.0 (updated 25/06/22).
#   --keepgeom            Force output geometry to be the same as input
#   --addctab             Embed colortable into seg output
#   --noaddctab           Do not embed colortable into seg output
#   --photo PHOTO         (optional) Photo-SynthSeg: segment 3D reconstructed stack of coronal dissection photos of the
#                         cerebrum; must be left, right, or both
#   --model MODEL         (optional) Provide an alternative model file

class SynthSegInputSpec(CommandLineInputSpec):
    image = Str(mandatory=True, argstr="--i %s", position=0, desc="Image(s) to segment. Can be a path to an image or to a folder.")
    out = Str(mandatory=True, argstr="--o %s", position=1, desc="Segmentation output(s). Must be a folder if --i designates a folder.")
    parc = Bool(argstr="--parc", desc="Whether to perform cortex parcellation.")
    robust = Bool(argstr="--robust", desc="Whether to use robust predictions (slower).")
    fast = Bool(argstr="--fast", desc="Bypass some processing for faster prediction.")
    ct = Bool(argstr="--ct", desc="Clip CT scans in Hounsfield scale to [0, 80]")
    vol = Str(argstr="--vol %s", desc="Output CSV file with volumes for all structures and subjects.")
    qc = Str(argstr="--qc %s", desc="Output CSV file with qc scores for all subjects.")
    post_dir = Str(argstr="--post %s", desc="Posteriors output(s). Must be a folder if --i designates a folder.")
    resample_dir = Str(argstr="--resample %s", desc="Resampled image(s). Must be a folder if --i is a folder.")
    crop = Int(argstr="--crop %d", nargs='*', desc="Only analyse an image patch of the given size.")
    autocrop = Bool(argstr="--autocrop", desc="Ignore background voxels in FOV.")
    threads = Int(1, argstr="--threads %d", desc="Number of cores to be used. Default is 1.")
    cpu = Bool( argstr="--cpu", desc="Enforce running with CPU rather than GPU.")
    v1 = Bool(argstr="--v1", desc="Use SynthSeg 1.0 (updated 25/06/22).")
    keepgeom = Bool(argstr="--keepgeom", desc="Force output geometry to be the same as input")
    addctab = Bool(argstr="--addctab", desc="Embed colortable into seg output")
    noaddctab = Bool(argstr="--noaddctab", desc="Do not embed colortable into seg output")
    photo = Str(argstr="--photo %s", desc="Photo-SynthSeg: segment 3D reconstructed stack of coronal dissection photos of the cerebrum; must be left, right, or both")
    model = Str(argstr="--model %s", desc="Provide an alternative model file")

class SynthSegOutputSpec(TraitedSpec):
    out = Str(desc="Segmentation outputs (single file or a directory).")
    vol = Either(File, None, desc="CSV file with volumes for all structures and subjects.")
    qc = Either(File, None, desc="CSV file with qc scores for all subjects.")

class SynthSeg(CommandLine):
    input_spec = SynthSegInputSpec
    output_spec = SynthSegOutputSpec
    _cmd = 'mri_synthseg'
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out'] = os.path.abspath(self.inputs.out)
        outputs['vol'] = os.path.abspath(self.inputs.vol) if self.inputs.vol else None
        outputs['qc'] = os.path.abspath(self.inputs.qc) if self.inputs.qc else None

        return outputs

class SynthSegPostProcessInputSpec(CommandLineInputSpec):
    synthseg_input = Str(mandatory=True, desc="Input SynthSeg segmentation file.", argstr="%s", position=0)
    wm_output = Str(mandatory=True, desc="Output white matter mask file.", argstr="%s", position=1)

class SynthSegPostProcessOutputSpec(TraitedSpec):
    wm_output = Str(desc="Output white matter mask file.")

class SynthSegPostProcess(CommandLine):
    input_spec = SynthSegPostProcessInputSpec
    output_spec = SynthSegPostProcessOutputSpec
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'freesurfer', 'mri_synthseg_post.sh')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['wm_output'] = os.path.abspath(self.inputs.wm_output)
        return outputs