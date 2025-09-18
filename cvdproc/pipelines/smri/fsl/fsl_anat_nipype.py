# fsl_anat

# Usage: fsl_anat [options] -i <structural image>
#        fsl_anat [options] -d <existing anat directory>

# Arguments (You may specify one or more of):
#   -i <strucural image>         filename of input image (for one image only)
#   -d <anat dir>                directory name for existing .anat directory where this script will be run in place
#   -o <output directory>        basename of directory for output (default is input image basename followed by .anat)
#   --clobber                    if .anat directory exist (as specified by -o or default from -i) then delete it and make a new one
#   --strongbias                 used for images with very strong bias fields
#   --weakbias                   used for images with smoother, more typical, bias fields (default setting)
#   --noreorient                 turn off step that does reorientation 2 standard (fslreorient2std)
#   --nocrop                     turn off step that does automated cropping (robustfov)
#   --nobias                     turn off steps that do bias field correction (via FAST)
#   --noreg                      turn off steps that do registration to standard (FLIRT and FNIRT)
#   --nononlinreg                turn off step that does non-linear registration (FNIRT)
#   --noseg                      turn off step that does tissue-type segmentation (FAST)
#   --nosubcortseg               turn off step that does sub-cortical segmentation (FIRST)
#   -s <value>                   specify the value for bias field smoothing (the -l option in FAST)
#   -t <type>                    specify the type of image (choose one of T1 T2 PD - default is T1)
#   --nosearch                   specify that linear registration uses the -nosearch option (FLIRT)
#   --betfparam                  specify f parameter for BET (only used if not running non-linear reg and also wanting brain extraction done)
#   --nocleanup                  do not remove intermediate files

import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError

class FSLANATInputSpec(CommandLineInputSpec):
    input_image = File(exists=True, mandatory=True, argstr='-i %s', position=0, desc='Input image to process')
    anat_directory = Directory(exists=True, argstr='-d %s', position=0, desc='Existing anat directory to run in place')
    output_directory = Str(argstr='-o %s', position=1, desc='Output directory')
    clobber = Bool(argstr='--clobber', desc='Delete existing .anat directory if it exists')
    strong_bias = Bool(argstr='--strongbias', desc='Use for images with very strong bias fields')
    weak_bias = Bool(argstr='--weakbias', desc='Use for images with smoother, more typical, bias fields (default setting)')
    no_reorient = Bool(argstr='--noreorient', desc='Turn off reorientation to standard space')
    no_crop = Bool(argstr='--nocrop', desc='Turn off automated cropping')
    no_bias = Bool(argstr='--nobias', desc='Turn off bias field correction')
    no_reg = Bool(argstr='--noreg', desc='Turn off registration to standard space')
    no_nonlin_reg = Bool(argstr='--nononlinreg', desc='Turn off non-linear registration')
    no_seg = Bool(argstr='--noseg', desc='Turn off tissue-type segmentation')
    no_subcort_seg = Bool(argstr='--nosubcortseg', desc='Turn off sub-cortical segmentation')
    bias_smoothing = Float(argstr='-s %f', desc='Specify the value for bias field smoothing (the -l option in FAST)')
    image_type = Enum('T1', 'T2', 'PD', argstr='-t %s', desc='Specify the type of image (default is T1)')
    no_search = Bool(argstr='--nosearch', desc='Specify that linear registration uses the -nosearch option (FLIRT)')
    betf_param = Float(argstr='--betfparam %f', desc='Specify f parameter for BET (only used if not running non-linear reg and also wanting brain extraction done)')
    no_cleanup = Bool(argstr='--nocleanup', desc='Do not remove intermediate files')

class FSLANATOutputSpec(TraitedSpec):
    output_directory = Str(desc='Output directory')
    t1w_to_mni_nonlin_field = File(desc='Path to the T1w to MNI non-linear field file')
    mni_to_t1w_nonlin_field = File(desc='Path to the MNI to T1w non-linear field file')

class FSLANAT(CommandLine):
    _cmd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bash", "fsl", "fsl_anat_custom.sh"))
    input_spec = FSLANATInputSpec
    output_spec = FSLANATOutputSpec
    #terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        # output_directory: -o + '.anat'
        # if -o was not specified, the output directory will be the [basename of inputimage].anat, and the same directory as the input image
        input_image = self.inputs.input_image
        out_base = self.inputs.output_directory or os.path.splitext(os.path.basename(input_image))[0]
        outputs["output_directory"] = os.path.abspath(f"{out_base}.anat")
        
        outputs["t1w_to_mni_nonlin_field"] = os.path.abspath(os.path.join(outputs["output_directory"], "T1_to_MNI_nonlin_field.nii.gz"))
        outputs["mni_to_t1w_nonlin_field"] = os.path.abspath(os.path.join(outputs["output_directory"], "MNI_to_T1_nonlin_field.nii.gz"))
        return outputs