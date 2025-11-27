import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError
from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import isdefined

# Usage: fsl_deface <input> <output>

#        Tool to deface a structural T1w image.

# Options:

#   -k                     apply the defacing to the cropped image instead of the original image
#   -d <defacing_mask>     filename to save the defacing mask;
#   -n <cropped_struc>     filename to save the new cropped struct;
#   -m13 <orig_2_std>      filename to save affine matrix from original struct to std;
#   -m12 <orig_2_cropped>  filename to save affine matrix from original struct to cropped struct;
#   -m23 <cropped_2_std>   filename to save affine matrix from cropped struct to std;
#   -nud <x y z>           Shift, in mm, x-, y- and z-directions, to shift face mask by;
#                          (These matrices will only work if the input has been previously reoriented to std)
#   -f <f>                 fractional intensity for bet (0->1); default=0.5;
#   -B                     Bias-correct the input image (with fast);
#   -c <x y z>             centre-of-gravity for bet (voxels, not mm);
#   -p <images_baseName>   generate 2 pngs to show how the defacing worked for QC purposes

class FSLDefaceInputSpec(CommandLineInputSpec):
    input_image = File(exists=True, mandatory=True, argstr='%s', position=0, desc='Input image to deface')
    output_image = Str(mandatory=True, argstr='%s', position=1, desc='Output defaced image')
    keep_cropped = Bool(argstr='-k', desc='Apply the defacing to the cropped image instead of the original image')
    defacing_mask = Str(argstr='-d %s', desc='Filename to save the defacing mask')
    cropped_struct = Str(argstr='-n %s', desc='Filename to save the new cropped struct')
    orig_to_std_matrix = Str(argstr='-m13 %s', desc='Filename to save affine matrix from original struct to std')
    orig_to_cropped_matrix = Str(argstr='-m12 %s', desc='Filename to save affine matrix from original struct to cropped struct')
    cropped_to_std_matrix = Str(argstr='-m23 %s', desc='Filename to save affine matrix from cropped struct to std')
    nud_shift = List(Int, argstr='-nud %s', desc='Shift, in mm, x-, y- and z-directions, to shift face mask by', minlen=3, maxlen=3)
    bet_f = Float(argstr='-f %f', desc='Fractional intensity for bet (0->1); default=0.5')
    bias_correct = Bool(argstr='-B', desc='Bias-correct the input image (with fast)')
    cog = List(Int, argstr='-c %s', desc='Centre-of-gravity for bet (voxels, not mm)', minlen=3, maxlen=3)
    png_base_name = Str(argstr='-p %s', desc='Generate 2 pngs to show how the defacing worked for QC purposes')

class FSLDefaceOutputSpec(TraitedSpec):
    output_image = Str(desc='Output defaced image')
    defacing_mask = Str(desc='Path to the defacing mask file')
    cropped_struct = Str(desc='Path to the cropped struct file')
    orig_to_std_matrix = Str(desc='Path to the affine matrix from original struct to std')
    orig_to_cropped_matrix = Str(desc='Path to the affine matrix from original struct to cropped struct')
    cropped_to_std_matrix = Str(desc='Path to the affine matrix from cropped struct to std')
    png_files = List(Str, desc='List of generated PNG files for QC purposes')

class FSLDeface(CommandLine):
    _cmd = "fsl_deface"
    input_spec = FSLDefaceInputSpec
    output_spec = FSLDefaceOutputSpec
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.output_image):
            outputs['output_image'] = os.path.abspath(self.inputs.output_image)
        if isdefined(self.inputs.defacing_mask):
            outputs['defacing_mask'] = os.path.abspath(self.inputs.defacing_mask)
        if isdefined(self.inputs.cropped_struct):
            outputs['cropped_struct'] = os.path.abspath(self.inputs.cropped_struct)
        if isdefined(self.inputs.orig_to_std_matrix):
            outputs['orig_to_std_matrix'] = os.path.abspath(self.inputs.orig_to_std_matrix)
        if isdefined(self.inputs.orig_to_cropped_matrix):
            outputs['orig_to_cropped_matrix'] = os.path.abspath(self.inputs.orig_to_cropped_matrix)
        if isdefined(self.inputs.cropped_to_std_matrix):
            outputs['cropped_to_std_matrix'] = os.path.abspath(self.inputs.cropped_to_std_matrix)
        if isdefined(self.inputs.png_base_name):
            png1 = os.path.abspath(f"{self.inputs.png_base_name}_1.png")
            png2 = os.path.abspath(f"{self.inputs.png_base_name}_2.png")
            outputs['png_files'] = [png1, png2]
        else:
            outputs['png_files'] = []
        return outputs

if __name__ == "__main__":
    fsl_deface = FSLDeface()
    fsl_deface.inputs.input_image = '/mnt/e/Neuroimage/suit/sub-TAOHC0263_ses-baseline_acq-highres_T1w.nii'
    fsl_deface.inputs.output_image = '/mnt/e/Neuroimage/suit/sub-TAOHC0263_ses-baseline_acq-highres_T1w_deface.nii.gz'
    
    print("Command to be executed:")
    print(fsl_deface.cmdline)
    
    # To actually run the command, uncomment the following line:
    result = fsl_deface.run()