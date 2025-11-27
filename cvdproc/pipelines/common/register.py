from nipype import Node, Workflow
from nipype.interfaces import fsl
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Str
import os

#####################################
# Synthmorph Nonlinear Registration #
#####################################

class SynthmorphNonlinearInputSpec(CommandLineInputSpec):
    t1 = File(argstr='-t1 %s', desc='T1-weighted image')
    mni_template = File(exists=True, mandatory=True, argstr='-mni_template %s', desc='MNI template image')
    t1_mni_out = Str(mandatory=True, argstr='-t1_mni_out %s', desc='Output T1 image in MNI space')
    t1_2_mni_warp = Str(mandatory=True, argstr='-t1_2_mni_warp %s', desc='Output warp field from T1 to MNI')
    mni_2_t1_warp = Str(mandatory=True, argstr='-mni_2_t1_warp %s', desc='Output warp field from MNI to T1')
    t1_stripped = File(argstr='-t1_stripped %s', desc='Stripped T1-weighted image')
    register_between_stripped = Bool(False, argstr='-register_between_stripped', desc='If set, indicates that both T1 and MNI template are skull-stripped')
    brain_mask_out = Str(mandatory=False, argstr='-brain_mask_out %s', esc='Output brain mask in T1 space')

class SynthmorphNonlinearOutputSpec(TraitedSpec):
    t1_mni_out = Str(desc='Output T1 image in MNI space')
    t1_2_mni_warp = Str(desc='Output warp field from T1 to MNI')
    mni_2_t1_warp = Str(desc='Output warp field from MNI to T1')

class SynthmorphNonlinear(CommandLine):
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bash', 'freesurfer', 'mri_synthmorph_single.sh'))
    input_spec = SynthmorphNonlinearInputSpec
    output_spec = SynthmorphNonlinearOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['t1_mni_out'] = self.inputs.t1_mni_out
        outputs['t1_2_mni_warp'] = self.inputs.t1_2_mni_warp
        outputs['mni_2_t1_warp'] = self.inputs.mni_2_t1_warp
        return outputs

################
# FSL Register #
################
import os
from nipype.interfaces.base import (
    CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits
)
from traits.api import Str, Int

class ModalityRegistrationInputSpec(CommandLineInputSpec):
    image_target = File(exists=True, desc='Target image', argstr='%s', position=0, mandatory=True)
    image_target_strip = traits.Int(desc='Target image stripped (1) or not (0)', argstr='%d', position=1, mandatory=True)
    image_source = File(exists=True, desc='Source image', argstr='%s', position=2, mandatory=True)
    image_source_strip = traits.Int(desc='Source image stripped (1) or not (0)', argstr='%d', position=3, mandatory=True)
    flirt_direction = traits.Int(desc='FLIRT direction: 0 = -ref use the source, 1 = -ref use the target', argstr='%d', position=4, mandatory=True)
    output_dir = Directory(desc='Output directory', argstr='%s', position=5, mandatory=True)
    registered_image_filename = Str(desc='Registered image filename', argstr='%s', position=6, mandatory=True)
    source_to_target_mat_filename = Str(desc='Source to target transformation matrix filename', argstr='%s', position=7, mandatory=True)
    target_to_source_mat_filename = Str(desc='Target to source transformation matrix filename', argstr='%s', position=8, mandatory=True)
    dof = traits.Int(desc='Degrees of freedom', argstr='%d', position=9, mandatory=True)

class ModalityRegistrationOutputSpec(TraitedSpec):
    output_image = File(desc='Registered source image in target space')
    source_to_target_mat = File(desc='FLIRT transformation matrix')
    target_to_source_mat = File(desc='FLIRT inverse transformation matrix')

class ModalityRegistration(CommandLine):
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bash'))
    _cmd = 'bash ' + os.path.join(script_dir, 'register.sh')

    input_spec = ModalityRegistrationInputSpec
    output_spec = ModalityRegistrationOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()

        out_dir = self.inputs.output_dir
        outputs['output_image'] = os.path.join(out_dir, self.inputs.registered_image_filename)
        outputs['source_to_target_mat'] = os.path.join(out_dir, self.inputs.source_to_target_mat_filename)
        outputs['target_to_source_mat'] = os.path.join(out_dir, self.inputs.target_to_source_mat_filename)

        return outputs

############################
# Apply Warp (mri_convert) #
############################
class MRIConvertApplyWarpInputSpec(CommandLineInputSpec):
    warp_image = File(exists=True, desc='Warp image (e.g., .nii.gz)', argstr='-at %s', mandatory=True)
    input_image = File(exists=True, desc='Input image to be warped', argstr='%s', position=1, mandatory=True)
    output_image = Str(desc='Output warped image', argstr='%s', position=2, mandatory=True)
    interp = Str('interpolate', desc='Interpolation method: <interpolate|weighted|nearest|cubic>', argstr='-rt %s', position=3)

class MRIConvertApplyWarpOutputSpec(TraitedSpec):
    output_image = File(desc='Output warped image')

class MRIConvertApplyWarp(CommandLine):
    input_spec = MRIConvertApplyWarpInputSpec
    output_spec = MRIConvertApplyWarpOutputSpec
    _cmd = 'mri_convert'

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = self.inputs.output_image
        return outputs

#####################################
# 2-step Normalization to MNI space #
#####################################
from nipype import Node, Workflow
from nipype.interfaces import fsl
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, InputMultiPath
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Str, List
import os

# Usage:
#   $0 \
#     --t1w <T1w image> \
#     --t1w_to_mni_warp <T1w->MNI warp .nii.gz/.mgz> \
#     --qsm_to_t1w_affine <QSM->T1w affine .mat> \
#     --output_dir <Output directory> \
#     --input   <in1.nii.gz [in2.nii.gz ...]> \
#     --output1 <out1_T1w.nii.gz [out2_T1w.nii.gz ...]> \
#     --output2 <out1_MNI.nii.gz [out2_MNI.nii.gz ...]>

class TwoStepNormalizationInputSpec(CommandLineInputSpec):
    struct = File(exists=True, desc="Struct image", mandatory=True, argstr="--t1w %s")
    struct_to_mni_warp = File(exists=True, desc="Struct to MNI warp file (.nii.gz/.mgz)", mandatory=True, argstr="--t1w_to_mni_warp %s")
    source_to_struct_affine = File(exists=True, desc="source to struct affine matrix file (.mat)", mandatory=True, argstr="--qsm_to_t1w_affine %s")
    output_dir = Str(desc="Output directory", mandatory=True, argstr="--output_dir %s")
    # input = List(Str(exists=True), desc="Input QSM files", mandatory=True, argstr="--input %s...")
    # output1 = List(Str(), desc="Output files in T1w space", argstr="--output1 %s...")
    # output2 = List(Str(), desc="Output files in MNI space", argstr="--output2 %s...")
    input = InputMultiPath(File(exists=True), argstr="--input %s", sep=" ", mandatory=True)
    output_struct = List(Str, argstr="--output1 %s", sep=" ", mandatory=True)
    output_mni = List(Str, argstr="--output2 %s", sep=" ", mandatory=True)


class TwoStepNormalizationOutputSpec(TraitedSpec):
    outputs_in_struct = List(Str(), desc="Outputs registered to Struct space")
    outputs_in_mni = List(Str(), desc="Outputs registered to MNI space")

class TwoStepNormalization(CommandLine):
    _cmd = 'bash ' + os.path.join(os.path.dirname(__file__), '..', 'bash', 'qsm', 'qsm_register2.sh')
    input_spec = TwoStepNormalizationInputSpec
    output_spec = TwoStepNormalizationOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs_in_t1w = []
        outputs_in_mni = []

        for out in self.inputs.output1:
            outputs_in_t1w.append(os.path.abspath(os.path.join(self.inputs.output_dir, os.path.basename(out))))
        for out in self.inputs.output2:
            outputs_in_mni.append(os.path.abspath(os.path.join(self.inputs.output_dir, os.path.basename(out))))
        
        outputs['outputs_in_struct'] = outputs_in_t1w
        outputs['outputs_in_mni'] = outputs_in_mni
        return outputs

###############
# tkregister2 #
###############

# tkregister2 --mov "${FS_OUTPUT}/mri/orig.mgz" \
#             --targ "${FS_OUTPUT}/mri/rawavg.mgz" \
#             --regheader \
#             --reg junk \
#             --fslregout "${OUTPUT_DIR}/freesurfer2struct.mat" \
#             --noedit

class Tkregister2fs2t1wInputSpec(CommandLineInputSpec):
    fs_subjects_dir = Directory(exists=True, desc="Freesurfer SUBJECTS_DIR", argstr="%s", mandatory=True, position=0)
    fs_subject_id = Str(desc="Freesurfer subject ID", argstr="%s", mandatory=True, position=1)
    output_matrix = Str(desc="Output matrix filename (.mat)", argstr="%s", mandatory=True, position=2)
    output_inverse_matrix = Str(desc="Output inverse matrix filename (.mat)", argstr="%s", mandatory=True, position=3)

class Tkregister2fs2t1wOutputSpec(TraitedSpec):
    output_matrix = File(desc="Output matrix filename (.mat)")
    output_inverse_matrix = File(desc="Output inverse matrix filename (.mat)")

class Tkregister2fs2t1w(CommandLine):
    _cmd = 'bash ' + os.path.join(os.path.dirname(__file__), '..', 'bash', 'freesurfer', 'tkregister2fs2t1w.sh')
    input_spec = Tkregister2fs2t1wInputSpec
    output_spec = Tkregister2fs2t1wOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_matrix'] = os.path.abspath(self.inputs.output_matrix)
        outputs['output_inverse_matrix'] = os.path.abspath(self.inputs.output_inverse_matrix)
        return outputs