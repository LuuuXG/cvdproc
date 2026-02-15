from nipype import Node, Workflow
from nipype.interfaces import fsl
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Str, Either
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
    t1_stripped_out = Str(argstr='-t1_stripped_out %s', desc='Output stripped T1-weighted image')
    register_between_stripped = Bool(False, argstr='-register_between_stripped', desc='If set, indicates that both T1 and MNI template are skull-stripped')
    brain_mask_out = Str(mandatory=False, argstr='-brain_mask_out %s', esc='Output brain mask in T1 space')

class SynthmorphNonlinearOutputSpec(TraitedSpec):
    t1_mni_out = Str(desc='Output T1 image in MNI space')
    t1_2_mni_warp = Str(desc='Output warp field from T1 to MNI')
    mni_2_t1_warp = Str(desc='Output warp field from MNI to T1')
    t1_stripped_out = Either(Str, None, desc='Output stripped T1-weighted image')

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
        outputs['t1_stripped_out'] = self.inputs.t1_stripped_out if self.inputs.t1_stripped_out else None
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

###############
# mri_vol2vol #
###############

# mri_vol2vol \
#   --mov /mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/lesion_mask/sub-SSI0114/ses-F1/sub-SSI0114_ses-F1_space-T1w_desc-RSSI_mask.nii.gz \
#   --targ /mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/freesurfer/sub-SSI0114/ses-F1/mri/orig.mgz \
#   --o /mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/lesion_mask/sub-SSI0114/ses-F1/sub-SSI0114_ses-F1_space-fs_desc-RSSI_mask.nii.gz \
#   --fsl /mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/xfm/sub-SSI0114/ses-F1/sub-SSI0114_ses-F1_from-T1w_to-fs_xfm.mat \
#   --nearest

# mri_vol2vol

#   --mov  movvol       : input (or output template with --inv)
#   --targ targvol      : output template (or input with --inv)
#   --o    outvol       : output volume
#   --disp dispvol      : displacement volume
#   --downsample N1 N2 N3 : downsample factor (eg, 2) (do not include a targ or regsitration)
#          sets --fill-average, --fill-upsample 2, and --regheader

#   --reg  register.dat : tkRAS-to-tkRAS matrix   (tkregister2 format)
#   --lta  register.lta : Linear Transform Array (usually only 1 transform)
#   --lta-inv  register.lta : LTA, invert (may not be the same as --lta --inv with --fstal)
#   --fsl  register.fsl : fslRAS-to-fslRAS matrix (FSL format)
#   --xfm  register.xfm : ScannerRAS-to-ScannerRAS matrix (MNI format)
#   --regheader         : ScannerRAS-to-ScannerRAS matrix = identity
#   --mni152reg         : target MNI152 space (need FSL installed)
#   --s subject         : set matrix = identity and use subject for any templates

#   --inv               : sample from targ to mov

#   --tal               : map to a sub FOV of MNI305 (with --reg only)
#   --talres resolution : set voxel size 1mm or 2mm (def is 1)
#   --talxfm xfmfile    : default is talairach.xfm (looks in mri/transforms)

#   --m3z morph    : non-linear morph encoded in the m3z format
#   --noDefM3zPath : flag indicating that the code should not be looking for
#        the non-linear m3z morph in the default location (subj/mri/transforms), but should use
#        the morph name as is
#   --inv-morph    : compute and use the inverse of the m3z morph

#   --fstarg <vol>      : optionally use vol from subject in --reg as target. default is orig.mgz
#   --crop scale        : crop and change voxel size
#   --slice-crop sS sE  : crop output slices to be within sS and sE
#   --slice-reverse     : reverse order of slices, update vox2ras
#   --slice-bias alpha  : apply half-cosine bias field

#   --trilin            : trilinear interpolation (default)
#   --nearest           : nearest neighbor interpolation
#   --cubic             : cubic B-Spline interpolation
#   --interp interptype : interpolation cubic, trilin, nearest (def is trilin)
#   --fill-average      : compute mean of all source voxels in a given target voxel
#   --fill-conserve     : compute sum  of all source voxels in a given target voxel
#   --fill-upsample USF : source upsampling factor for --fill-{avg,cons} (default is 2)

#   --mul mulval   : multiply output by mulval

#   --vsm vsmvol <pedir> : Apply a voxel shift map. pedir: +/-1=+/-x, +/-2=+/-y, +/-3=+/-z (default +2)
#   --vsm-pedir pedir : set pedir +/-1=+/-x, +/-2=+/-y, +/-3=+/-z (default +2)

#   --precision precisionid : output precision (def is float)
#   --keep-precision  : set output precision to that of input
#   --kernel            : save the trilinear interpolation kernel instead
#    --copy-ctab : setenv FS_COPY_HEADER_CTAB to copy any ctab in the mov header
#    --vg-thresh thresh : set threshold for comparing vol geom

#   --gcam mov srclta gcam dstlta vsm interp out
#      srclta, gcam, or vsm can be set to 0 to indicate identity (not regheader)
#      if dstlta is 0, then uses gcam atlas geometry as output target
#      direction is automatically determined from srclta and dstlta
#      interp 0=nearest, 1=trilin, 5=cubicbspline
#      vsm pedir can be set with --vsm-pedir
#      DestVol -> dstLTA -> CVSVol -> gcam -> AnatVol -> srcLTA -> B0UnwarpedVol -> VSM -> MovVol (b0Warped)

#   --spm-warp mov movlta warp interp output
#      mov is the input to be mapped
#      movlta maps mov to the vbm input space (use 0 to ignore)
#        if movlta=0, then input is anything that shares a RAS space with the VBM input
#      warp is typically y_rinput.nii
#      interp 0=nearest, 1=trilin

#   --map-point a b c incoords lta outcoords outfile : stand-alone option to map a point to another space
#      coords: 1=tkras, 2=scannerras, 3=vox; outfile can be nofile
#   --map-point-inv-lta a b c incoords lta outcoords outfile
#       same as --map-point but inverts the lta


#   --no-resample : do not resample, just change vox2ras matrix
#   --no-resample-scale : do not resample, just change vox2ras matrix, using scale=voxsize

#   --rot   Ax Ay Az : rotation angles (deg) to apply to reg matrix
#   --trans Tx Ty Tz : translation (mm) to apply to reg matrix
#   --shear Sxy Sxz Syz : xz is in-plane
#   --reg-final regfinal.dat : final reg after rot and trans (but not inv)
#   --ctab ctabfile : embed colortable into output (note: this will override any embedded ctab)

#   --synth : replace input with white gaussian noise
#   --seed seed : seed for synth (def is to set from time of day)

#   --save-reg : write out output volume registration matrix

#   --help : go ahead, make my day
#   --debug
#   --version

class MRIVol2VolInputSpec(CommandLineInputSpec):
    moving_image = File(exists=True, desc="Moving image", argstr="--mov %s", mandatory=True)
    target_image = File(exists=True, desc="Target image", argstr="--targ %s", mandatory=True)
    output_image = Str(desc="Output image", argstr="--o %s", mandatory=True)
    fsl_matrix = File(exists=True, desc="FSL transformation matrix (.mat)", argstr="--fsl %s", mandatory=True)
    interp = Str('interpolate', desc='Interpolation method: <interpolate|weighted|nearest|cubic>', argstr='--%s', mandatory=False)

class MRIVol2VolOutputSpec(TraitedSpec):
    output_image = File(desc="Output image", exists=True)

class MRIVol2Vol(CommandLine):
    _cmd = 'mri_vol2vol'
    input_spec = MRIVol2VolInputSpec
    output_spec = MRIVol2VolOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_image'] = os.path.abspath(self.inputs.output_image)
        return outputs