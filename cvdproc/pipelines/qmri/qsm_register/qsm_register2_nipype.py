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

class QSMRegisterInputSpec(CommandLineInputSpec):
    t1w = File(exists=True, desc="T1w image", mandatory=True, argstr="--t1w %s")
    t1w_to_mni_warp = File(exists=True, desc="T1w to MNI warp file (.nii.gz/.mgz)", mandatory=True, argstr="--t1w_to_mni_warp %s")
    qsm_to_t1w_affine = File(exists=True, desc="QSM to T1w affine matrix file (.mat)", mandatory=True, argstr="--qsm_to_t1w_affine %s")
    output_dir = Str(desc="Output directory", mandatory=True, argstr="--output_dir %s")
    # input = List(Str(exists=True), desc="Input QSM files", mandatory=True, argstr="--input %s...")
    # output1 = List(Str(), desc="Output files in T1w space", argstr="--output1 %s...")
    # output2 = List(Str(), desc="Output files in MNI space", argstr="--output2 %s...")
    input = InputMultiPath(File(exists=True), argstr="--input %s", sep=" ", mandatory=True)
    output1 = List(Str, argstr="--output1 %s", sep=" ", mandatory=True)
    output2 = List(Str, argstr="--output2 %s", sep=" ", mandatory=True)


class QSMRegisterOutputSpec(TraitedSpec):
    outputs_in_t1w = List(Str(), desc="Outputs registered to T1w space")
    outputs_in_mni = List(Str(), desc="Outputs registered to MNI space")

class QSMRegister(CommandLine):
    """
    A Nipype interface to register QSM scalar maps to T1w and MNI space using FSL's applywarp and flirt.

    Example
    -------
    >>> from cvdproc.pipelines.qmri.qsm_register import QSMRegister
    >>> qsm_register = QSMRegister()
    >>> qsm_register.inputs.t1w = 'sub-01_ses-01_T1w.nii.gz'
    >>> qsm_register.inputs.t1w_to_mni_warp = 'sub-01_ses-01_from-T1w_to-MNI_warp.nii.gz'
    >>> qsm_register.inputs.qsm_to_t1w_affine = 'sub-01_ses-01_from-QSM_to-T1w.mat'
    >>> qsm_register.inputs.output_dir = './output'
    >>> qsm_register.inputs.input = ['sub-01_ses-01_qsm.nii.gz', 'sub-01_ses-01_r2star.nii.gz']
    >>> qsm_register.cmdline  # doctest: +ELLIPSIS
    'qsm_register --t1w sub-01_ses-01_T1w.nii.gz --t1w_to_mni_warp sub-01_ses-01_from-T1w_to-MNI_warp.nii.gz --qsm_to_t1w_affine sub-01_ses-01_from-QSM_to-T1w.mat --output_dir ./output --input sub-01_ses-01_qsm.nii.gz sub-01_ses-01_r2star.nii.gz --output1 sub-01_ses-01_qsm_T1w.nii.gz sub-01_ses-01_r2star_T1w.nii.gz --output2 sub-01_ses-01_qsm_MNI.nii.gz sub-01_ses-01_r2star_MNI.nii.gz'
    """

    _cmd = 'bash ' + os.path.join(os.path.dirname(__file__), '..', '..', 'bash', 'qsm', 'qsm_register2.sh')
    input_spec = QSMRegisterInputSpec
    output_spec = QSMRegisterOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs_in_t1w = []
        outputs_in_mni = []

        for out in self.inputs.output1:
            outputs_in_t1w.append(os.path.abspath(os.path.join(self.inputs.output_dir, os.path.basename(out))))
        for out in self.inputs.output2:
            outputs_in_mni.append(os.path.abspath(os.path.join(self.inputs.output_dir, os.path.basename(out))))
        
        outputs['outputs_in_t1w'] = outputs_in_t1w
        outputs['outputs_in_mni'] = outputs_in_mni
        return outputs