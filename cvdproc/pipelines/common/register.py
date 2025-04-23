from nipype import Node, Workflow
from nipype.interfaces import fsl
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool

class SynthStripInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='-i %s', position=0, desc='Input image to process')
    out_file = File(mandatory=True, argstr='-o %s', position=1, desc='Output stripped image')
    mask_file = File(mandatory=True, argstr='-m %s', position=2, desc='Binary brain mask image')
    use_gpu = Bool(False, argstr='-g', desc='Use GPU for processing')

class SynthStripOutputSpec(TraitedSpec):
    out_file = File(desc='Output stripped image')
    mask_file = File(desc='Binary brain mask image')

class SynthStrip(CommandLine):
    _cmd = 'mri_synthstrip'
    input_spec = SynthStripInputSpec
    output_spec = SynthStripOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        outputs['mask_file'] = self.inputs.mask_file
        return outputs


class ConvertXfmInputSpec(CommandLineInputSpec):
    out_matrix_file = File(mandatory=True, argstr='-omat %s', position=0,
                          desc='Input transformation matrix file')
    in_matrix_file = File(mandatory=True, exists=True, argstr='-inverse %s', position=1, desc='Output inverse matrix file')


class ConvertXfmOutputSpec(TraitedSpec):
    out_matrix_file = File(desc='Inverse transformation matrix')

class ConvertXfm(CommandLine):
    _cmd = 'convert_xfm'
    input_spec = ConvertXfmInputSpec
    output_spec = ConvertXfmOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_matrix_file'] = self.inputs.out_matrix_file
        return outputs


def create_register_workflow(highres_in_file, highres_out_file, highres_mask_file,
                             lowres_in_file, lowres_out_file, lowres_mask_file,
                             flirt_out_matrix_file, flirt_inverse_out_matrix_file, flirt_out_file, base_dir,
                             is_highres_skullstripped=False, is_lowres_skullstripped=False):
    
    inputnode = Node(IdentityInterface(fields=["highres_in_file", "lowres_in_file",
                                               "highres_out_file", "lowres_out_file",
                                               "highres_mask_file", "lowres_mask_file",
                                               "flirt_out_matrix_file", "flirt_inverse_out_matrix_file", "flirt_out_file"]), name="inputnode")
    outputnode = Node(IdentityInterface(fields=["highres_out_file", "lowres_out_file",
                                                "flirt_out_matrix_file", "flirt_out_file", "flirt_inverse_out_matrix_file"]),
                      name="outputnode")
    synthstrip_highres = Node(SynthStrip(), name='synthstrip_highres')
    synthstrip_lowres = Node(SynthStrip(), name='synthstrip_lowres')
    flirt_register = Node(fsl.FLIRT(), name='flirt_register')
    convert_xfm = Node(ConvertXfm(), name='convert_xfm')

    inputnode.inputs.highres_in_file = highres_in_file
    inputnode.inputs.lowres_in_file = lowres_in_file
    inputnode.inputs.highres_out_file = highres_out_file
    inputnode.inputs.lowres_out_file = lowres_out_file
    inputnode.inputs.highres_mask_file = highres_mask_file
    inputnode.inputs.lowres_mask_file = lowres_mask_file
    inputnode.inputs.flirt_out_matrix_file = flirt_out_matrix_file
    inputnode.inputs.flirt_inverse_out_matrix_file = flirt_inverse_out_matrix_file
    inputnode.inputs.flirt_out_file = flirt_out_file

    workflow = Workflow(name='register_between_modalities')
    workflow.base_dir = base_dir

    if not is_highres_skullstripped:
        workflow.connect(inputnode, 'highres_in_file', synthstrip_highres, 'in_file')
        workflow.connect(inputnode, 'highres_out_file', synthstrip_highres, 'out_file')
        workflow.connect(inputnode, 'highres_mask_file', synthstrip_highres, 'mask_file')
        workflow.connect(synthstrip_highres, 'out_file', flirt_register, 'reference')
    else:
        workflow.connect(inputnode, 'highres_in_file', flirt_register, 'reference')

    if not is_lowres_skullstripped:
        workflow.connect(inputnode, 'lowres_in_file', synthstrip_lowres, 'in_file')
        workflow.connect(inputnode, 'lowres_out_file', synthstrip_lowres, 'out_file')
        workflow.connect(inputnode, 'lowres_mask_file', synthstrip_lowres, 'mask_file')
        workflow.connect(synthstrip_lowres, 'out_file', flirt_register, 'in_file')
    else:
        workflow.connect(inputnode, 'lowres_in_file', flirt_register, 'in_file')

    workflow.connect(inputnode, "flirt_out_file", flirt_register, "out_file")
    workflow.connect(inputnode, "flirt_inverse_out_matrix_file", convert_xfm, "out_matrix_file")
    workflow.connect(flirt_register, 'out_matrix_file', convert_xfm, 'in_matrix_file')

    workflow.connect(inputnode, 'highres_out_file', outputnode, 'highres_out_file')
    workflow.connect(inputnode, 'lowres_out_file', outputnode, 'lowres_out_file')
    workflow.connect(flirt_register, 'out_file', outputnode, 'flirt_out_file')
    workflow.connect(flirt_register, 'out_matrix_file', outputnode, 'flirt_out_matrix_file')
    workflow.connect(convert_xfm, 'out_matrix_file', outputnode, 'flirt_inverse_out_matrix_file')

    return workflow

# if __name__ == '__main__':
#     from nipype.interfaces.utility import Merge
#     # 示例输入
#     highres_in_file = '/mnt/f/BIDS/UKB_AFproject/sub-AF1000077/ses-01/anat/sub-AF1000077_ses-01_T1w.nii.gz'
#     highres_out_file = '/mnt/f/BIDS/UKB_AFproject/derivatives/nipype/sub-AF1000077_ses-01_T1w_stripped.nii.gz'
#     highres_mask_file = '/mnt/f/BIDS/UKB_AFproject/derivatives/nipype/sub-AF1000077_ses-01_T1w_mask.nii.gz'
#     lowres_in_file = '/mnt/f/BIDS/UKB_AFproject/sub-AF1000077/ses-01/anat/sub-AF1000077_ses-01_FLAIR.nii.gz'
#     lowres_out_file = '/mnt/f/BIDS/UKB_AFproject/derivatives/nipype/sub-AF1000077_ses-01_FLAIR_stripped.nii.gz'
#     lowres_mask_file = '/mnt/f/BIDS/UKB_AFproject/derivatives/nipype/sub-AF1000077_ses-01_FLAIR_mask.nii.gz'
#     flirt_out_matrix_file = '/mnt/f/BIDS/UKB_AFproject/derivatives/nipype/flirt2t1w.mat'
#     flirt_inverse_out_matrix_file = '/mnt/f/BIDS/UKB_AFproject/derivatives/nipype/t1w2flair.mat'
#     flirt_out_file = '/mnt/f/BIDS/UKB_AFproject/derivatives/nipype/sub-AF1000077_ses-01_FLAIR_registered.nii.gz'
#     base_dir = '/mnt/f/BIDS/UKB_AFproject/derivatives/nipype'
#
#     # 创建工作流
#     workflow = create_register_workflow(highres_in_file, highres_out_file, highres_mask_file,
#                                         lowres_in_file, lowres_out_file, lowres_mask_file,
#                                         flirt_out_matrix_file, flirt_inverse_out_matrix_file, flirt_out_file, base_dir,
#                                         is_highres_skullstripped=True, is_lowres_skullstripped=True)
#
#     merge_node = Node(Merge(numinputs=2, outputnames=["image_to_predict"]), name="merge_node")
#
#     workflow_test = Workflow(name="workflow_test")
#     workflow_test.base_dir = '/mnt/f/BIDS/UKB_AFproject/derivatives/nipype'
#
#     workflow_test.connect([
#         (workflow, merge_node, [("outputnode.highres_out_file", "in1")]),
#         (workflow, merge_node, [("outputnode.flirt_out_file", "in2")])
#     ])
#
#     workflow_test.run()

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
    _cmd = os.path.join(script_dir, 'register.sh')

    input_spec = ModalityRegistrationInputSpec
    output_spec = ModalityRegistrationOutputSpec

    # def _strip_nii_ext(self, filename):
    #     if filename.endswith('.nii.gz'):
    #         return filename[:-7]
    #     elif filename.endswith('.nii'):
    #         return filename[:-4]
    #     return filename

    def _list_outputs(self):
        outputs = self._outputs().get()

        #target_name = self._strip_nii_ext(os.path.basename(self.inputs.image_target))
        #source_name = self._strip_nii_ext(os.path.basename(self.inputs.image_source))

        out_dir = self.inputs.output_dir
        outputs['output_image'] = os.path.join(out_dir, self.inputs.registered_image_filename)
        outputs['source_to_target_mat'] = os.path.join(out_dir, self.inputs.source_to_target_mat_filename)
        outputs['target_to_source_mat'] = os.path.join(out_dir, self.inputs.target_to_source_mat_filename)

        return outputs

if __name__ == '__main__':
    # Example usage
    image_target = '/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-01/anat/sub-SVD0100_ses-01_acq-highres_T1w.nii.gz'
    image_target_strip = 1
    image_source = '/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-01/anat/sub-SVD0100_ses-01_acq-highres_FLAIR.nii.gz'
    image_source_strip = 1
    flirt_direction = 1
    output_dir = '/mnt/f/BIDS/SVD_BIDS/derivatives/xfm/sub-SVD0100/ses-01'

    registration = ModalityRegistration()
    registration.inputs.image_target = image_target
    registration.inputs.image_target_strip = image_target_strip
    registration.inputs.image_source = image_source
    registration.inputs.image_source_strip = image_source_strip
    registration.inputs.flirt_direction = flirt_direction
    registration.inputs.output_dir = output_dir
    registration.inputs.registered_image_filename = 'sub-SVD0100_ses-01_acq-highres_FLAIR_registered.nii.gz'
    registration.inputs.source_to_target_mat_filename = 'sub-SVD0100_ses-01_acq-highres_FLAIR2T1w.mat'
    registration.inputs.target_to_source_mat_filename = 'sub-SVD0100_ses-01_acq-highres_T1w2FLAIR.mat'
    registration.inputs.dof = 6

    result = registration.run()