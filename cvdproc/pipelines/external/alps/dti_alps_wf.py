import os
from nipype import Node, Workflow
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str

class DTIALPSInputSpec(CommandLineInputSpec):
    # -a 4d nifti file
    input_dwi = Str(argstr='-a %s', position=0, desc='Input 4D nifti file', mandatory=False)
    # -b bval
    bval = Str(argstr='-b %s', position=1, desc='bval', mandatory=False)
    # -c bvec
    bvec = Str(argstr='-c %s', position=2, desc='bvec', mandatory=False)
    # -m json
    json = Str(argstr='-m %s', position=3, desc='json', mandatory=False)
    # -i a second dwi file
    input_dwi2 = Str(argstr='-i %s', position=4, desc='A second dwi file', mandatory=False)
    # -j bval
    bval2 = Str(argstr='-j %s', position=5, desc='bval', mandatory=False)
    # -k bvec
    bvec2 = Str(argstr='-k %s', position=6, desc='bvec', mandatory=False)
    # -m json
    json2 = Str(argstr='-m %s', position=7, desc='json', mandatory=False)
    # -d 0=skip, 1=both denoising and unringing, 3=only denoising, 4=only unringing
    preprocessing_steps = Str(argstr='-d %s', position=8, desc='Preprocessing steps', mandatory=False)
    # -e 0=skip eddy, 1=eddy_cpu, 2=eddy, 3=eddy_correct
    eddy = Str(argstr='-e %s', position=9, desc='Eddy', mandatory=False)
    # -r 0=skip ROI analysis, 1=do ROI analysis
    roi_analysis = Str(argstr='-r %s', position=10, desc='ROI analysis', mandatory=False)
    # -t 0=native, 1=JHU-ICBM
    template = Str(argstr='-t %s', position=11, desc='Template', mandatory=False)
    # -v volume structure file
    volume_file = Str(argstr='-v %s', position=12, desc='Volume structure file', mandatory=False)
    # -h 1=t1w, 2=t2w
    struc_type = Str(argstr='-h %s', position=13, desc='Structural file modality', mandatory=False)
    # -w 0=linear, 1=nonlinear, 2=both
    warp = Str(argstr='-w %s', position=14, desc='Warp', mandatory=False)
    # -f 1=flirt or applywrap, 2=vecreg
    tensor_transform = Str(argstr='-f %s', position=15, desc='Tensor transform', mandatory=False)
    # -o output directory
    output_dir = Directory(exists=True, argstr='-o %s', position=16, desc='Output directory', mandatory=True)
    # -s 0=do, 1=skip preprocessing
    skip_preprocessing = Str(argstr='-s %s', position=17, desc='Skip preprocessing', mandatory=False)

class DTIALPSOutputSpec(TraitedSpec):
    # will generate a folder 'alps.stat' in the output directory
    alps_stat = File(desc='ALPS statistics')

class DTIALPS(CommandLine):
    _cmd = 'alps.sh'
    input_spec = DTIALPSInputSpec
    output_spec = DTIALPSOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['alps_stat'] = os.path.join(self.inputs.output_dir, 'alps.stat')
        return outputs

def create_dtialps_workflow(input_dwi=None, bval=None, bvec=None, json=None, input_dwi2=None, bval2=None, bvec2=None, json2=None,
                            preprocessing_steps=None, eddy=None, roi_analysis=None, template=None, volume_file=None,
                            struc_type=None, warp=None, tensor_transform=None, output_dir=None, skip_preprocessing=None):
    inputnode = Node(IdentityInterface(fields=['input_dwi', 'bval', 'bvec', 'json', 'input_dwi2', 'bval2',
                                                'bvec2', 'json2', 'preprocessing_steps', 'eddy', 'roi_analysis',
                                                'template', 'volume_file', 'struc_type', 'warp', 'tensor_transform',
                                                'output_dir', 'skip_preprocessing']), name='inputnode')
    outputnode = Node(IdentityInterface(fields=['alps_stat']), name='outputnode')

    alps = Node(DTIALPS(), name='alps')

    # Assigning input values
    if input_dwi is not None:
        inputnode.inputs.input_dwi = input_dwi
    if bval is not None:
        inputnode.inputs.bval = bval
    if bvec is not None:
        inputnode.inputs.bvec = bvec
    if json is not None:
        inputnode.inputs.json = json
    if input_dwi2 is not None:
        inputnode.inputs.input_dwi2 = input_dwi2
    if bval2 is not None:
        inputnode.inputs.bval2 = bval2
    if bvec2 is not None:
        inputnode.inputs.bvec2 = bvec2
    if preprocessing_steps is not None:
        inputnode.inputs.preprocessing_steps = preprocessing_steps
    if eddy is not None:
        inputnode.inputs.eddy = eddy
    if roi_analysis is not None:
        inputnode.inputs.roi_analysis = roi_analysis
    if template is not None:
        inputnode.inputs.template = template
    if volume_file is not None:
        inputnode.inputs.volume_file = volume_file
    if struc_type is not None:
        inputnode.inputs.struc_type = struc_type
    if warp is not None:
        inputnode.inputs.warp = warp
    if tensor_transform is not None:
        inputnode.inputs.tensor_transform = tensor_transform
    if output_dir is not None:
        inputnode.inputs.output_dir = output_dir
    if skip_preprocessing is not None:
        inputnode.inputs.skip_preprocessing = skip_preprocessing

    # Creating workflow
    workflow = Workflow(name='dti_alps_wf')
    workflow.base_dir = output_dir

    # Connecting input node and alps node
    workflow.connect(inputnode, 'input_dwi', alps, 'input_dwi')
    workflow.connect(inputnode, 'bval', alps, 'bval')
    workflow.connect(inputnode, 'bvec', alps, 'bvec')
    workflow.connect(inputnode, 'json', alps, 'json')
    workflow.connect(inputnode, 'input_dwi2', alps, 'input_dwi2')
    workflow.connect(inputnode, 'bval2', alps, 'bval2')
    workflow.connect(inputnode, 'bvec2', alps, 'bvec2')
    workflow.connect(inputnode, 'json2', alps, 'json2')
    workflow.connect(inputnode, 'preprocessing_steps', alps, 'preprocessing_steps')
    workflow.connect(inputnode, 'eddy', alps, 'eddy')
    workflow.connect(inputnode, 'roi_analysis', alps, 'roi_analysis')
    workflow.connect(inputnode, 'template', alps, 'template')
    workflow.connect(inputnode, 'volume_file', alps, 'volume_file')
    workflow.connect(inputnode, 'struc_type', alps, 'struc_type')
    workflow.connect(inputnode, 'warp', alps, 'warp')
    workflow.connect(inputnode, 'tensor_transform', alps, 'tensor_transform')
    workflow.connect(inputnode, 'output_dir', alps, 'output_dir')
    workflow.connect(inputnode, 'skip_preprocessing', alps, 'skip_preprocessing')

    workflow.connect(alps, 'alps_stat', outputnode, 'alps_stat')

    return workflow
    
# Test
if __name__ == '__main__':
    input_dwi = ''
    bval = ''
    bvec = ''
    json = ''
    input_dwi2 = ''
    bval2 = ''
    bvec2 = ''
    json2 = ''
    preprocessing_steps = '0'
    eddy = '0'
    roi_analysis = '1'
    template = '1'
    #volume_file = '/mnt/f/BIDS/SVD_BIDS/sub-SVD0035/ses-02/anat/sub-SVD0035_ses-02_acq-highres_T1w.nii.gz'
    volume_file = None
    #struc_type = '1'
    struc_type = ''
    warp = '0'
    tensor_transform = '1'
    output_dir = '/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02'
    skip_preprocessing = '1'

    wf = create_dtialps_workflow(roi_analysis=roi_analysis,
                                 template=template, output_dir=output_dir, skip_preprocessing=skip_preprocessing)
    wf.base_dir = output_dir
    wf.run()