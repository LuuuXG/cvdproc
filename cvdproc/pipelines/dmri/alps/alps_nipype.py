import os
import subprocess
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str

from cvdproc.config.paths import get_package_path

############
# DTI-ALPS # (Deprecated)
############
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
    output_dir = Directory(exists=True, desc='Output directory')
    alps_stat = Str(desc='ALPS statistics')

class DTIALPS(CommandLine):
    _cmd = 'alps.sh'
    input_spec = DTIALPSInputSpec
    output_spec = DTIALPSOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['alps_stat'] = os.path.join(self.inputs.output_dir, 'alps.stat')
        outputs['output_dir'] = self.inputs.output_dir
        return outputs

def input_for_alps(dti_fa, dti_md, dti_tensor, input_dir, alps_script_path):
    import os
    import subprocess
    os.makedirs(input_dir, exist_ok=True)

    alps_dir = os.path.dirname(alps_script_path)
    os.environ["PATH"] = os.environ.get("PATH", "") + ":" + alps_dir

    # copy the files to the input directory
    fa_file = subprocess.run(['cp', dti_fa, input_dir], check=True)
    md_file = subprocess.run(['cp', dti_md, input_dir], check=True)
    tensor_file = subprocess.run(['cp', dti_tensor, input_dir], check=True)

    return fa_file, md_file, tensor_file, input_dir

def delete_alps_inputs(input_dir):
    import os
    import subprocess
    files_to_delete = [os.path.join(input_dir, 'dti_FA.nii.gz'), os.path.join(input_dir, 'dti_MD.nii.gz'), os.path.join(input_dir, 'dti_tensor.nii.gz')]
    for file in files_to_delete:
        os.remove(file)
    
    return files_to_delete

####################################
# DTI-ALPS simple python interface # (Also deprecated)
####################################
class DTIALPSsimpleInputSpec(BaseInterfaceInputSpec):
    perform_roi_analysis = Str(desc='Perform ROI analysis', default_value='1')
    use_templete = Str(desc='Use template', default_value='1')
    t1w_file = Str(desc='Path to the T1-weighted image')
    fa_file = File(exists=True, desc='Path to the FA image')
    md_file = File(exists=True, desc='Path to the MD image')
    tensor_file = File(exists=True, desc='Path to the tensor image')
    alps_input_dir = Directory(desc='ALPS input directory')
    skip_preprocessing = Str(desc='Skip preprocessing', default_value='0')
    alps_script_path = Str(desc='Path to the alps script')

class DTIALPSsimpleOutputSpec(TraitedSpec):
    alps_stat = Str(desc='ALPS statistics')

class DTIALPSsimple(BaseInterface):
    input_spec = DTIALPSsimpleInputSpec
    output_spec = DTIALPSsimpleOutputSpec

    def _run_interface(self, runtime):
        perform_roi_analysis = self.inputs.perform_roi_analysis
        use_templete = self.inputs.use_templete
        alps_input_dir = self.inputs.alps_input_dir
        skip_preprocessing = self.inputs.skip_preprocessing
        alps_script_path = self.inputs.alps_script_path

        os.makedirs(alps_input_dir, exist_ok=True)
        # copy the files to the input directory
        shutil.copy(self.inputs.fa_file, os.path.join(alps_input_dir, 'dti_FA.nii.gz'))
        shutil.copy(self.inputs.md_file, os.path.join(alps_input_dir, 'dti_MD.nii.gz'))
        shutil.copy(self.inputs.tensor_file, os.path.join(alps_input_dir, 'dti_tensor.nii.gz'))

        # if provide T1w
        if os.path.exists(self.inputs.t1w_file):
            subprocess.run([
                'bash', alps_script_path,
                '-s', skip_preprocessing,
                '-r', perform_roi_analysis,
                '-t', use_templete,
                '-o', alps_input_dir,
                '-v', self.inputs.t1w_file
            ])
        else:
            subprocess.run([
                'bash', alps_script_path,
                '-s', skip_preprocessing,
                '-r', perform_roi_analysis,
                '-t', use_templete,
                '-o', alps_input_dir
            ])

        # delete the input files
        files_to_delete = [
            os.path.join(alps_input_dir, 'dti_FA.nii.gz'),
            os.path.join(alps_input_dir, 'dti_MD.nii.gz'),
            os.path.join(alps_input_dir, 'dti_tensor.nii.gz')
        ]

        for file in files_to_delete:
            os.remove(file)

        self._alps_stat = os.path.join(alps_input_dir, 'alps.stat')

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['alps_stat'] = self._alps_stat

        return outputs
    
########
# ALPS #
########
# # REQUIRED INPUTS:
# fa_img='' # Input FA image (NIfTI format): -fa_img <path>
# output_dir='' # Output directory: -output_dir <path>
# alps_dir='' # ALPS script directory: -alps_dir <path>
# register_method='flirt' # Registration method: flirt or synthmorph (default: flirt): -register_method <method>

# # OPTIONAL INPUTS:
# xx_img='' # Input XX image (NIfTI format): -xx_img <path>
# yy_img='' # Input YY image (NIfTI format): -yy_img <path>
# zz_img='' # Input ZZ image (NIfTI format): -zz_img <path>
# tensor_img='' # Input 4D tensor image (NIfTI format): -tensor_img <path> (if provided, will ignore XX, YY, ZZ images)
# t1_img='' # Input T1w image (NIfTI format): -t1_img <path>
# t1_to_mni_warp='' # Input T1 to MNI warp file (ANTs format): -t1_to_mni_warp <path>
class ALPSInputSpec(CommandLineInputSpec):
    fa_img = Str(argstr='-fa_img %s', position=0, desc='Input FA image (NIfTI format)', mandatory=True)
    output_dir = Str(argstr='-output_dir %s', position=1, desc='Output directory', mandatory=True)
    alps_dir = Str(argstr='-alps_dir %s', position=2, desc='ALPS script directory', mandatory=True)
    register_method = Str(argstr='-register_method %s', position=3, desc='Registration method: flirt or synthmorph', default_value='flirt')
    xx_img = Str(argstr='-xx_img %s', position=4, desc='Input XX image (NIfTI format)', mandatory=False)
    yy_img = Str(argstr='-yy_img %s', position=5, desc='Input YY image (NIfTI format)', mandatory=False)
    zz_img = Str(argstr='-zz_img %s', position=6, desc='Input ZZ image (NIfTI format)', mandatory=False)
    tensor_img = Str(argstr='-tensor_img %s', position=7, desc='Input 4D tensor image (NIfTI format)', mandatory=False)
    t1_img = Str(argstr='-t1_img %s', position=8, desc='Input T1w image (NIfTI format)', mandatory=False)
    t1_to_mni_warp = Str(argstr='-t1_to_mni_warp %s', position=9, desc='Input T1 to MNI warp file (Synthmorph format)', mandatory=False)

class ALPSOutputSpec(TraitedSpec):
    alps_stat = Str(desc='ALPS statistics file path')

class ALPS(CommandLine):
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'alps', 'alps.sh')
    input_spec = ALPSInputSpec
    output_spec = ALPSOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['alps_stat'] = os.path.join(self.inputs.output_dir, 'alps.stat', 'alps.csv')
        return outputs