from nipype.interfaces.base import (CommandLine, CommandLineInputSpec, traits, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec, File)
from nipype.interfaces.matlab import MatlabCommand
import re
import os
import shlex
import subprocess
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError

class ConcInputSpec(BaseInterfaceInputSpec):
    toolbox_dir = Directory(exists=True, mandatory=True, desc='Path to the DSC-MRI toolbox')
    pwi_path = File(exists=True, mandatory=True, desc='Path to the PWI image file')
    mask_path = File(exists=True, mandatory=True, desc='Path to the mask image file')
    echo_time = Float(mandatory=True, desc='Echo time in seconds')
    repetition_time = Float(mandatory=True, desc='Repetition time in seconds')
    output_path = Directory(exists=True, mandatory=True, desc='Output directory for the modified script')
    output_conc_path = File(mandatory=True, desc='Output path for the concentration image')

    script_path = Str(desc='Path to the MATLAB script that runs the concentration processing')

class ConcOutputSpec(TraitedSpec):
    output_conc_path = File(desc='Path to the output concentration image')

class Conc(BaseInterface):
    input_spec = ConcInputSpec
    output_spec = ConcOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.script_path) as script_file:
            script_content = script_file.read()
        
        # Replace the placeholders in the script
        subject_matlab_script = os.path.join(self.inputs.output_path, 'conc.m')
        with open(subject_matlab_script, 'w') as script_file:
            new_script_content = script_content
            new_script_content = new_script_content.replace('/this/is/for/nipype/dsc_mri_toolbox', 
                                                            self.inputs.toolbox_dir)
            new_script_content = new_script_content.replace('/this/is/for/nipype/pwi_path',
                                                            self.inputs.pwi_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/mask_path',
                                                            self.inputs.mask_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/echo_time',
                                                            str(self.inputs.echo_time))
            new_script_content = new_script_content.replace('/this/is/for/nipype/repetition_time',
                                                            str(self.inputs.repetition_time))
            new_script_content = new_script_content.replace('/this/is/for/nipype/conc_path',
                                                            self.inputs.output_conc_path)
            
            script_file.write(new_script_content)

        cmd_str = f"run('{subject_matlab_script}'); exit;"
        mlab = CommandLine('matlab', args=f"-nodisplay -nosplash -nodesktop -r \"{cmd_str}\"", terminal_output='stream')
        result = mlab.run()
            
        return result.runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_conc_path'] = self.inputs.output_conc_path + '.nii.gz'
        
        return outputs

class DSCMRIInputSpec(BaseInterfaceInputSpec):
    toolbox_dir = Directory(exists=True, mandatory=True, desc='Path to the DSC-MRI toolbox')
    pwi_path = File(exists=True, mandatory=True, desc='Path to the PWI image file')
    aif_conc = File(exists=True, mandatory=True, desc='Path to the AIF concentration file')
    mask_path = File(exists=True, mandatory=True, desc='Path to the mask image file')
    echo_time = Float(mandatory=True, desc='Echo time in seconds')
    repetition_time = Float(mandatory=True, desc='Repetition time in seconds')
    output_path = Directory(exists=True, mandatory=True, desc='Output directory for the modified script')
    output_cbv_path = File(mandatory=True, desc='Output path for the CBV image')
    output_cbv_lc_path = File(mandatory=True, desc='Output path for the CBV_LC image')
    output_k2_path = File(mandatory=True, desc='Output path for the K2 image')
    output_cbf_svd_path = File(mandatory=True, desc='Output path for the CBF_SVD image')
    output_cbf_csvd_path = File(mandatory=True, desc='Output path for the CBF_CSV image')
    output_cbf_osvd_path = File(mandatory=True, desc='Output path for the CBF_OSVD image')
    output_mtt_svd_path = File(mandatory=True, desc='Output path for the MTT_SVD image')
    output_mtt_csvd_path = File(mandatory=True, desc='Output path for the MTT_CSV image')
    output_mtt_osvd_path = File(mandatory=True, desc='Output path for the MTT_OSVD image')
    output_ttp_path = File(mandatory=True, desc='Output path for the TTP image')
    output_s0_path = File(mandatory=True, desc='Output path for the S0 image')

    script_path = Str(desc='Path to the MATLAB script that runs the DSC-MRI processing')

class DSCMRIOutputSpec(TraitedSpec):
    output_cbv_path = File(desc='Path to the output CBV image')
    output_cbv_lc_path = File(desc='Path to the output CBV_LC image')
    output_k2_path = File(desc='Path to the output K2 image')
    output_cbf_svd_path = File(desc='Path to the output CBF_SVD image')
    output_cbf_csvd_path = File(desc='Path to the output CBF_CSV image')
    output_cbf_osvd_path = File(desc='Path to the output CBF_OSVD image')
    output_mtt_svd_path = File(desc='Path to the output MTT_SVD image')
    output_mtt_csvd_path = File(desc='Path to the output MTT_CSV image')
    output_mtt_osvd_path = File(desc='Path to the output MTT_OSVD image')
    output_ttp_path = File(desc='Path to the output TTP image')
    output_s0_path = File(desc='Path to the output S0 image')

class DSCMRI(BaseInterface):
    input_spec = DSCMRIInputSpec
    output_spec = DSCMRIOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.script_path) as script_file:
            script_content = script_file.read()
        
        # Replace the placeholders in the script
        subject_matlab_script = os.path.join(self.inputs.output_path, 'dsc_mri_process.m')
        with open(subject_matlab_script, 'w') as script_file:
            new_script_content = script_content
            new_script_content = new_script_content.replace('/this/is/for/nipype/dsc_mri_toolbox', 
                                                            self.inputs.toolbox_dir)
            new_script_content = new_script_content.replace('/this/is/for/nipype/pwi_path',
                                                            self.inputs.pwi_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/mask_path',
                                                            self.inputs.mask_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/aif_conc.mat',
                                                            self.inputs.aif_conc)
            new_script_content = new_script_content.replace('/this/is/for/nipype/echo_time',
                                                            str(self.inputs.echo_time))
            new_script_content = new_script_content.replace('/this/is/for/nipype/repetition_time',
                                                            str(self.inputs.repetition_time))
            new_script_content = new_script_content.replace('/this/is/for/nipype/cbv_path',
                                                            self.inputs.output_cbv_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/cbv_lc_path',
                                                            self.inputs.output_cbv_lc_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/k2_path',
                                                            self.inputs.output_k2_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/cbf_svd_path',
                                                            self.inputs.output_cbf_svd_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/cbf_csvd_path',
                                                            self.inputs.output_cbf_csvd_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/cbf_osvd_path',
                                                            self.inputs.output_cbf_osvd_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/mtt_svd_path',
                                                            self.inputs.output_mtt_svd_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/mtt_csvd_path',
                                                            self.inputs.output_mtt_csvd_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/mtt_osvd_path',
                                                            self.inputs.output_mtt_osvd_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/ttp_path',
                                                            self.inputs.output_ttp_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/s0_path',
                                                            self.inputs.output_s0_path)
            
            script_file.write(new_script_content)

        cmd_str = f"run('{subject_matlab_script}'); exit;"
        mlab = CommandLine('matlab', args=f"-nodisplay -nosplash -nodesktop -r \"{cmd_str}\"", terminal_output='stream')
        result = mlab.run()
            
        return result.runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_cbv_path'] = self.inputs.output_cbv_path + '.nii.gz'
        outputs['output_cbv_lc_path'] = self.inputs.output_cbv_lc_path + '.nii.gz'
        outputs['output_k2_path'] = self.inputs.output_k2_path + '.nii.gz'
        outputs['output_cbf_svd_path'] = self.inputs.output_cbf_svd_path + '.nii.gz'
        outputs['output_cbf_csvd_path'] = self.inputs.output_cbf_csvd_path + '.nii.gz'
        outputs['output_cbf_osvd_path'] = self.inputs.output_cbf_osvd_path + '.nii.gz'
        outputs['output_mtt_svd_path'] = self.inputs.output_mtt_svd_path + '.nii.gz'
        outputs['output_mtt_csvd_path'] = self.inputs.output_mtt_csvd_path + '.nii.gz'
        outputs['output_mtt_osvd_path'] = self.inputs.output_mtt_osvd_path + '.nii.gz'
        outputs['output_ttp_path'] = self.inputs.output_ttp_path + '.nii.gz'
        outputs['output_s0_path'] = self.inputs.output_s0_path + '.nii.gz'
        
        return outputs

if __name__ == "__main__":
    from nipype import Node, Workflow

    # Example usage
    dsc_mri_node = Node(DSCMRI(), name='dsc_mri')
    dsc_mri_node.inputs.toolbox_dir = '/mnt/e/Codes/cvdproc/cvdproc/pipelines/external/dsc-mri-toolbox'
    dsc_mri_node.inputs.pwi_path = '/mnt/f/BIDS/demo_BIDS/sub-SVDPWI01/ses-01/pwi/sub-SVDPWI01_ses-01_pwi.nii.gz'
    dsc_mri_node.inputs.aif_conc = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_AIF.mat'
    dsc_mri_node.inputs.mask_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_space-pwi_brainmask.nii.gz'
    dsc_mri_node.inputs.echo_time = 0.032
    dsc_mri_node.inputs.repetition_time = 1.5
    dsc_mri_node.inputs.output_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01'
    dsc_mri_node.inputs.output_cbv_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_CBV.nii.gz'
    dsc_mri_node.inputs.output_cbv_lc_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_desc-LeakageCorrection_CBV.nii.gz'
    dsc_mri_node.inputs.output_k2_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_K2map.nii.gz'
    dsc_mri_node.inputs.output_cbf_svd_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_desc-SVD_CBF.nii.gz'
    dsc_mri_node.inputs.output_cbf_csvd_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_desc-CSVD_CBF.nii.gz'
    dsc_mri_node.inputs.output_cbf_osvd_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_desc-OSVD_CBF.nii.gz'
    dsc_mri_node.inputs.output_mtt_svd_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_desc-SVD_MTT.nii.gz'
    dsc_mri_node.inputs.output_mtt_csvd_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_desc-CSVD_MTT.nii.gz'
    dsc_mri_node.inputs.output_mtt_osvd_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_desc-OSVD_MTT.nii.gz'
    dsc_mri_node.inputs.output_ttp_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_TTP.nii.gz'
    dsc_mri_node.inputs.output_s0_path = '/mnt/f/BIDS/demo_BIDS/derivatives/pwi_pipeline/sub-SVDPWI01/ses-01/sub-SVDPWI01_ses-01_S0map.nii.gz'

    dsc_mri_node.inputs.script_path = '/mnt/e/Codes/cvdproc/cvdproc/pipelines/matlab/dsc_mri_toolbox/pwimap.m'

    dsc_mri_node.run()