from nipype.interfaces.base import (CommandLine, CommandLineInputSpec, traits, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec, File)
from nipype.interfaces.matlab import MatlabCommand
import re
import os
import shlex
import subprocess
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError

class SepiaQSMInputSpec(BaseInterfaceInputSpec):
    input_qsm_bids_dir = Directory(exists=True, mandatory=True, desc='BIDS directory containing QSM data')
    phase_image_correction = Bool(False, usedefault=True, desc='Perform phase image correction')
    reverse_phase = Int(0, usedefault=True, desc='Reverse phase image')
    subject_output_folder = Str(desc='Output folder name')
    script_path = Str(desc='Path to the script')
    sepia_toolbox_path = Str(desc='Path to the SEPIA toolbox')

class SepiaQSMOutputSpec(TraitedSpec):
    output_folder = Directory(exists=True, desc='Output folder')
    susceptibility_map = Str(desc='Path to the susceptibility map')
    s0_map = Str(desc='Path to the S0 map')
    r2star_map = Str(desc='Path to the R2* map')
    t2star_map = Str(desc='Path to the T2* map')
    swi = Str(desc='Path to the SWI')
    mip = Str(desc='Path to the MIP')

class SepiaQSM(BaseInterface):
    input_spec = SepiaQSMInputSpec
    output_spec = SepiaQSMOutputSpec

    def _run_interface(self, runtime):
        # Load script
        with open(self.inputs.script_path) as script_file:
            script_content = script_file.read()
        
        # Replace the placeholders in the script
        subject_matlab_script = os.path.join(self.inputs.subject_output_folder, 'sepia_qsm_script.m')
        with open(subject_matlab_script, 'w') as script_file:
            new_script_content = script_content
            new_script_content = new_script_content.replace('/this/is/prepared/for/nipype/input_qsm_bids_dir', 
                                                            self.inputs.input_qsm_bids_dir)
            new_script_content = new_script_content.replace('/this/is/prepared/for/nipype/subject_output_folder',
                                                            self.inputs.subject_output_folder)
            new_script_content = new_script_content.replace('/this/is/prepared/for/nipype/sepia_toolbox_path',
                                                            self.inputs.sepia_toolbox_path)
            # Replace 'phase_image_correction = true;' if needed
            if self.inputs.phase_image_correction:
                new_script_content = re.sub(r'phase_image_correction = false;', 'phase_image_correction = true;', new_script_content)
            # Replace 'reverse_phase = 0;' if needed
            if self.inputs.reverse_phase:
                new_script_content = re.sub(r'reverse_phase = 0;', f'reverse_phase = {self.inputs.reverse_phase};', new_script_content)
            
            script_file.write(new_script_content)

        command = [
            'matlab',
            '-nodisplay',
            '-nosplash',
            '-nodesktop',
            '-r',
            f'run(\'{subject_matlab_script}\')'
        ]
        subprocess.run(command)

        self._output_folder = self.inputs.subject_output_folder
            
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_folder'] = self._output_folder
        outputs['susceptibility_map'] = os.path.join(self._output_folder, 'Sepia_Chimap.nii.gz')
        outputs['s0_map'] = os.path.join(self._output_folder, 'Sepia_S0map.nii.gz')
        outputs['r2star_map'] = os.path.join(self._output_folder, 'Sepia_R2starmap.nii.gz')
        outputs['t2star_map'] = os.path.join(self._output_folder, 'Sepia_T2starmap.nii.gz')
        outputs['swi'] = os.path.join(self._output_folder, 'Sepia_clearswi.nii.gz')
        outputs['mip'] = os.path.join(self._output_folder, 'Sepia_clearswi-minIP.nii.gz')

        return outputs


class QSMRegisterInputSpec(BaseInterfaceInputSpec):
    t1w_image = File(mandatory=True, desc='T1w image')
    mag_image = File(mandatory=True, desc='Magnitude image')
    output_dir = Directory(desc='Output directory')
    fsl_anat_dir = Directory(exists=True, desc='FSL ANAT directory')
    qsm_images = traits.List(Str(), desc='List of QSM images')

    qsm_register_script = File(exists=True, desc='Path to the QSM register script')

class QSMRegisterOutputSpec(TraitedSpec):
    output_dir = Directory(desc='Output directory')

class QSMRegister(BaseInterface):
    input_spec = QSMRegisterInputSpec
    output_spec = QSMRegisterOutputSpec

    def _run_interface(self, runtime):
        try:
            subprocess.run([
                'bash', self.inputs.qsm_register_script,
                '--t1w', self.inputs.t1w_image,
                '--mag', self.inputs.mag_image,
                '--output', self.inputs.output_dir,
                '--anat', self.inputs.fsl_anat_dir,
                '--input', *self.inputs.qsm_images
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"QSM registration script failed with error: {e}")

        return runtime

    def _list_outputs(self):
        outputs = {}
        outputs['output_dir'] = self.inputs.output_dir
        return outputs