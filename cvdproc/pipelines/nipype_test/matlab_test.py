from nipype.interfaces.base import (CommandLine, traits, TraitedSpec,
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
                new_script_content = re.sub(r'phase_image_correction = false;', 'phase_image_correction = true;',
                                            new_script_content)
            # Replace 'reverse_phase = 0;' if needed
            if self.inputs.reverse_phase:
                new_script_content = re.sub(r'reverse_phase = 0;', f'reverse_phase = {self.inputs.reverse_phase};',
                                            new_script_content)

            script_file.write(new_script_content)

            # Run the script
            # mlab = CommandLine('matlab', args=f"-nodisplay -nosplash -nodesktop -r \"run('{subject_matlab_script}')\"; exit;", terminal_output='stream')
            # mlab.run()

            # alternative: use subprocess
            # matlab -nodisplay -nosplash -nodesktop -r "run('/mnt/f/BIDS/demo_BIDS/derivatives/sepia_qsm/sub-YCHC0001/ses-01/sepia_qsm_script.m'); exit;"

            # command = [
            #     'matlab',
            #     '-nodisplay',
            #     '-nosplash',
            #     '-nodesktop',
            #     '-r',
            #     f'run(\'{subject_matlab_script}\')'
            # ]
            command = f'matlab -nodisplay -nosplash -nodesktop -r "run(\'{subject_matlab_script}\')"; exit;'
            subprocess.run(command, shell=True)

            self._output_folder = self.inputs.subject_output_folder

            return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_folder'] = self._output_folder
        return outputs

if __name__ == '__main__':
    sepia_qsm = SepiaQSM()
    sepia_qsm.inputs.input_qsm_bids_dir = '/mnt/f/BIDS/demo_BIDS/sub-YCHC0001/ses-01/qsm'
    sepia_qsm.inputs.phase_image_correction = True
    sepia_qsm.inputs.reverse_phase = 1
    sepia_qsm.inputs.subject_output_folder = '/mnt/f/BIDS/demo_BIDS/derivatives/sepia_qsm/sub-YCHC0001/ses-01'
    sepia_qsm.inputs.script_path = '/mnt/e/Codes/cvdproc/cvdproc/pipelines/matlab/sepia_qsm/sepia_process.m'
    sepia_qsm.inputs.sepia_toolbox_path = '/mnt/e/Neuroimage/Software/sepia-master'

    sepia_qsm.run()