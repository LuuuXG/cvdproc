from nipype.interfaces.base import (CommandLine, CommandLineInputSpec, traits, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec, File)
from nipype.interfaces.matlab import MatlabCommand
import re
import os
import shlex
import subprocess
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError

class QSMPipelinePart1InputSpec(BaseInterfaceInputSpec):
    bids_root_dir = Directory(desc="BIDS root directory", exists=True, mandatory=True)
    subject_id = Str(desc="Subject ID", mandatory=True)
    session_id = Str(desc="Session ID", mandatory=True)
    cvdproc_dir = Directory(desc="Path to the cvdproc directory", exists=True, mandatory=True)
    phase_image_correction = Bool(desc="If True, apply phase image correction using FSL PRELUDE", mandatory=True)
    reverse_phase = Int(desc="Set to 1 to reverse phase polarity (for GE scanners)", mandatory=True)  # 0=no need, 1=reverse phase image (For GE scans)

    script_path = File(desc="Path to the QSM_pipeline_part1.m script", exists=True, mandatory=True)

class QSMPipelinePart1OutputSpec(TraitedSpec):
    header_path = Str(desc="Path to the sepia header file")
    processed_mag_path = Str(desc="Path to the processed magnitude image")
    processed_phase_path = Str(desc="Path to the processed phase image")
    qsm_mask_path = Str(desc="Path to the QSM brain mask image")
    raw_qsm_path = Str(desc="Path to the output raw QSM image")
    chisep_qsm_path = Str(desc="Path to the output ChiSEP QSM image")
    r2star_path = Str(desc="Path to the output R2* image")
    s0_path = Str(desc="Path to the output S0 image")
    t2star_path = Str(desc="Path to the output T2star image")
    chipara_path = Str(desc="Path to the output ChiPara image")
    chidia_path = Str(desc="Path to the output ChiDia image")
    chitotal_path = Str(desc="Path to the output ChiTotal image")

class QSMPipelinePart1(BaseInterface):
    input_spec = QSMPipelinePart1InputSpec
    output_spec = QSMPipelinePart1OutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.script_path) as script_file:
            script_content = script_file.read()
        
        # Replace the placeholders in the script
        subject_matlab_script = os.path.join(self.inputs.bids_root_dir, 'derivatives', 'qsm_pipeline', f'sub-{self.inputs.subject_id}', f'ses-{self.inputs.session_id}', 'qsm_pipeline_part1_script.m')
        with open(subject_matlab_script, 'w') as script_file:
            new_script_content = script_content
            new_script_content = new_script_content.replace('/this/is/for/nipype/bids_root_dir', 
                                                            self.inputs.bids_root_dir)
            new_script_content = new_script_content.replace('/this/is/for/nipype/subject_id',
                                                            self.inputs.subject_id)
            new_script_content = new_script_content.replace('/this/is/for/nipype/session_id',
                                                            self.inputs.session_id)
            new_script_content = new_script_content.replace('/this/is/for/nipype/cvdproc_dir',
                                                            str(self.inputs.cvdproc_dir))
            new_script_content = new_script_content.replace('/this/is/for/nipype/phase_image_correction',
                                                            str(self.inputs.phase_image_correction))
            new_script_content = new_script_content.replace('/this/is/for/nipype/reverse_phase',
                                                            str(self.inputs.reverse_phase))
            
            script_file.write(new_script_content)
        
        subject_output_dir = os.path.join(self.inputs.bids_root_dir, 'derivatives', 'qsm_pipeline', f'sub-{self.inputs.subject_id}', f'ses-{self.inputs.session_id}')
        self._header_path = os.path.join(subject_output_dir, f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_desc-sepia_header.mat')
        self._processed_mag_path = os.path.join(subject_output_dir, f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_part-mag_desc-smoothed_GRE.nii.gz')
        self._processed_phase_path = os.path.join(subject_output_dir, f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_part-phase_desc-smoothed_GRE.nii.gz')
        self._qsm_mask_path = os.path.join(subject_output_dir, 'QSM_reconstruction', f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_mask_QSM.nii.gz')
        self._raw_qsm_path = os.path.join(subject_output_dir, 'QSM_reconstruction', f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_desc-raw_Chimap.nii.gz')
        self._chisep_qsm_path = os.path.join(subject_output_dir, 'QSM_reconstruction', f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_desc-Chisep_Chimap.nii.gz')
        self._r2star_path = os.path.join(subject_output_dir, 'sepia_output', f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_R2starmap.nii.gz')
        self._s0_path = os.path.join(subject_output_dir, 'sepia_output', f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_S0map.nii.gz')
        self._t2star_path = os.path.join(subject_output_dir, 'sepia_output', f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_T2starmap.nii.gz')
        self._chipara_path = os.path.join(subject_output_dir, 'QSM_reconstruction', f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_ChiPara.nii.gz')
        self._chidia_path = os.path.join(subject_output_dir, 'QSM_reconstruction', f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_ChiDia.nii.gz')
        self._chitotal_path = os.path.join(subject_output_dir, 'QSM_reconstruction', f'sub-{self.inputs.subject_id}_ses-{self.inputs.session_id}_ChiTotal.nii.gz')

        if not os.path.exists(self._chisep_qsm_path):
            cmd_str = f"run('{subject_matlab_script}'); exit;"
            mlab = CommandLine('matlab', args=f"-nodisplay -nosplash -nodesktop -r \"{cmd_str}\"", terminal_output='stream')
            result = mlab.run()
            
            return result.runtime
        else:
            print(f"QSM part 1 already done for sub-{self.inputs.subject_id} ses-{self.inputs.session_id}. Skipping...")
            return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["header_path"] = self._header_path
        outputs["processed_mag_path"] = self._processed_mag_path
        outputs["processed_phase_path"] = self._processed_phase_path
        outputs["qsm_mask_path"] = self._qsm_mask_path
        outputs["raw_qsm_path"] = self._raw_qsm_path
        outputs["chisep_qsm_path"] = self._chisep_qsm_path
        outputs["r2star_path"] = self._r2star_path
        outputs["s0_path"] = self._s0_path
        outputs["t2star_path"] = self._t2star_path
        outputs["chipara_path"] = self._chipara_path
        outputs["chidia_path"] = self._chidia_path
        outputs["chitotal_path"] = self._chitotal_path

        return outputs