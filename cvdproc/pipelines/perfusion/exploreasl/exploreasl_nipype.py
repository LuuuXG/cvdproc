import os
import json
import shutil
import subprocess
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory

from traits.api import Directory, Str, List
from cvdproc.bids_data.rename_bids_file import rename_bids_file
from cvdproc.config.paths import get_package_path


class ExploreASLCustomInputSpec(BaseInterfaceInputSpec):
    bids_root_dir = Directory(exists=True, mandatory=True, desc="BIDS root directory")
    subject_id = Str(mandatory=True, desc="Subject ID for the raw data")
    session_id = Str(mandatory=True, desc="Session ID for the raw data")
    output_dir = Str(mandatory=True, desc="Output directory for the raw data")
    t1w_filter_filename = Str(mandatory=True, desc="Only copy anat files containing this string in filename")
    asl_filter_filename = Str(mandatory=True, desc="Only copy perf files containing this string in filename")

    script_path = Str(desc='Path to the MATLAB script for ExploreASL', mandatory=True)
    exploreasl_dir = Str(desc='Path to the ExploreASL directory', mandatory=True)

class ExploreASLCustomOutputSpec(TraitedSpec):
    rt1 = Str(desc="Path to the T1.nii.gz file")
    cbf = Str(desc="Path to the CBF.nii.gz file")
    att = Str(desc="Path to the ATT.nii.gz file")
    cbf_and_att = List(desc="Paths to the CBF and ATT.nii.gz files")

class ExploreASLCustom(BaseInterface):
    input_spec = ExploreASLCustomInputSpec
    output_spec = ExploreASLCustomOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.script_path) as script_file:
            script_content = script_file.read()
        
        subject_matlab_script = os.path.join(self.inputs.output_dir, 'ExploreASL_script.m')
        with open(subject_matlab_script, 'w') as f:
            new_script_content = script_content
            new_script_content = new_script_content.replace('/this/is/for/nipype/bids_root_dir', self.inputs.bids_root_dir)
            new_script_content = new_script_content.replace('/this/is/for/nipype/subject_id', self.inputs.subject_id)
            new_script_content = new_script_content.replace('/this/is/for/nipype/session_id', self.inputs.session_id)
            new_script_content = new_script_content.replace('/this/is/for/nipype/output_dir', self.inputs.output_dir)
            new_script_content = new_script_content.replace('/this/is/for/nipype/t1w_filter_filename', self.inputs.t1w_filter_filename)
            new_script_content = new_script_content.replace('/this/is/for/nipype/asl_filter_filename', self.inputs.asl_filter_filename)
            new_script_content = new_script_content.replace('/this/is/for/nipype/exploreasl_dir', self.inputs.exploreasl_dir)
            f.write(new_script_content)
        
        cmd_str = f"run('{subject_matlab_script}'); exit;"
        mlab = CommandLine('matlab', args=f"-nodisplay -nosplash -nodesktop -r \"{cmd_str}\"", terminal_output='stream')
        result = mlab.run()
            
        return result.runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["rt1"] = os.path.join(self.inputs.output_dir, f"sub-{self.inputs.subject_id}_{self.inputs.session_id}", "T1.nii.gz")
        outputs["cbf"] = os.path.join(self.inputs.output_dir, f"sub-{self.inputs.subject_id}_{self.inputs.session_id}", "ASL_1", "CBF.nii.gz")
        outputs["att"] = os.path.join(self.inputs.output_dir, f"sub-{self.inputs.subject_id}_{self.inputs.session_id}", "ASL_1", "ATT.nii.gz")
        outputs["cbf_and_att"] = [outputs["cbf"], outputs["att"]]
        return outputs

class ASLtoT1RegisterInputSpec(CommandLineInputSpec):
    asl_space_img=File(exists=True, mandatory=True, desc="ASL space image", argstr="%s", position=0)
    asl_space_t1w_img=File(exists=True, mandatory=True, desc="ASL space T1w image", argstr="%s", position=1)
    target_t1w_img=File(exists=True, mandatory=True, desc="Target T1w image", argstr="%s", position=2)
    asl_in_t1w_img=Str(desc="Output ASL image in T1w space", mandatory=True, argstr="%s", position=3)

class ASLtoT1RegisterOutputSpec(TraitedSpec):
    out_file=File(exists=True, desc="Output ASL image in T1w space")

class ASLtoT1Register(CommandLine):
    input_spec = ASLtoT1RegisterInputSpec
    output_spec = ASLtoT1RegisterOutputSpec
    _cmd = get_package_path('pipelines', 'bash', 'exploreasl', 'asl_register.sh')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self.inputs.asl_in_t1w_img)
        return outputs