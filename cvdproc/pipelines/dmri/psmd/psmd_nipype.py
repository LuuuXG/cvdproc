import os
import re
import subprocess
from nipype.interfaces.base import (
    CommandLine, CommandLineInputSpec, BaseInterface, BaseTraitedSpec, BaseInterfaceInputSpec, TraitedSpec,
    File, traits
)

from traits.api import Either, Float

class PSMDInputSpec(CommandLineInputSpec):
    dwi_data = File(exists=True, desc="Preprocessed DWI data file", argstr="-p %s")
    bval_file = File(exists=True, desc="B-value file", argstr="-b %s")
    bvec_file = File(exists=True, desc="B-vector file", argstr="-r %s")
    mask_file = File(exists=True, desc="Skeleton mask file", argstr="-s %s")

    enhanced_masking = traits.Int(desc="Use enhanced masking (provide b value)", argstr="-e %d")
    lesion_mask = File(exists=True, desc="Lesion mask file to exclude", argstr="-l %s")
    output_msmd = traits.Bool(desc="Output MSMD instead of PSMD", argstr="-o")
    output_psmd_hemispheres = traits.Bool(desc="Output PSMD separately for hemispheres", argstr="-g")
    clear_temp = traits.Bool(desc="Clear temp folder", argstr="-c")
    quiet = traits.Bool(desc="Quiet mode", argstr="-q")
    verbose = traits.Bool(desc="Verbose mode", argstr="-v")
    troubleshooting = traits.Bool(desc="Keep temp files for troubleshooting", argstr="-t")

    output_dir = File(desc="Directory to write psmd_out.txt", mandatory=True, argstr="-w %s")

class PSMDOutputSpec(TraitedSpec):
    dwi = File(exists=True, desc="DWI data file")
    psmd_out_file = File(desc="Output PSMD file path")

class PSMDCommandLine(CommandLine):
    _cmd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "psmd", "psmd.sh"))
    input_spec = PSMDInputSpec
    output_spec = PSMDOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["dwi"] = os.path.abspath(self.inputs.dwi_data)
        outputs["psmd_out_file"] = os.path.join(self.inputs.output_dir, "psmd_out.txt")
        return outputs
    
# a python interface to save PSMD output value to a csv file
class SavePSMDOutputInputSpec(BaseInterfaceInputSpec):
    psmd = Either(None, Float, desc="PSMD value")
    psmd_left = Either(None, Float, desc="Left hemisphere PSMD value")
    psmd_right = Either(None, Float, desc="Right hemisphere PSMD value")
    output_file = File(desc="Output CSV file path")
    subject_id = Either(None, traits.Str, desc="Subject ID")
    session_id = Either(None, traits.Str, desc="Session ID")

class SavePSMDOutputOutputSpec(TraitedSpec):
    saved_file = File(exists=True, desc="Saved output CSV file")

class SavePSMDOutputCommandLine(BaseInterface):
    input_spec = SavePSMDOutputInputSpec
    output_spec = SavePSMDOutputOutputSpec

    def _run_interface(self, runtime):
        psmd = self.inputs.psmd
        psmd_left = self.inputs.psmd_left
        psmd_right = self.inputs.psmd_right
        output_file = self.inputs.output_file
        subject_id = self.inputs.subject_id
        session_id = self.inputs.session_id

        with open(output_file, "w") as f:
            f.write("subject,session,PSMD,PSMD_Left,PSMD_Right\n")
            f.write(f"sub-{subject_id},ses-{session_id},{psmd},{psmd_left},{psmd_right}\n")

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["saved_file"] = os.path.abspath(self.inputs.output_file)
        return outputs

if __name__ == '__main__':
    from nipype import Workflow, Node

    psmd_node = Node(PSMDCommandLine(), name="psmd_node")
    psmd_node.inputs.dwi_data = "/mnt/f/BIDS/WZdata/derivatives/dwi_pipeline/sub-WZMCI001/ses-01/data.nii"
    psmd_node.inputs.bval_file = "/mnt/f/BIDS/WZdata/derivatives/dwi_pipeline/sub-WZMCI001/ses-01/dwi_denoise_degibbs.bval"
    psmd_node.inputs.bvec_file = "/mnt/f/BIDS/WZdata/derivatives/dwi_pipeline/sub-WZMCI001/ses-01/eddy_corrected_data.eddy_rotated_bvecs"
    psmd_node.inputs.mask_file = "/mnt/e/Codes/cvdproc/cvdproc/pipelines/external/psmd/skeleton_mask_2019.nii.gz"
    psmd_node.inputs.output_dir = "/mnt/f/BIDS/WZdata/derivatives/dwi_pipeline/sub-WZMCI001/ses-01/psmd_out"
    psmd_node.inputs.verbose = True
    
    psmd_node.run()


