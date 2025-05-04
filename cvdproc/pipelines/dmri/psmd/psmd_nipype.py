import os
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


class PSMDOutputSpec(TraitedSpec):
    dwi = File(exists=True, desc="DWI data file")
    psmd = traits.Float(desc="Overall PSMD (or same as left if -g used)")
    psmd_left = Either(None, Float, desc="Left hemisphere PSMD (or None)")
    psmd_right = Either(None, Float, desc="Right hemisphere PSMD (or None)")

class PSMDCommandLine(CommandLine):
    _cmd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "psmd", "psmd.sh"))
    input_spec = PSMDInputSpec
    output_spec = PSMDOutputSpec

    def _run_interface(self, runtime):
        result = subprocess.run(
            self.cmdline.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        runtime.returncode = result.returncode
        runtime.stdout = result.stdout
        runtime.stderr = result.stderr

        self._psmd = None
        self._psmd_left = None
        self._psmd_right = None

        if result.returncode == 0:
            for line in result.stdout.splitlines()[::-1]:
                line = line.strip()
                if not line:
                    continue
                if "(" in line and "," in line:
                    # -g 模式：左右半球
                    values = line.split()[-1]
                    left, right = values.split(",")
                    self._psmd_left = float(left)
                    self._psmd_right = float(right)
                else:
                    # 非 -g 模式：整脑
                    val = line.split()[-1]
                    self._psmd = float(val)
                break

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["dwi"] = os.path.abspath(self.inputs.dwi_data)
        outputs["psmd"] = self._psmd
        outputs["psmd_left"] = self._psmd_left
        outputs["psmd_right"] = self._psmd_right
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
    from cvdproc.pipelines.common.unzip import UnzipCommandLine
    from cvdproc.pipelines.common.move_file import MoveFileCommandLine
    from cvdproc.pipelines.common.delete_file import DeleteFileCommandLine
    from nipype import Workflow, Node

    test_wf = Workflow(name="test_wf")
    test_wf.base_dir = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02"

    unzip_node = Node(UnzipCommandLine(), name="unzip_node")
    unzip_node.inputs.file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/bedpostX_input/data.nii.gz"
    unzip_node.inputs.decompress = True
    unzip_node.inputs.keep = True

    move_file_node = Node(MoveFileCommandLine(), name="move_file_node")
    test_wf.connect(unzip_node, "unzipped_file", move_file_node, "source_file")
    move_file_node.inputs.destination_file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/data.nii"

    psmd_node = Node(PSMDCommandLine(), name="psmd_node")
    test_wf.connect(move_file_node, "moved_file", psmd_node, "dwi_data")
    psmd_node.inputs.bval_file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/bedpostX_input/bvals"
    psmd_node.inputs.bvec_file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/bedpostX_input/bvecs"
    psmd_node.inputs.mask_file = "/mnt/e/Codes/cvdproc/cvdproc/pipelines/external/psmd/skeleton_mask_2019.nii.gz"
    psmd_node.inputs.lesion_mask = "/mnt/f/BIDS/SVD_BIDS/derivatives/lesion_mask/sub-SVD0035/ses-02/dti_infarction.nii.gz"

    save_psmd_node = Node(SavePSMDOutputCommandLine(), name="save_psmd_node")
    save_psmd_node.inputs.output_file = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0035/ses-02/psmd_output.csv"
    test_wf.connect(psmd_node, "psmd", save_psmd_node, "psmd")
    test_wf.connect(psmd_node, "psmd_left", save_psmd_node, "psmd_left")
    test_wf.connect(psmd_node, "psmd_right", save_psmd_node, "psmd_right")

    delete_file_node = Node(DeleteFileCommandLine(), name="delete_file_node")
    test_wf.connect(psmd_node, "dwi", delete_file_node, "file")
    
    test_wf.run()


