import os
import shutil
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError

from cvdproc.config.paths import get_package_path

class QSIPrepOrigToACPCInputSpec(CommandLineInputSpec):
    subject_id = Str(mandatory=True, desc="BIDS Subject ID", argstr="%s", position=0)
    session_id = Str(mandatory=True, desc="BIDS Session ID", argstr="%s", position=1)
    dwimap_file = File(exists=True, mandatory=True, desc="Path to the ACPC space dwimap file", argstr="%s", position=2)
    preprocess_t1w_file = File(exists=True, mandatory=True, desc="Path to the preprocessed T1w file", argstr="%s", position=3)
    orig_t1w_file = File(exists=True, mandatory=True, desc="Path to the original T1w file", argstr="%s", position=4)
    output_dir = Directory(mandatory=True, desc="Output directory", argstr="%s", position=5)

class QSIPrepOrigToACPCOutputSpec(TraitedSpec):
    out_matrix_file = File(desc="Output transformation matrix file")
    out_file = File(desc="Output transformed file")

class QSIPrepOrigToACPC(CommandLine):
    _cmd = 'bash ' + get_package_path('pipelines', 'bash', 'qsiprep_qsirecon', 'qsiprep_orig_to_acpc.sh')
    input_spec = QSIPrepOrigToACPCInputSpec
    output_spec = QSIPrepOrigToACPCOutputSpec
    def _list_outputs(self):
        outputs = self._outputs().get()
        subject_id = self.inputs.subject_id
        session_id = self.inputs.session_id
        output_dir = self.inputs.output_dir

        matrix_filename = f"sub-{subject_id}_ses-{session_id}_from-ACPC_to-T1w_xfm.mat"
        dwimap_base = os.path.basename(self.inputs.dwimap_file)
        if "space-ACPC" in dwimap_base:
            transformed_filename = dwimap_base.replace("space-ACPC", "space-T1w")
        else:
            if dwimap_base.endswith(".nii.gz"):
                transformed_filename = dwimap_base[:-7] + "_space-T1w.nii.gz"
            elif dwimap_base.endswith(".nii"):
                transformed_filename = dwimap_base[:-4] + "_space-T1w.nii"
            else:
                transformed_filename = dwimap_base + "_space-T1w.nii.gz"

        outputs['out_matrix_file'] = os.path.join(output_dir, matrix_filename)
        outputs['out_file'] = os.path.join(output_dir, transformed_filename)

        return outputs