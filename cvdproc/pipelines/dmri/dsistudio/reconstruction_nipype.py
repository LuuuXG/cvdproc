import os
import subprocess
import shutil
import nibabel as nib
import time
import numpy as np
import pandas as pd
from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory, isdefined
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str, Float, Either

from cvdproc.config.paths import get_package_path

dsi_studio_path = get_package_path('data', 'lqt', 'extdata', 'DSI_studio', 'dsi-studio', 'dsi_studio')
dsi_studio_newver_path = get_package_path('data', 'dsi-studio', 'dsi_studio')

class DSIstudioReconstructionInputSpec(CommandLineInputSpec):
    # Core Reconstruction Options
    source = Str(desc="SRC File", argstr="--source=%s", mandatory=True)
    method = Int(desc="Reconstruction Method. 1=DTI, 4=GQI, 7=QSDR", argstr="--method=%d", mandatory=True)
    param0 = Float(desc="Parameter 0", argstr="--param0=%f", mandatory=False, default_value=1.25)
    thread_count = Int(desc="Number of Threads", argstr="--thread_count=%d", mandatory=False)
    qsdr_reso = Float(desc="QSDR Resolution", argstr="--qsdr_reso=%f", mandatory=False, default_value=2.0)
    other_output = Str(desc="Other Output File", argstr="--other_output=%s", mandatory=False)
    # Preprocessing Options
    check_btable = Int(desc="Check b-table consistency", argstr="--check_btable=%d", mandatory=False, default_value=0)
    motion_correction = Int(desc="Apply motion correction", argstr="--motion_correction=%d", mandatory=False, default_value=0)
    # I/O & Miscellaneous
    output = Str(desc="Output Fib File Name or Directory", argstr="--output=%s", mandatory=False)
    save_nii = Bool(desc="Save intermediate NIfTI files", argstr="--save_nii", mandatory=False, default_value=False)

class DSIstudioReconstructionOutputSpec(TraitedSpec):
    out_file = File(desc="Output Fib File")

class DSIstudioReconstruction(CommandLine):
    _cmd = dsi_studio_path + " --action=rec"
    input_spec = DSIstudioReconstructionInputSpec
    output_spec = DSIstudioReconstructionOutputSpec

    def _expected_out_file(self):
        """
        Determine the expected output fib file path.
        """
        if self.inputs.output:
            if os.path.isdir(self.inputs.output):
                base_name = os.path.basename(self.inputs.source)
                base_name_noext = os.path.splitext(os.path.splitext(base_name)[0])[0]
                return os.path.join(self.inputs.output, base_name_noext + ".fib.gz")
            else:
                return self.inputs.output
        else:
            base_name = os.path.basename(self.inputs.source)
            base_name_noext = os.path.splitext(os.path.splitext(base_name)[0])[0]
            return os.path.abspath(base_name_noext + ".fib.gz")

    def _run_interface(self, runtime):
        out_file = self._expected_out_file()

        if out_file and os.path.exists(out_file):
            runtime.stdout = f"DSI Studio reconstruction skipped (output exists): {out_file}\n"
            runtime.stderr = ""
            runtime.returncode = 0
            return runtime

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._expected_out_file()
        return outputs


###################################
# Using New Version of DSI-Studio #
###################################
class DSIstudioReconstruction2InputSpec(CommandLineInputSpec):
    source = Str(desc="SRC File", argstr="--source=%s", mandatory=True)
    method = Int(desc="Reconstruction Method", argstr="--method=%d", mandatory=False)
    param0 = Float(desc="Parameter 0", argstr="--param0=%f", mandatory=False, default_value=1.25)
    thread_count = Int(desc="Number of Threads", argstr="--thread_count=%d", mandatory=False)
    qsdr_reso = Float(desc="QSDR Resolution", argstr="--qsdr_reso=%f", mandatory=False, default_value=2.0)
    other_output = Str(desc="Other Output File", argstr="--other_output=%s", mandatory=False)

    check_btable = Int(desc="Check b-table consistency", argstr="--check_btable=%d", mandatory=False, default_value=0)
    motion_correction = Int(desc="Apply motion correction", argstr="--motion_correction=%d", mandatory=False, default_value=0)
    make_isotropic = Float(desc="Make isotropic voxel size (mm)", argstr="--make_isotropic=%f", mandatory=False)

    output = Str(desc="Output Fib File Name or Directory", argstr="--output=%s", mandatory=False)
    save_nii = Str(desc="Save intermediate NIfTI files", argstr="--save_nii=%s", mandatory=False)


class DSIstudioReconstruction2OutputSpec(TraitedSpec):
    out_file = Str(desc="Output Fib File")
    out_nii = Str(desc="Output NIfTI File")
    out_bval = Str(desc="Output BVAL File")
    out_bvec = Str(desc="Output BVEC File")


class DSIstudioReconstruction2(CommandLine):
    _cmd = dsi_studio_newver_path + " --action=rec"
    input_spec = DSIstudioReconstruction2InputSpec
    output_spec = DSIstudioReconstruction2OutputSpec

    def _run_interface(self, runtime):
        # === skip if output already exists ===
        if self.inputs.save_nii and os.path.exists(self.inputs.save_nii):
            print(f"[INFO] File exists, skipping DSI Studio: {self.inputs.save_nii}")
            runtime.returncode = 0
            return runtime

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()

        outputs['out_file'] = self.inputs.output if self.inputs.output else ""
        outputs['out_nii'] = self.inputs.save_nii

        if self.inputs.save_nii:
            outputs['out_bval'] = self.inputs.save_nii.replace(".nii.gz", ".bval")
            outputs['out_bvec'] = self.inputs.save_nii.replace(".nii.gz", ".bvec")
        else:
            outputs['out_bval'] = ""
            outputs['out_bvec'] = ""

        return outputs

# change bvec
class DSIStudioBvecToFSLInputSpec(BaseInterfaceInputSpec):
    in_bvec = File(
        exists=True,
        mandatory=True,
        desc="Input DSI Studio bvec file in N x 3 format",
    )
    out_bvec = File(
        mandatory=False,
        desc="Output FSL bvec file in 3 x N format",
    )
    flip_x = Bool(
        True,
        usedefault=True,
        desc="Flip the x component. This is usually required for DSI Studio to FSL conversion.",
    )
    flip_y = Bool(
        False,
        usedefault=True,
        desc="Flip the y component.",
    )
    flip_z = Bool(
        False,
        usedefault=True,
        desc="Flip the z component.",
    )
    precision = traits.Int(
        10,
        usedefault=True,
        desc="Number of decimal places in the output bvec file.",
    )


class DSIStudioBvecToFSLOutputSpec(TraitedSpec):
    out_bvec = File(
        exists=True,
        desc="Converted FSL bvec file in 3 x N format",
    )


class DSIStudioBvecToFSL(BaseInterface):
    input_spec = DSIStudioBvecToFSLInputSpec
    output_spec = DSIStudioBvecToFSLOutputSpec

    def _run_interface(self, runtime):
        in_bvec = self.inputs.in_bvec

        if isdefined(self.inputs.out_bvec):
            out_bvec = os.path.abspath(self.inputs.out_bvec)
        else:
            base = os.path.basename(in_bvec)
            if base.endswith(".bvec"):
                base = base[:-5]
            out_bvec = os.path.abspath(base + "_fsl.bvec")

        bvec = np.loadtxt(in_bvec)

        if bvec.ndim != 2:
            raise ValueError("Input bvec must be a 2D numeric matrix.")

        if bvec.shape[1] == 3:
            gradients = bvec.copy()
        elif bvec.shape[0] == 3:
            gradients = bvec.T.copy()
        else:
            raise ValueError(
                f"Input bvec must have shape N x 3 or 3 x N, but got {bvec.shape}."
            )

        if self.inputs.flip_x:
            gradients[:, 0] *= -1
        if self.inputs.flip_y:
            gradients[:, 1] *= -1
        if self.inputs.flip_z:
            gradients[:, 2] *= -1

        fsl_bvec = gradients.T

        fmt = f"%.{self.inputs.precision}f"
        np.savetxt(out_bvec, fsl_bvec, fmt=fmt)

        self._out_bvec = out_bvec
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_bvec"] = self._out_bvec
        return outputs

if __name__ == "__main__":
    # Example usage of DSIstudioReconstruction
    recon = DSIstudioReconstruction2()
    recon.inputs.source = "/mnt/e/Neuroimage/demo_data/YC_thalamic_infarcts/sub-TI050/ses-01/dwi/sub-TI050_ses-01_acq-b4000_dwi.sz"
    recon.inputs.thread_count = 14
    recon.inputs.motion_correction = 1
    recon.inputs.check_btable = 1
    recon.inputs.save_nii = "/mnt/e/Neuroimage/demo_data/YC_thalamic_infarcts/sub-TI050/ses-01/dwi/sub-TI050_ses-01_acq-b4000_dwi_mc.nii.gz"
    recon.run()

    fix_bvec = DSIStudioBvecToFSL()
    fix_bvec.inputs.in_bvec = "/mnt/e/Neuroimage/demo_data/YC_thalamic_infarcts/sub-TI050/ses-01/dwi/sub-TI050_ses-01_acq-b4000_dwi_mc.bvec"
    fix_bvec.inputs.out_bvec = "/mnt/e/Neuroimage/demo_data/YC_thalamic_infarcts/sub-TI050/ses-01/dwi/sub-TI050_ses-01_acq-b4000_dwi_mc.bvec"
    fix_bvec.run()