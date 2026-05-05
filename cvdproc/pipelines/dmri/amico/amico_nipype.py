import os
import subprocess
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str
import amico

class AmicoNoddiInputSpec(BaseInterfaceInputSpec):
    dwi = File(exists=True, desc="Path to the input DWI NIfTI file", mandatory=True)
    bval = File(exists=True, desc="Path to the input b-values file", mandatory=True)
    bvec = File(exists=True, desc="Path to the input b-vectors file", mandatory=True)
    mask = File(exists=True, desc="Path to the brain mask NIfTI file", mandatory=True)
    output_dir = Directory(desc="Directory to save AMICO NODDI results", mandatory=True)

    direction_filename = traits.Str(
    default_value="fit_dir.nii.gz", usedefault=True,
    desc="Filename for the direction map"
    )
    icvf_filename = traits.Str(
        default_value="fit_NDI.nii.gz", usedefault=True,
        desc="Filename for the ICVF map"
    )
    isovf_filename = traits.Str(
        default_value="fit_FWF.nii.gz", usedefault=True,
        desc="Filename for the ISOVF map"
    )
    od_filename = traits.Str(
        default_value="fit_ODI.nii.gz", usedefault=True,
        desc="Filename for the ODI map"
    )
    modulated_icvf_filename = traits.Str(
        default_value="fit_NDI_modulated.nii.gz", usedefault=True,
        desc="Filename for the modulated ICVF map"
    )
    modulated_od_filename = traits.Str(
        default_value="fit_ODI_modulated.nii.gz", usedefault=True,
        desc="Filename for the modulated ODI map"
    )
    config_filename = traits.Str(
        default_value="config.pickle", usedefault=True,
        desc="Filename for the AMICO config file"
    )


class AmicoNoddiOutputSpec(TraitedSpec):
    direction = File(desc="Path to the direction map")
    icvf = File(desc="Path to the ICVF map")
    isovf = File(desc="Path to the ISOVF map")
    od = File(desc="Path to the ODI map")
    modulated_icvf = File(desc="Path to the modulated ICVF map")
    modulated_od = File(desc="Path to the modulated ODI map")
    config = File(desc="Path to the AMICO config file")


class AmicoNoddi(BaseInterface):
    input_spec = AmicoNoddiInputSpec
    output_spec = AmicoNoddiOutputSpec

    def _expected_outputs(self):
        out_dir = os.path.abspath(self.inputs.output_dir)
        return {
            "direction": os.path.join(out_dir, self.inputs.direction_filename),
            "icvf": os.path.join(out_dir, self.inputs.icvf_filename),
            "isovf": os.path.join(out_dir, self.inputs.isovf_filename),
            "od": os.path.join(out_dir, self.inputs.od_filename),
            "modulated_icvf": os.path.join(out_dir, self.inputs.modulated_icvf_filename),
            "modulated_od": os.path.join(out_dir, self.inputs.modulated_od_filename),
            "config": os.path.join(out_dir, self.inputs.config_filename),
        }

    @staticmethod
    def _is_valid_file(path):
        return os.path.isfile(path) and os.path.getsize(path) > 0

    def _run_interface(self, runtime):
        output_dir = os.path.abspath(self.inputs.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        expected = self._expected_outputs()
        expected_paths = list(expected.values())

        # 1) Skip if all expected outputs already exist and are non-empty
        if all(self._is_valid_file(p) for p in expected_paths):
            return runtime

        # 2) If some outputs exist but not all, do NOT overwrite anything
        existing = [p for p in expected_paths if os.path.exists(p)]
        if len(existing) > 0:
            raise RuntimeError(
                "Partial outputs detected. Refusing to run to avoid overwriting existing files. "
                f"Existing files: {existing}"
            )

        amico.setup()
        ae = amico.Evaluation()

        kernels_dir = "/tmp/amico_kernels"
        os.makedirs(kernels_dir, exist_ok=True)
        ae.set_config("KERNELS_path", kernels_dir)

        scheme_path = os.path.join(output_dir, "dwi.scheme")
        amico.util.fsl2scheme(self.inputs.bval, self.inputs.bvec, schemeFilename=scheme_path)

        # Use a clean temporary directory for AMICO internal outputs
        amico_work_dir = os.path.join(output_dir, "_amico_tmp")
        if os.path.exists(amico_work_dir):
            shutil.rmtree(amico_work_dir)
        os.makedirs(amico_work_dir, exist_ok=True)

        try:
            ae.load_data(
                self.inputs.dwi,
                scheme_path,
                mask_filename=self.inputs.mask,
                b0_thr=0,
                replace_bad_voxels=0,
            )
            ae.set_model("NODDI")
            ae.set_config("OUTPUT_path", amico_work_dir)
            ae.set_config("doSaveModulatedMaps", True)

            ae.generate_kernels(regenerate=True)
            ae.load_kernels()
            ae.fit()
            ae.save_results()

            produced = {
                "direction": os.path.join(amico_work_dir, "fit_dir.nii.gz"),
                "icvf": os.path.join(amico_work_dir, "fit_NDI.nii.gz"),
                "isovf": os.path.join(amico_work_dir, "fit_FWF.nii.gz"),
                "od": os.path.join(amico_work_dir, "fit_ODI.nii.gz"),
                "modulated_icvf": os.path.join(amico_work_dir, "fit_NDI_modulated.nii.gz"),
                "modulated_od": os.path.join(amico_work_dir, "fit_ODI_modulated.nii.gz"),
                "config": os.path.join(amico_work_dir, "config.pickle"),
            }

            missing = [src for src in produced.values() if not self._is_valid_file(src)]
            if missing:
                raise RuntimeError(f"AMICO finished but expected output files are missing: {missing}")

            for key, src in produced.items():
                dst = expected[key]
                if os.path.exists(dst):
                    raise RuntimeError(f"Refusing to overwrite existing file: {dst}")
                shutil.move(src, dst)

        finally:
            if os.path.exists(scheme_path):
                try:
                    os.remove(scheme_path)
                except OSError:
                    pass

            if os.path.exists(kernels_dir):
                try:
                    shutil.rmtree(kernels_dir)
                except OSError:
                    pass

            if os.path.exists(amico_work_dir):
                try:
                    shutil.rmtree(amico_work_dir)
                except OSError:
                    pass

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        expected = self._expected_outputs()
        outputs.update(expected)
        return outputs

# class AmicoSANDIInputSpec(BaseInterfaceInputSpec):
#     dwi = File(exists=True, mandatory=True, desc="Input DWI file")

if __name__ == "__main__":
    alps_root = "/mnt/f/UKBdata/WCH_output/WCH_DTI_ALPS"
    dwi_root = "/mnt/f/UKBdata/WCH_DTI"

    sub_dirs = sorted(
        d for d in os.listdir(alps_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(alps_root, d))
    )

    print(f"Found {len(sub_dirs)} subjects.")

    for subject_id in sub_dirs:
        print(f"Processing {subject_id}...")

        dwi = os.path.join(dwi_root, subject_id, "data.nii.gz")
        bval = os.path.join(dwi_root, subject_id, "data.bval")
        bvec = os.path.join(dwi_root, subject_id, "data.bvec")
        mask = os.path.join(dwi_root, subject_id, "mask.nii.gz")
        output_dir = os.path.join(alps_root, subject_id)

        missing_inputs = [p for p in [dwi, bval, bvec, mask] if not os.path.exists(p)]
        if missing_inputs:
            print(f"Skipping {subject_id}, missing inputs: {missing_inputs}")
            continue

        noddi_node = AmicoNoddi()
        noddi_node.inputs.dwi = dwi
        noddi_node.inputs.bval = bval
        noddi_node.inputs.bvec = bvec
        noddi_node.inputs.mask = mask
        noddi_node.inputs.output_dir = output_dir

        try:
            noddi_node.run()
            print(f"Finished {subject_id}")
        except Exception as e:
            print(f"Failed {subject_id}: {e}")