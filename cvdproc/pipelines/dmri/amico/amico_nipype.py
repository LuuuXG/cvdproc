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

    direction_filename = Str("fit_dir.nii.gz", desc="Filename for the direction map")
    icvf_filename = Str("fit_NDI.nii.gz", desc="Filename for the ICVF map")
    isovf_filename = Str("fit_FWF.nii.gz", desc="Filename for the ISOVF map")
    od_filename = Str("fit_ODI.nii.gz", desc="Filename for the ODI map")
    modulated_icvf_filename = Str("fit_NDI_modulated.nii.gz", desc="Filename for the modulated ICVF map")
    modulated_od_filename = Str("fit_ODI_modulated.nii.gz", desc="Filename for the modulated ODI map")
    config_filename = Str("config.pickle", desc="Filename for the AMICO config file")


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

        # 2) If some outputs exist but not all, do NOT overwrite anything.
        #    Fail early to protect existing results from being overwritten by AMICO.
        existing = [p for p in expected_paths if os.path.exists(p)]
        if len(existing) > 0:
            raise RuntimeError(
                "Partial outputs detected. Refusing to run to avoid overwriting existing files. "
                f"Existing files: {existing}"
            )

        # AMICO setup
        amico.setup()
        ae = amico.Evaluation()

        # Put kernels somewhere temporary
        kernels_dir = "/tmp/amico_kernels"
        os.makedirs(kernels_dir, exist_ok=True)
        ae.set_config("KERNELS_path", kernels_dir)

        # Build scheme file inside output_dir
        scheme_path = os.path.join(output_dir, "dwi.scheme")
        amico.util.fsl2scheme(self.inputs.bval, self.inputs.bvec, schemeFilename=scheme_path)

        try:
            ae.load_data(
                self.inputs.dwi,
                scheme_path,
                mask_filename=self.inputs.mask,
                b0_thr=0,
                replace_bad_voxels=0,
            )
            ae.set_model("NODDI")
            ae.set_config("OUTPUT_path", output_dir)
            ae.set_config("doSaveModulatedMaps", True)

            ae.generate_kernels(regenerate=True)
            ae.load_kernels()
            ae.fit()
            ae.save_results()

            # AMICO writes fixed filenames. Rename to user-specified filenames if needed.
            produced = {
                "direction": os.path.join(output_dir, "fit_dir.nii.gz"),
                "icvf": os.path.join(output_dir, "fit_NDI.nii.gz"),
                "isovf": os.path.join(output_dir, "fit_FWF.nii.gz"),
                "od": os.path.join(output_dir, "fit_ODI.nii.gz"),
                "modulated_icvf": os.path.join(output_dir, "fit_NDI_modulated.nii.gz"),
                "modulated_od": os.path.join(output_dir, "fit_ODI_modulated.nii.gz"),
                "config": os.path.join(output_dir, "config.pickle"),
            }

            for key, src in produced.items():
                dst = expected[key]
                if os.path.abspath(src) == os.path.abspath(dst):
                    continue
                if os.path.exists(dst):
                    raise RuntimeError(f"Refusing to overwrite existing file: {dst}")
                shutil.move(src, dst)

        finally:
            # Cleanup temp files
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

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        expected = self._expected_outputs()
        outputs.update(expected)
        return outputs

# class AmicoSANDIInputSpec(BaseInterfaceInputSpec):
#     dwi = File(exists=True, mandatory=True, desc="Input DWI file")