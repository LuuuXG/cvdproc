from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, Str, traits
import os
import glob
import re
import subprocess
import numpy as np
import nibabel as nib

from cvdproc.config.paths import get_package_path


HCP842_QA = get_package_path(
    "data",
    "standard",
    "MNI152",
    "HCP842_QA_1mm.nii.gz"
)

HCP842_TO_HCP1065_WARP = get_package_path(
    "data",
    "standard",
    "MNI152",
    "from-HCP842_to-HCP1065_warp.nii.gz"
)


class LQTInputSpec(BaseInterfaceInputSpec):
    patient_id = Str(desc="Patient ID", mandatory=True)
    lesion_file = File(exists=True, desc="Lesion file", mandatory=True)
    output_dir = Directory(desc="Output directory", mandatory=True)
    parcel_path = File(exists=True, desc="Path to the parcel file", mandatory=True)
    lqt_script = File(exists=True, desc="Path to the LQT R script template", mandatory=True)
    dsi_path = Str(desc="Path to DSI Studio", mandatory=True)

    postprocess_percent_tdi = traits.Bool(
        True,
        usedefault=True,
        desc="Postprocess raw LQT percent TDI maps"
    )

    convert_percent_to_fraction = traits.Bool(
        True,
        usedefault=True,
        desc="Convert percent TDI values from 0-100 to 0-1 before spatial transformation"
    )

    output_space_label = Str(
        "MNI152NLin6ASym",
        usedefault=True,
        desc="Output space label for the final postprocessed TDI file"
    )

    output_desc_label = Str(
        "LQTdisconnection",
        usedefault=True,
        desc="Output description label"
    )

    keep_intermediate = traits.Bool(
        False,
        usedefault=True,
        desc="Keep intermediate HCP842 TDI file"
    )


class LQTOutputSpec(TraitedSpec):
    output_dir = Directory(desc="Directory containing LQT analysis results")
    subject_output_dir = Directory(desc="Directory containing subject-level LQT outputs")
    postprocessed_percent_tdi_file = File(desc="Main postprocessed LQT disconnection TDI file")
    postprocessed_percent_tdi_files = traits.List(File(), desc="All postprocessed LQT disconnection TDI files")


class LQT(BaseInterface):
    input_spec = LQTInputSpec
    output_spec = LQTOutputSpec

    def _normalize_path(self, path):
        return os.path.abspath(path).replace("\\", "/")

    def _run_interface(self, runtime):
        with open(self.inputs.lqt_script, "r", encoding="utf-8") as file:
            script_content = file.read()

        script_content = script_content.replace(
            "/this/is/for/nipype/patient_id",
            self.inputs.patient_id
        )
        script_content = script_content.replace(
            "/this/is/for/nipype/source_lesion_file",
            self._normalize_path(self.inputs.lesion_file)
        )
        script_content = script_content.replace(
            "/this/is/for/nipype/output_dir",
            self._normalize_path(self.inputs.output_dir)
        )
        script_content = script_content.replace(
            "/this/is/for/nipype/parcel_path",
            self._normalize_path(self.inputs.parcel_path)
        )
        script_content = script_content.replace(
            "/this/is/for/nipype/dsi_path",
            self._normalize_path(self.inputs.dsi_path)
        )

        os.makedirs(self.inputs.output_dir, exist_ok=True)
        generated_script_path = os.path.join(
            self.inputs.output_dir,
            "generated_lqt_analysis.R"
        )

        with open(generated_script_path, "w", encoding="utf-8") as file:
            file.write(script_content)

        result = subprocess.run(
            ["Rscript", generated_script_path],
            cwd=self.inputs.output_dir
        )

        if result.returncode != 0:
            raise RuntimeError("Rscript execution failed. Check console output for details.")

        if self.inputs.postprocess_percent_tdi:
            self._postprocess_percent_tdi_maps()

        return runtime

    def _find_lqt_subject_output_dir(self):
        direct_dir = os.path.join(self.inputs.output_dir, self.inputs.patient_id)

        if os.path.isdir(direct_dir):
            return direct_dir

        candidates = []

        for root, dirs, files in os.walk(self.inputs.output_dir):
            if os.path.basename(root) == "Disconnection_Maps":
                candidates.append(os.path.dirname(root))

        candidates = sorted(set(candidates))

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            matched = [
                p for p in candidates
                if os.path.basename(p) == self.inputs.patient_id
            ]

            if len(matched) == 1:
                return matched[0]

            raise RuntimeError(f"Multiple LQT subject output directories found: {candidates}")

        return direct_dir

    def _postprocess_percent_tdi_maps(self):
        subject_output_dir = self._find_lqt_subject_output_dir()
        map_dir = os.path.join(subject_output_dir, "Disconnection_Maps")

        if not os.path.isdir(map_dir):
            print(f"No Disconnection_Maps directory found: {map_dir}")
            return None

        raw_pattern = os.path.join(map_dir, "*_percent_tdi.nii.gz")
        tdi_files = sorted(glob.glob(raw_pattern))

        tdi_files = [
            f for f in tdi_files
            if "_desc-" not in os.path.basename(f)
        ]

        if len(tdi_files) == 0:
            print(f"No raw percent TDI maps found in: {map_dir}")
            return None

        if len(tdi_files) > 1:
            print(f"Multiple raw percent TDI maps found. The first one will be used: {tdi_files[0]}")

        raw_tdi_file = tdi_files[0]

        hcp842_file = self._build_intermediate_hcp842_name(raw_tdi_file)
        final_file = self._build_final_postprocessed_name(raw_tdi_file)

        self._copy_hcp842_header_to_raw_tdi(
            raw_tdi_file=raw_tdi_file,
            hcp842_ref_file=HCP842_QA,
            out_file=hcp842_file
        )

        self._apply_hcp842_to_target_warp(
            in_file=hcp842_file,
            warp_file=HCP842_TO_HCP1065_WARP,
            out_file=final_file
        )

        if not self.inputs.keep_intermediate:
            self._safe_remove(hcp842_file)

        print(f"Final postprocessed LQT TDI: {final_file}")

        return final_file

    def _extract_bids_prefix(self, tdi_file):
        basename = os.path.basename(tdi_file)

        match = re.match(
            r"^(sub-[^_]+_ses-[^_]+)_.*_percent_tdi\.nii\.gz$",
            basename
        )

        if match:
            return match.group(1)

        return self.inputs.patient_id

    def _build_intermediate_hcp842_name(self, tdi_file):
        dirname = os.path.dirname(tdi_file)
        prefix = self._extract_bids_prefix(tdi_file)
        desc_label = str(self.inputs.output_desc_label).strip()

        basename = f"{prefix}_space-HCP842_desc-{desc_label}_tdi.nii.gz"
        return os.path.join(dirname, basename)

    def _build_final_postprocessed_name(self, tdi_file):
        dirname = os.path.dirname(tdi_file)
        prefix = self._extract_bids_prefix(tdi_file)
        target_space = str(self.inputs.output_space_label).strip()
        desc_label = str(self.inputs.output_desc_label).strip()

        entities = [prefix]

        if target_space:
            entities.append(f"space-{target_space}")

        entities.append(f"desc-{desc_label}")

        basename = "_".join(entities) + "_tdi.nii.gz"
        return os.path.join(dirname, basename)

    def _copy_hcp842_header_to_raw_tdi(self, raw_tdi_file, hcp842_ref_file, out_file):
        raw_img = nib.load(raw_tdi_file)
        ref_img = nib.load(hcp842_ref_file)

        if raw_img.shape[:3] != ref_img.shape[:3]:
            raise ValueError(
                f"Shape mismatch between raw LQT TDI and HCP842 reference: "
                f"{raw_img.shape[:3]} vs {ref_img.shape[:3]}"
            )

        data = raw_img.get_fdata(dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data[data < 0] = 0

        if self.inputs.convert_percent_to_fraction:
            data = data / 100.0

        header = ref_img.header.copy()
        header.set_data_dtype(np.float32)

        out_img = nib.Nifti1Image(
            data.astype(np.float32),
            ref_img.affine,
            header
        )

        out_img.set_qform(ref_img.affine, code=1)
        out_img.set_sform(ref_img.affine, code=1)

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        nib.save(out_img, out_file)

        print("Copied HCP842 reference header to raw LQT TDI:")
        print(f"  input:  {raw_tdi_file}")
        print(f"  ref:    {hcp842_ref_file}")
        print(f"  output: {out_file}")
        print(f"  shape:  {raw_img.shape[:3]}")
        print(f"  convert_percent_to_fraction: {self.inputs.convert_percent_to_fraction}")
        print(f"  range: {float(np.nanmin(data))}, {float(np.nanmax(data))}")
        print("  affine:")
        print(ref_img.affine)

    def _apply_hcp842_to_target_warp(self, in_file, warp_file, out_file):
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"Input HCP842 TDI file not found: {in_file}")

        if not os.path.exists(warp_file):
            raise FileNotFoundError(f"HCP842-to-target warp file not found: {warp_file}")

        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        cmd = [
            "mri_convert",
            "-at",
            warp_file,
            in_file,
            out_file
        ]

        print("Running mri_convert warp:")
        print(" ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"mri_convert failed with return code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        if not os.path.exists(out_file):
            raise FileNotFoundError(f"mri_convert did not create output file: {out_file}")

        out_img = nib.load(out_file)
        out_data = out_img.get_fdata(dtype=np.float32)
        out_data = np.nan_to_num(out_data, nan=0.0, posinf=0.0, neginf=0.0)

        print("Warped HCP842 TDI to target space:")
        print(f"  input:  {in_file}")
        print(f"  warp:   {warp_file}")
        print(f"  output: {out_file}")
        print(f"  output shape: {out_img.shape[:3]}")
        print(f"  output range: {float(np.nanmin(out_data))}, {float(np.nanmax(out_data))}")

    def _safe_remove(self, file_path):
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed intermediate file: {file_path}")

    def _get_postprocessed_output_files(self):
        subject_output_dir = self._find_lqt_subject_output_dir()
        map_dir = os.path.join(subject_output_dir, "Disconnection_Maps")

        if not os.path.isdir(map_dir):
            return []

        desc_label = str(self.inputs.output_desc_label).strip()
        target_space = str(self.inputs.output_space_label).strip()

        if target_space:
            pattern = os.path.join(
                map_dir,
                f"*_space-{target_space}_desc-{desc_label}_tdi.nii.gz"
            )
        else:
            pattern = os.path.join(
                map_dir,
                f"*_desc-{desc_label}_tdi.nii.gz"
            )

        return sorted(glob.glob(pattern))

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_dir"] = self.inputs.output_dir

        subject_output_dir = self._find_lqt_subject_output_dir()
        outputs["subject_output_dir"] = subject_output_dir

        out_files = self._get_postprocessed_output_files()
        outputs["postprocessed_percent_tdi_files"] = out_files

        if len(out_files) > 0:
            outputs["postprocessed_percent_tdi_file"] = out_files[0]
        else:
            outputs["postprocessed_percent_tdi_file"] = ""

        return outputs