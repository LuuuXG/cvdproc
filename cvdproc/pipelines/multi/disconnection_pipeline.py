"""Normative IIT and individual-connectome disconnection workflows."""
from __future__ import annotations

import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nipype import Node, Workflow

from cvdproc.bids_data.rename_bids_file import rename_bids_file
from cvdproc.config.paths import get_package_path
from cvdproc.pipelines.multi.disconnection.disconnection_nipype import IITDisconnection, IndividualDisconnection


class DisconnectionPipeline:
    def __init__(self, subject, session, output_path, methods=None, structural_methods=None, functional_methods=None,
                 mni_lesion_mask="lesion_mask", use_which_mni_lesion_mask=None,
                 t1w_lesion_mask="lesion_mask", use_which_t1w_lesion_mask=None,
                 individual_connectome_source="mrtrix3", use_freesurfer_transform=False,
                 extract_from=None, **kwargs):
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.methods = ["normative", "individual"] if methods is None else methods
        self.structural_methods = ["any_hit"] if structural_methods is None else structural_methods
        self.functional_methods = [] if functional_methods is None else functional_methods
        self.mni_lesion_mask = mni_lesion_mask
        self.use_which_mni_lesion_mask = use_which_mni_lesion_mask
        self.t1w_lesion_mask = t1w_lesion_mask
        self.use_which_t1w_lesion_mask = use_which_t1w_lesion_mask
        self.individual_connectome_source = individual_connectome_source.lower()
        self.use_freesurfer_transform = use_freesurfer_transform
        self.extract_from = extract_from

    def check_data_requirements(self):
        # As in LQT, input selection and validation take place in create_workflow.
        return True

    def create_workflow(self):
        for name in ("methods", "structural_methods", "functional_methods"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"{name} must be a list of method names")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if not self.methods or set(self.methods) - {"normative", "individual"}:
            raise ValueError("methods must select 'normative', 'individual', or both")
        supported_models = ("any_hit", "absolute_length", "fractional_length")
        unknown = set(self.structural_methods) - set(supported_models)
        if unknown:
            raise ValueError(f"Unknown structural_methods: {sorted(unknown)}")
        if self.functional_methods:
            raise NotImplementedError("Functional disconnection is not implemented")
        if not self.structural_methods:
            raise ValueError("Select at least one structural method; functional analysis is not implemented")
        for method, folder, pattern in (
                ("normative", self.mni_lesion_mask, self.use_which_mni_lesion_mask),
                ("individual", self.t1w_lesion_mask, self.use_which_t1w_lesion_mask)):
            if method in self.methods and (not folder or not pattern):
                raise ValueError(f"Both lesion directory and filename match must be provided for {method}")
        if "individual" in self.methods and self.individual_connectome_source not in {"mrtrix3", "dsistudio"}:
            raise ValueError("individual_connectome_source must be 'mrtrix3' or 'dsistudio'")

        visit = [f"sub-{self.subject.subject_id}"]
        if self.session is not None:
            visit.append(f"ses-{self.session.session_id}")
        prefix = "_".join(visit)
        derivatives = Path(self.subject.bids_dir) / "derivatives"
        xfm_dir = derivatives.joinpath("xfm", *visit)
        mni_space = "MNI152NLin6Asym"
        atlas_names = ["AAL116", "Shen268", "FreeSurfer86Avg"]

        def select_one(directory, pattern):
            matches = sorted(Path(directory).glob(pattern))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one file matching {directory}/{pattern}; found {len(matches)}: {matches}")
            return str(matches[0].resolve())

        mni_reference = get_package_path("data", "standard", "JHU", "JHU-ICBM-labels-1mm.nii.gz")
        lesion_files = {}
        individual_inputs = {}
        if "individual" in self.methods:
            if self.individual_connectome_source == "dsistudio":
                reference_dir = derivatives.joinpath("qsiprep", *visit, "dwi")
                tract_dir = derivatives.joinpath("qsirecon-DSIStudio", *visit, "dwi")
                tractogram = select_one(tract_dir, f"{prefix}_*_space-*_desc-preproc_streamlines.trk.gz")
            else:
                reference_dir = derivatives.joinpath("dwi_pipeline", *visit)
                tractogram = select_one(reference_dir / "connectome", f"{prefix}_*_space-*_streamlines.tck")
            dwi_space = re.search(r"_space-([^_]+)_", Path(tractogram).name).group(1)
            individual_inputs = {
                "tractogram_file": tractogram,
                "connectome_source": self.individual_connectome_source,
                "mni_reference": mni_reference,
                "dwi_reference": select_one(reference_dir, f"{prefix}_*_space-{dwi_space}_dwiref.nii.gz"),
                "t1w_reference": select_one(xfm_dir, f"{prefix}_acq-highres_desc-brain_T1w.nii.gz"),
                "mni_to_t1w_warp": select_one(xfm_dir, f"{prefix}_from-{mni_space}_to-T1w_warp.nii.gz"),
                "t1w_to_mni_warp": select_one(xfm_dir, f"{prefix}_from-T1w_to-{mni_space}_warp.nii.gz"),
                "t1w_to_dwi_matrix": select_one(xfm_dir, f"{prefix}_from-T1w_to-{dwi_space}_xfm.mat"),
                "dwi_to_t1w_matrix": select_one(xfm_dir, f"{prefix}_from-{dwi_space}_to-T1w_xfm.mat"),
                "use_freesurfer_transform": self.use_freesurfer_transform,
            }
            if self.use_freesurfer_transform:
                freesurfer_dir = getattr(self.session, "freesurfer_dir", None)
                if not freesurfer_dir:
                    raise FileNotFoundError("use_freesurfer_transform requires this session's FreeSurfer output")
                freesurfer_dir = Path(freesurfer_dir).resolve()
                individual_inputs.update(freesurfer_subjects_dir=str(freesurfer_dir.parent),
                                         freesurfer_subject_id=freesurfer_dir.name)
            if self.individual_connectome_source == "dsistudio":
                anatomy_dir = derivatives.joinpath("dwi_pipeline", *visit, "anat")
                individual_inputs.update({
                    "anatomical_segmentation": select_one(anatomy_dir, f"{prefix}_space-{dwi_space}_aparcaseg.nii.gz"),
                    "cortical_gm_mask": select_one(anatomy_dir, f"{prefix}_space-{dwi_space}_label-corticalGM_mask.nii.gz"),
                    "brain_mask": select_one(reference_dir, f"{prefix}_*_space-{dwi_space}_desc-brain_mask.nii.gz"),
                })

        for method, folder, pattern, space, reference in (
                ("normative", self.mni_lesion_mask, self.use_which_mni_lesion_mask, mni_space, mni_reference),
                ("individual", self.t1w_lesion_mask, self.use_which_t1w_lesion_mask,
                 "T1w", individual_inputs.get("t1w_reference"))):
            if method not in self.methods:
                continue
            lesion_dir = derivatives.joinpath(folder, *visit)
            lesions = [path for path in lesion_dir.rglob("*")
                       if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
                       and pattern in path.name and f"_space-{space}_" in path.name]
            if len(lesions) != 1:
                raise FileNotFoundError(f"Expected one {space} lesion matching '{pattern}' in {lesion_dir}; found {len(lesions)}")
            lesion_files[method] = str(lesions[0].resolve())
            lesion_image, reference_image = nib.load(lesion_files[method]), nib.load(reference)
            if (lesion_image.ndim != 3 or lesion_image.shape != reference_image.shape[:3]
                    or not np.allclose(lesion_image.affine, reference_image.affine, rtol=0, atol=1e-3)):
                raise ValueError(f"{method} lesion must match the {space} reference grid: {lesions[0]}")

        atlas_dir = Path(get_package_path("data", "atlas", "nemo"))
        atlas_files = [select_one(atlas_dir, f"tpl-MNI152NLin6Asym_res-1_atlas-{name}_dseg.nii.gz") for name in atlas_names]
        atlas_labels = [select_one(atlas_dir, f"tpl-MNI152NLin6Asym_res-1_atlas-{name}_dseg.tsv") for name in atlas_names]
        common_inputs = dict(atlas_files=atlas_files, atlas_label_files=atlas_labels, atlas_names=atlas_names)
        workflow = Workflow(name="disconnection_workflow")
        workflow.base_dir = str(derivatives.joinpath("workflows", *visit))

        def model_outputs(lesion_file, method, model):
            entities = {"space": mni_space, "model": model.replace("_", "")}
            traversal_desc, endpoint_desc, region_desc = "disconnection", "chacovolEndpoint", "chacovol"
            suffix = "probability"
            if model == "absolute_length":
                traversal_desc = region_desc = "meanAffectedLength"
                endpoint_desc, suffix = "meanAffectedEndpointLength", "map"
            elif model == "fractional_length":
                traversal_desc = region_desc = "meanAffectedFraction"
                endpoint_desc = "meanAffectedEndpointFraction"
            output_dir = Path(self.output_path) / "structural" / method / model
            outputs = {
                "output_disconnection_probability": str(output_dir / rename_bids_file(
                    lesion_file, {**entities, "desc": traversal_desc}, suffix, ".nii.gz")),
                "output_chacovol_endpoint_voxelwise": str(output_dir / rename_bids_file(
                    lesion_file, {**entities, "desc": endpoint_desc}, suffix, ".nii.gz")),
                "output_qc": str(output_dir / rename_bids_file(
                    lesion_file, {**entities, "desc": traversal_desc}, "qc", ".png")),
            }
            for hemisphere, side in (("lh", "L"), ("rh", "R")):
                filename = rename_bids_file(
                    lesion_file, {**entities, "space": "fsaverage", "den": "164k", "hemi": side,
                                  "desc": endpoint_desc}, "map", ".func.gii")
                outputs[f"output_chacovol_endpoint_surface_{hemisphere}"] = str(output_dir / filename)
            outputs["output_chacovol_regionwise_csvs"] = [
                str(output_dir / rename_bids_file(
                    lesion_file, {**entities, "atlas": name, "desc": region_desc}, "regions", ".csv"))
                for name in atlas_names]
            network_desc = ("meanAffectedLength" if model == "absolute_length" else
                            "meanAffectedFraction" if model == "fractional_length" else "chacoconn")
            outputs["output_chacoconn_regionwise_csvs"] = [
                str(output_dir / rename_bids_file(
                    lesion_file, {**entities, "atlas": name, "desc": network_desc}, "connectivity", ".csv"))
                for name in atlas_names]
            jhu_desc = ("meanAffectedLengthVoxelMean" if model == "absolute_length" else
                        "meanAffectedFractionVoxelMean" if model == "fractional_length" else "disconnectionVoxelMean")
            outputs["output_jhu_regionwise_csv"] = str(output_dir / rename_bids_file(
                lesion_file, {**entities, "atlas": "JHU", "desc": jhu_desc}, "regions", ".csv"))
            return outputs

        for method in self.methods:
            lesion_file = lesion_files[method]
            outputs_by_model = [model_outputs(lesion_file, method, model) for model in self.structural_methods]
            if method == "normative":
                interface = IITDisconnection(
                    **common_inputs, lesion_files=[lesion_file], damage_models=list(self.structural_methods))
                node_name = (f"iit_disconnection_{self.structural_methods[0]}"
                             if len(self.structural_methods) == 1 else "iit_disconnection")
                node = Node(interface, name=node_name)
                scalar_fields = (
                    "output_disconnection_probability", "output_chacovol_endpoint_voxelwise",
                    "output_chacovol_endpoint_surface_lh", "output_chacovol_endpoint_surface_rh",
                    "output_jhu_regionwise_csv", "output_qc")
                for field in scalar_fields:
                    setattr(node.inputs, field, [outputs[field] for outputs in outputs_by_model])
                for field in ("output_chacovol_regionwise_csvs", "output_chacoconn_regionwise_csvs"):
                    setattr(node.inputs, field, [path for outputs in outputs_by_model for path in outputs[field]])
                workflow.add_nodes([node])
                continue
            for model, outputs in zip(self.structural_methods, outputs_by_model):
                interface = IndividualDisconnection(
                    **common_inputs, **individual_inputs, lesion_file=lesion_file, damage_model=model)
                node = Node(interface, name=f"individual_disconnection_{model}")
                for field, value in outputs.items():
                    setattr(node.inputs, field, value)
                workflow.add_nodes([node])
        return workflow

    def extract_results(self):
        """Collect regional values and network edges across subjects and sessions."""
        if not self.extract_from:
            raise ValueError("extract_from must specify the disconnection derivatives directory")
        source = Path(self.extract_from).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        tables = {"regions": [], "connectivity": []}
        for kind in tables:
            for path in sorted(source.rglob(f"*_{kind}.csv")):
                modality, connectome, model = path.parents[2].name, path.parents[1].name, path.parent.name
                selected_models = self.structural_methods if modality == "structural" else self.functional_methods
                if modality not in {"structural", "functional"} or connectome not in self.methods or model not in selected_models:
                    continue
                entities = dict(re.findall(r"(?:^|_)(sub|ses|space|atlas|model|label|desc)-([^_]+)", path.name))
                metadata = {"Subject": entities.get("sub", ""), "Session": entities.get("ses", ""),
                            "Modality": modality, "Connectome": connectome, "Method": model,
                            "Atlas": entities.get("atlas", ""), "SourceFile": str(path)}
                frame = pd.read_csv(path, encoding="utf-8-sig")
                if kind == "regions":
                    if not {"label_id", "region_name", "disconnection_value"}.issubset(frame.columns):
                        raise ValueError(f"Invalid regional disconnection CSV: {path}")
                else:
                    if frame.columns[0] != "region" or list(frame["region"]) != list(frame.columns[1:]):
                        raise ValueError(f"Invalid ChaCo connectivity matrix: {path}")
                    frame = frame.rename(columns={"region": "Region1"}).melt(id_vars="Region1", var_name="Region2", value_name="disconnection_value")
                for column, value in reversed(list(metadata.items())):
                    frame.insert(0, column, value)
                tables[kind].append(frame)
        if not any(tables.values()):
            raise FileNotFoundError(f"No disconnection result CSVs found in {source}")
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        outputs = {}
        for kind, frames in tables.items():
            if frames:
                output = Path(self.output_path) / f"disconnection_{kind}_summary.csv"
                pd.concat(frames, ignore_index=True).to_csv(output, index=False)
                outputs[kind] = str(output)
                print(f"Saved disconnection {kind} summary: {output}")
        return outputs


__all__ = ["DisconnectionPipeline"]
