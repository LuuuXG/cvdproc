"""Normative IIT and subject-specific lesion disconnection workflow.

The pipeline locates one MNI152NLin6ASym lesion mask in a BIDS derivatives
folder and runs all four lesion-connectivity models exposed by :class:`IITDisconnection`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from nipype import Node, Workflow

from cvdproc.bids_data.rename_bids_file import rename_bids_file
from cvdproc.config.paths import get_package_path
from cvdproc.pipelines.dmri.disconnection.disconnection_nipype import IITDisconnection
from cvdproc.pipelines.dmri.disconnection.individual_disconnection_nipype import IndividualDisconnection


class DisconnectionPipeline:
    """Run four normative and four individual lesion-connectivity models."""

    DAMAGE_MODELS = ("any_hit", "streamline_mean", "total_length_ratio", "lesion_voxel_mean")
    MODEL_LABELS = {
        "any_hit": "anyhit",
        "streamline_mean": "streamlinemean",
        "total_length_ratio": "totallengthratio",
        "lesion_voxel_mean": "lesionvoxelmean",
    }
    MNI_SPACE = "MNI152NLin6ASym"
    MNI_SHAPE = (182, 218, 182)
    ATLASES = (
        ("AAL116", "tpl-MNI152NLin6Asym_res-1_atlas-AAL116_dseg"),
        ("Shen268", "tpl-MNI152NLin6Asym_res-1_atlas-Shen268_dseg"),
        ("FreeSurfer86Avg", "tpl-MNI152NLin6Asym_res-1_atlas-FreeSurfer86Avg_dseg"),
    )

    def __init__(
        self,
        subject,
        session,
        output_path,
        lesion_mask: str = "lesion_mask",
        use_which_lesion_mask: str | None = None,
        force_lesion_probability_one: bool = True,
        lesion_threshold: float = 0.0,
        atlas_assignment_radius_mm: float = 2.0,
        individual_connectome_source: str = "qsirecon",
        mrtrix_bin_dir: str | None = None,
        nthreads: int = 0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        lesion_mask
            Derivatives folder containing the lesion, for example
            ``lesion_mask`` for ``derivatives/lesion_mask``.
        use_which_lesion_mask
            Text used to select exactly one NIfTI lesion file. The selected
            filename must contain ``space-MNI152NLin6ASym``.
        force_lesion_probability_one
            Force lesion voxels to one in the ``any_hit`` disconnection map.
            The three non-binary models are not altered by this option.
        individual_connectome_source
            ``qsirecon`` selects a QSIRECON-DSIStudio ``.trk.gz`` file;
            ``mrtrix3`` selects a dwi_pipeline ``connectome/*.tck`` file.
            The DWI space is read from the tractogram's BIDS ``space`` entity.
        """
        self.subject = subject
        self.session = session
        self.output_path = os.path.abspath(output_path)
        self.lesion_mask = lesion_mask
        self.use_which_lesion_mask = use_which_lesion_mask
        self.force_lesion_probability_one = force_lesion_probability_one
        self.lesion_threshold = lesion_threshold
        self.atlas_assignment_radius_mm = atlas_assignment_radius_mm
        self.individual_connectome_source = individual_connectome_source.lower()
        self.mrtrix_bin_dir = mrtrix_bin_dir
        self.nthreads = nthreads

        if not self.lesion_mask:
            raise ValueError("lesion_mask must name a derivatives folder")
        if not self.use_which_lesion_mask:
            raise ValueError("use_which_lesion_mask must be provided")
        if self.individual_connectome_source not in {"mrtrix3", "qsirecon"}:
            raise ValueError(
                "individual_connectome_source must be 'mrtrix3' or 'qsirecon'"
            )

    def _visit_parts(self):
        parts = [f"sub-{self.subject.subject_id}"]
        if self.session is not None:
            parts.append(f"ses-{self.session.session_id}")
        return parts

    def _lesion_directory(self):
        return Path(self.subject.bids_dir, "derivatives", self.lesion_mask, *self._visit_parts())

    def _find_lesion(self):
        lesion_dir = self._lesion_directory()
        if not lesion_dir.is_dir():
            raise FileNotFoundError(f"Lesion derivatives directory does not exist: {lesion_dir}")

        name_matches = sorted(
            path for path in lesion_dir.rglob("*")
            if path.is_file()
            and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))
            and self.use_which_lesion_mask in path.name
        )
        candidates = [
            path for path in name_matches
            if re.search(
                rf"(?:^|_)space-{re.escape(self.MNI_SPACE)}(?:_|$)", path.name
            )
        ]
        if len(candidates) != 1:
            matched = "\n".join(f"  - {path}" for path in name_matches) or "  (none)"
            raise FileNotFoundError(
                f"Expected exactly one space-{self.MNI_SPACE} lesion NIfTI in "
                f"{lesion_dir} containing '{self.use_which_lesion_mask}', found "
                f"{len(candidates)}. All name matches:\n{matched}"
            )
        self._validate_mni_lesion(candidates[0])
        return str(candidates[0].resolve())

    def _validate_mni_lesion(self, lesion_file):
        space_match = re.search(r"(?:^|_)space-([^_]+)(?:_|$)", lesion_file.name)
        space = space_match.group(1) if space_match else None
        if space != self.MNI_SPACE:
            raise ValueError(
                f"Lesion must have BIDS entity space-{self.MNI_SPACE}; got "
                f"{f'space-{space}' if space else 'no space entity'} in {lesion_file.name}"
            )

        lesion_image = nib.load(str(lesion_file))
        if len(lesion_image.shape) != 3 or lesion_image.shape != self.MNI_SHAPE:
            raise ValueError(
                f"Lesion must be a 3D 1 mm {self.MNI_SPACE} image with shape "
                f"{self.MNI_SHAPE}; got {lesion_image.shape}"
            )

        warp_file = Path(
            get_package_path(
                "data", "standard", "MNI152", "custom",
                "from-MNI152NLin6ASym_to-IIT_warp.nii.gz",
            )
        )
        if not warp_file.is_file():
            raise FileNotFoundError(f"Required MNI-to-IIT warp is missing: {warp_file}")
        reference_image = nib.load(str(warp_file))
        if reference_image.shape[:3] != self.MNI_SHAPE or not np.allclose(
            lesion_image.affine, reference_image.affine, rtol=0.0, atol=1e-3
        ):
            raise ValueError(
                f"Lesion grid/affine does not match the required 1 mm {self.MNI_SPACE} grid: "
                f"{lesion_file}"
            )

    def check_data_requirements(self):
        """Return whether lesion, individual connectome and transforms exist."""
        try:
            self._find_lesion()
            self._individual_inputs()
            self._atlas_inputs()
        except (FileNotFoundError, ValueError, OSError):
            return False
        return True

    @staticmethod
    def _one(directory, pattern, label):
        matches = sorted(Path(directory).glob(pattern)) if directory and Path(directory).is_dir() else []
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {label} matching {pattern} in {directory}; found {len(matches)}"
            )
        return str(matches[0].resolve())

    def _individual_inputs(self):
        prefix = "_".join(self._visit_parts())
        xfm_dir = Path(self.subject.bids_dir, "derivatives", "xfm", *self._visit_parts())
        derivatives = Path(self.subject.bids_dir, "derivatives")
        if self.individual_connectome_source == "qsirecon":
            tract_dir = derivatives / "qsirecon-DSIStudio" / Path(*self._visit_parts()) / "dwi"
            reference_dir = derivatives / "qsiprep" / Path(*self._visit_parts()) / "dwi"
            tractogram = self._one(
                tract_dir,
                f"{prefix}_*_space-*_desc-preproc_streamlines.trk.gz",
                "QSIRECON-DSIStudio tractogram",
            )
        else:
            reference_dir = derivatives / "dwi_pipeline" / Path(*self._visit_parts())
            tract_dir = reference_dir / "connectome"
            tractogram = self._one(
                tract_dir,
                f"{prefix}_*_space-*_streamlines.tck",
                "MRtrix3 tractogram",
            )

        space_match = re.search(r"(?:^|_)space-([^_]+)(?:_|$)", Path(tractogram).name)
        if space_match is None:
            raise ValueError(f"Tractogram has no BIDS space entity: {tractogram}")
        dwi_space = space_match.group(1)
        dwi_reference = self._one(
            reference_dir,
            f"{prefix}_*_space-{dwi_space}_dwiref.nii.gz",
            f"space-{dwi_space} DWI reference",
        )
        inputs = {
            "_dwi_space": dwi_space,
            "tractogram_file": tractogram,
            "dwi_reference": dwi_reference,
            "t1w_reference": self._one(xfm_dir, f"{prefix}_acq-highres_desc-brain_T1w.nii.gz", "T1w reference"),
            "mni_to_t1w_warp": self._one(xfm_dir, f"{prefix}_from-MNI152NLin6ASym_to-T1w_warp.nii.gz", "MNI-to-T1w warp"),
            "t1w_to_mni_warp": self._one(xfm_dir, f"{prefix}_from-T1w_to-MNI152NLin6ASym_warp.nii.gz", "T1w-to-MNI warp"),
            "t1w_to_dwi_matrix": self._one(
                xfm_dir,
                f"{prefix}_from-T1w_to-{dwi_space}_xfm.mat",
                f"T1w-to-{dwi_space} matrix",
            ),
            "dwi_to_t1w_matrix": self._one(
                xfm_dir,
                f"{prefix}_from-{dwi_space}_to-T1w_xfm.mat",
                f"{dwi_space}-to-T1w matrix",
            ),
        }
        if self.individual_connectome_source == "qsirecon":
            anatomy_dir = derivatives / "dwi_pipeline" / Path(*self._visit_parts()) / "anat"
            inputs.update({
                "anatomical_segmentation": self._one(
                    anatomy_dir,
                    f"{prefix}_space-{dwi_space}_aparcaseg.nii.gz",
                    f"space-{dwi_space} aparc+aseg",
                ),
                "cortical_gm_mask": self._one(
                    anatomy_dir,
                    f"{prefix}_space-{dwi_space}_label-corticalGM_mask.nii.gz",
                    f"space-{dwi_space} cortical GM mask",
                ),
                "brain_mask": self._one(
                    reference_dir,
                    f"{prefix}_*_space-{dwi_space}_desc-brain_mask.nii.gz",
                    f"space-{dwi_space} DWI brain mask",
                ),
            })
        return inputs

    @classmethod
    def _atlas_inputs(cls):
        atlas_dir = Path(get_package_path("data", "atlas", "nemo"))
        files, labels, names = [], [], []
        for name, stem in cls.ATLASES:
            nifti, tsv = atlas_dir / f"{stem}.nii.gz", atlas_dir / f"{stem}.tsv"
            if not nifti.is_file() or not tsv.is_file():
                raise FileNotFoundError(f"Missing BIDS atlas pair for {name}: {nifti}, {tsv}")
            files.append(str(nifti.resolve()))
            labels.append(str(tsv.resolve()))
            names.append(name)
        return files, labels, names

    @staticmethod
    def _output_names(lesion_file, model_label, damage_model, space=None):
        common = {
            "space": space or DisconnectionPipeline.MNI_SPACE,
            "model": model_label,
        }
        if damage_model == "lesion_voxel_mean":
            traversal_desc, endpoint_desc, region_desc, suffix = (
                "lesionNormalizedConnectivity",
                "lesionNormalizedEndpointConnectivity",
                "lesionNormalizedConnectivity",
                "score",
            )
        else:
            traversal_desc, endpoint_desc, region_desc, suffix = (
                "disconnection", "chacovolEndpoint", "chacovol", "probability"
            )
        return {
            "output_disconnection_probability": rename_bids_file(
                lesion_file, {**common, "desc": traversal_desc}, suffix, ".nii.gz"
            ),
            "output_chacovol_endpoint_voxelwise": rename_bids_file(
                lesion_file, {**common, "desc": endpoint_desc}, suffix, ".nii.gz"
            ),
            "output_qc": rename_bids_file(
                lesion_file, {**common, "desc": traversal_desc}, "qc", ".png"
            ),
            "region_desc": region_desc,
        }

    @classmethod
    def _region_names(cls, lesion_file, model_label, region_desc):
        return [
            rename_bids_file(
                lesion_file,
                {"space": cls.MNI_SPACE, "model": model_label, "atlas": atlas_name, "desc": region_desc},
                "regions", ".csv",
            )
            for atlas_name, _ in cls.ATLASES
        ]

    @classmethod
    def _network_names(cls, lesion_file, model_label):
        return [
            rename_bids_file(
                lesion_file,
                {"space": cls.MNI_SPACE, "model": model_label, "atlas": atlas_name,
                 "desc": "chacoconn"},
                "connectivity", ".csv",
            )
            for atlas_name, _ in cls.ATLASES
        ]

    def create_workflow(self):
        lesion_file = self._find_lesion()
        individual_inputs = self._individual_inputs()
        individual_space = individual_inputs.pop("_dwi_space")
        atlas_files, atlas_label_files, atlas_names = self._atlas_inputs()
        os.makedirs(self.output_path, exist_ok=True)

        workflow = Workflow(name="disconnection_workflow")
        workflow.base_dir = os.path.join(
            self.subject.bids_dir, "derivatives", "workflows", *self._visit_parts()
        )

        nodes = []
        for damage_model in self.DAMAGE_MODELS:
            method_dir = os.path.join(self.output_path, "normative_disconnection", damage_model)
            os.makedirs(method_dir, exist_ok=True)
            model_label = self.MODEL_LABELS[damage_model]

            node = Node(
                IITDisconnection(),
                name=f"iit_disconnection_{damage_model}",
            )
            node.inputs.lesion_file = lesion_file
            node.inputs.damage_model = damage_model
            node.inputs.force_lesion_probability_one = self.force_lesion_probability_one
            node.inputs.lesion_threshold = self.lesion_threshold
            if self.mrtrix_bin_dir:
                node.inputs.mrtrix_bin_dir = self.mrtrix_bin_dir
            node.inputs.nthreads = self.nthreads
            node.inputs.atlas_assignment_radius_mm = self.atlas_assignment_radius_mm
            node.inputs.atlas_files = atlas_files
            node.inputs.atlas_label_files = atlas_label_files
            node.inputs.atlas_names = atlas_names
            names = self._output_names(lesion_file, model_label, damage_model)
            region_desc = names.pop("region_desc")
            for field, filename in names.items():
                setattr(node.inputs, field, os.path.join(method_dir, filename))
            node.inputs.output_chacovol_regionwise_csvs = [
                os.path.join(method_dir, name)
                for name in self._region_names(lesion_file, model_label, region_desc)
            ]
            if damage_model != "lesion_voxel_mean":
                node.inputs.output_chacoconn_regionwise_csvs = [
                    os.path.join(method_dir, name)
                    for name in self._network_names(lesion_file, model_label)
                ]
            nodes.append(node)

        individual = Node(IndividualDisconnection(), name="individual_disconnection")
        individual.inputs.lesion_file = lesion_file
        for field, value in individual_inputs.items():
            setattr(individual.inputs, field, value)
        individual.inputs.atlas_files = atlas_files
        individual.inputs.atlas_label_files = atlas_label_files
        individual.inputs.atlas_names = atlas_names
        individual.inputs.force_lesion_probability_one = self.force_lesion_probability_one
        individual.inputs.lesion_threshold = self.lesion_threshold
        if self.mrtrix_bin_dir:
            individual.inputs.mrtrix_bin_dir = self.mrtrix_bin_dir
        individual.inputs.nthreads = self.nthreads
        individual.inputs.atlas_assignment_radius_mm = self.atlas_assignment_radius_mm
        traversal_outputs, endpoint_outputs = [], []
        native_traversal_outputs, native_endpoint_outputs = [], []
        region_outputs, network_outputs, qc_outputs = [], [], []
        for damage_model in self.DAMAGE_MODELS:
            method_dir = os.path.join(self.output_path, "indivisual_disconnection", damage_model)
            os.makedirs(method_dir, exist_ok=True)
            model_label = self.MODEL_LABELS[damage_model]
            names = self._output_names(lesion_file, model_label, damage_model)
            region_desc = names.pop("region_desc")
            traversal_outputs.append(os.path.join(method_dir, names["output_disconnection_probability"]))
            endpoint_outputs.append(os.path.join(method_dir, names["output_chacovol_endpoint_voxelwise"]))
            native_names = self._output_names(
                lesion_file, model_label, damage_model, space=individual_space
            )
            native_traversal_outputs.append(
                os.path.join(method_dir, native_names["output_disconnection_probability"])
            )
            native_endpoint_outputs.append(
                os.path.join(method_dir, native_names["output_chacovol_endpoint_voxelwise"])
            )
            qc_outputs.append(os.path.join(method_dir, names["output_qc"]))
            region_outputs.extend(
                os.path.join(method_dir, name)
                for name in self._region_names(lesion_file, model_label, region_desc)
            )
            if damage_model != "lesion_voxel_mean":
                network_outputs.extend(
                    os.path.join(method_dir, name)
                    for name in self._network_names(lesion_file, model_label)
                )
        individual.inputs.output_disconnection_probabilities = traversal_outputs
        individual.inputs.output_chacovol_endpoint_voxelwise = endpoint_outputs
        individual.inputs.output_native_disconnection_probabilities = native_traversal_outputs
        individual.inputs.output_native_chacovol_endpoint_voxelwise = native_endpoint_outputs
        individual.inputs.output_chacovol_regionwise_csvs = region_outputs
        individual.inputs.output_chacoconn_regionwise_csvs = network_outputs
        individual.inputs.output_qcs = qc_outputs
        nodes.append(individual)

        workflow.add_nodes(nodes)
        return workflow


__all__ = ["DisconnectionPipeline"]
