"""Nipype interfaces for normative and individual lesion disconnectome analysis.

IIT analysis supports cache-efficient lesion batches and never reads tractograms.
Individual analysis uses the existing MRtrix3 installation.
"""
from __future__ import annotations

import csv
import gzip
import struct
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, Directory, File,
                                    InputMultiPath, OutputMultiPath, TraitedSpec, isdefined, traits)
from scipy import sparse
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates

from nipype.interfaces.fsl import FLIRT
from nipype.interfaces.mrtrix3 import ComputeTDI
from cvdproc.pipelines.common.register import MRIConvertApplyWarp
from cvdproc.pipelines.common.image_calc import CalculateScalarMaps
from cvdproc.pipelines.dmri.mrtrix3.tcksample_nipype import TckSampleCommand
from cvdproc.pipelines.smri.freesurfer.utils import MRIvol2surf
from cvdproc.config.paths import LH_MEDIAL_WALL_TXT, RH_MEDIAL_WALL_TXT, get_package_path


IIT_DIR = Path(get_package_path("data", "standard", "IIT"))
MNI_CUSTOM_DIR = Path(get_package_path("data", "standard", "MNI152", "custom"))
IIT_T1 = IIT_DIR / "IITmean_t1.nii.gz"
IIT_T1_256 = IIT_DIR / "IITmean_t1_256.nii.gz"
IIT_OPERATOR_CACHE = IIT_DIR / "weighted_disconnection_operator" / "10m"
CACHE_FORMAT = "cvdproc-iit-weighted-disconnection-v1"
IIT_LESION_BATCH_SIZE = 8
IIT_TO_MNI_WARP = MNI_CUSTOM_DIR / "from-IIT_to-MNI152NLin6Asym_warp.nii.gz"
MNI_TO_IIT_WARP = MNI_CUSTOM_DIR / "from-MNI152NLin6Asym_to-IIT_warp.nii.gz"
JHU_ATLAS = Path(get_package_path("data", "standard", "JHU", "JHU-ICBM-labels-1mm.nii.gz"))
JHU_LABELS = Path(get_package_path("data", "standard", "JHU", "JHU-labels.xml"))


class DisconnectionInputSpec(BaseInterfaceInputSpec):
    lesion_file = File(exists=True, mandatory=True, desc="MNI152NLin6Asym 1 mm nonnegative lesion mask or weights")
    output_disconnection_probability = File(
        mandatory=True, desc="Output MNI traversal map for the selected streamline damage model"
    )
    output_chacovol_endpoint_voxelwise = File(
        mandatory=True, desc="Output MNI endpoint map for the selected streamline damage model"
    )
    atlas_files = InputMultiPath(File(exists=True), mandatory=True, desc="MNI label atlases")
    atlas_label_files = InputMultiPath(File(exists=True), mandatory=True, desc="BIDS atlas label TSV files")
    atlas_names = traits.List(traits.Str, mandatory=True, desc="Atlas entity labels")
    output_chacovol_regionwise_csvs = InputMultiPath(File(), mandatory=True, desc="One regionwise CSV per atlas")
    output_chacoconn_regionwise_csvs = InputMultiPath(File(), mandatory=True, desc="One pairwise ChaCo matrix per atlas")
    output_chacovol_endpoint_surface_lh = File(mandatory=True, desc="Left fsaverage 164k endpoint map (.func.gii)")
    output_chacovol_endpoint_surface_rh = File(mandatory=True, desc="Right fsaverage 164k endpoint map (.func.gii)")
    output_jhu_regionwise_csv = File(mandatory=True, desc="JHU ROI means of the MNI traversal map, including zero voxels")
    output_qc = File(mandatory=True, desc="Output QC image")
    damage_model = traits.Enum("any_hit", "absolute_length", "fractional_length", usedefault=True,
                               desc="Saturated hit, affected length, or affected-length fraction")
    atlas_assignment_radius_mm = traits.Float(2.0, usedefault=True)
    overwrite = traits.Bool(True, usedefault=True)


class DisconnectionOutputSpec(TraitedSpec):
    disconnection_probability = File(exists=True)
    chacovol_endpoint_voxelwise = File(exists=True)
    chacovol_regionwise_csvs = OutputMultiPath(File(exists=True))
    chacoconn_regionwise_csvs = OutputMultiPath(File(exists=True))
    chacovol_endpoint_surface_lh = File(exists=True)
    chacovol_endpoint_surface_rh = File(exists=True)
    jhu_regionwise_csv = File(exists=True)
    qc = File(exists=True)


class IITDisconnectionInputSpec(BaseInterfaceInputSpec):
    lesion_files = InputMultiPath(File(exists=True), mandatory=True, desc="MNI152NLin6Asym 1 mm lesion masks or weights")
    output_disconnection_probability = InputMultiPath(File(), mandatory=True)
    output_chacovol_endpoint_voxelwise = InputMultiPath(File(), mandatory=True)
    atlas_files = InputMultiPath(File(exists=True), mandatory=True, desc="MNI label atlases")
    atlas_label_files = InputMultiPath(File(exists=True), mandatory=True, desc="BIDS atlas label TSV files")
    atlas_names = traits.List(traits.Str, mandatory=True, desc="Atlas entity labels")
    output_chacovol_regionwise_csvs = InputMultiPath(File(), mandatory=True, desc="Subject-major ROI CSVs")
    output_chacoconn_regionwise_csvs = InputMultiPath(File(), mandatory=True, desc="Subject-major network CSVs")
    output_chacovol_endpoint_surface_lh = InputMultiPath(File(), mandatory=True)
    output_chacovol_endpoint_surface_rh = InputMultiPath(File(), mandatory=True)
    output_jhu_regionwise_csv = InputMultiPath(File(), mandatory=True)
    output_qc = InputMultiPath(File(), mandatory=True)
    output_deduplicated_voxel_ratio = InputMultiPath(
        File(), desc="Optional MNI maps of excess distinct lesion-voxel hits divided by all distinct hits")
    damage_model = traits.Enum("any_hit", "absolute_length", "fractional_length", usedefault=True)
    damage_models = traits.List(traits.Enum("any_hit", "absolute_length", "fractional_length"),
                                desc="Models computed together; output paths use lesion-major, model-major order")
    atlas_assignment_radius_mm = traits.Float(2.0, usedefault=True)
    overwrite = traits.Bool(True, usedefault=True)


class IITDisconnectionOutputSpec(TraitedSpec):
    disconnection_probability = OutputMultiPath(File(exists=True))
    chacovol_endpoint_voxelwise = OutputMultiPath(File(exists=True))
    chacovol_regionwise_csvs = OutputMultiPath(File(exists=True))
    chacoconn_regionwise_csvs = OutputMultiPath(File(exists=True))
    chacovol_endpoint_surface_lh = OutputMultiPath(File(exists=True))
    chacovol_endpoint_surface_rh = OutputMultiPath(File(exists=True))
    jhu_regionwise_csv = OutputMultiPath(File(exists=True))
    qc = OutputMultiPath(File(exists=True))
    deduplicated_voxel_ratio = OutputMultiPath(File(exists=True))


def _save_like(data, reference, path: Path, dtype=np.float32, description=None):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    if description is not None:
        header["descrip"] = description
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    image.set_qform(qform, int(qcode or 0))
    image.set_sform(sform, int(scode or 0))
    nib.save(image, str(path))


def _warp_pullback(source, field, target, order, dtype, slab=16):
    if field.shape[:3] != target.shape[:3] or field.shape[-1] != 3:
        raise ValueError(f"Warp grid {field.shape} does not match target {target.shape[:3]}")
    source_data = np.asanyarray(source.dataobj)
    displacement = np.asanyarray(field.dataobj)
    output = np.zeros(target.shape[:3], dtype=dtype)
    inverse_affine = np.linalg.inv(source.affine)
    nx, ny, nz = target.shape[:3]
    x = np.arange(nx, dtype=np.float32)[:, None, None]
    y = np.arange(ny, dtype=np.float32)[None, :, None]
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        z = np.arange(z0, z1, dtype=np.float32)[None, None, :]
        i, j, k = np.broadcast_arrays(x, y, z)
        voxel = np.stack((i, j, k)).reshape(3, -1)
        world = target.affine[:3, :3] @ voxel + target.affine[:3, 3:4]
        delta = np.moveaxis(displacement[:, :, z0:z1, :], -1, 0).reshape(3, -1)
        source_voxel = inverse_affine[:3, :3] @ (world + delta) + inverse_affine[:3, 3:4]
        values = map_coordinates(source_data, source_voxel, order=order, mode="constant", cval=0, prefilter=False)
        output[:, :, z0:z1] = values.reshape(nx, ny, z1 - z0)
    return output


def _to_iit256(data):
    if data.shape != (182, 218, 182):
        raise ValueError(f"Expected IIT grid (182, 218, 182), got {data.shape}")
    output = np.zeros((256, 256, 256), dtype=data.dtype)
    output[37:219, 19:237, 37:219] = data[::-1, :, :]
    return output


def _from_iit256(data):
    if data.shape != (256, 256, 256):
        raise ValueError(f"Expected IIT-256 grid, got {data.shape}")
    return data[37:219, 19:237, 37:219][::-1, :, :]


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros(numerator.shape, np.float32), where=denominator > 0)


def _fractional_length_weights(affected_lengths, streamline_lengths):
    """Normalize cached affected lengths by total cached streamline lengths."""
    affected_lengths = np.asarray(affected_lengths).ravel()
    streamline_lengths = np.asarray(streamline_lengths).ravel()
    return np.divide(affected_lengths, streamline_lengths, out=np.zeros_like(affected_lengths),
                     where=streamline_lengths > 0)


def _write_csv(rows, path):
    fields = ("label_id", "region_name", "total_endpoints", "affected_endpoints", "disconnection_value",
              "damage_model", "denominator_unit", "numerator_unit", "normalization_denominator",
              "normalization_unit")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_bids_atlas_labels(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or "index" not in reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError(f"Atlas TSV must contain index and name columns: {path}")
        return {int(row["index"]): row["name"] for row in reader if row.get("index")}


def _atlas_region_values(atlas, endpoint_denominator, endpoint_numerator, names,
                         damage_model, lesion_voxel_count):
    labels = sorted(int(value) for value in np.unique(atlas) if value > 0)
    rows = []
    region_map = np.zeros(atlas.shape, np.float32)
    units = {
        "any_hit": ("streamlines", "streamlines"),
        "absolute_length": ("streamlines", "streamline_mm"),
        "fractional_length": ("streamlines", "sum_of_streamline_lesion_fractions"),
        "streamline_mean": ("streamlines", "sum_of_streamline_lesion_fractions"),
        "total_length_ratio": ("streamline_mm", "lesion_mm"),
        "lesion_voxel_mean": ("streamlines", "streamline_lesion_voxel_intersections"),
    }
    denominator_unit, numerator_unit = units[damage_model]
    for label in labels:
        mask = atlas == label
        denominator_value = float(np.asarray(endpoint_denominator[mask], np.float64).sum())
        numerator_value = float(np.asarray(endpoint_numerator[mask], np.float64).sum())
        if damage_model == "lesion_voxel_mean":
            value = numerator_value / float(lesion_voxel_count)
            normalization_denominator = float(lesion_voxel_count)
            normalization_unit = "lesion_voxels"
        else:
            value = numerator_value / denominator_value if denominator_value > 0 else 0.0
            normalization_denominator = denominator_value
            normalization_unit = denominator_unit
        region_map[mask] = value
        rows.append({
            "label_id": label,
            "region_name": names.get(label, f"label-{label}"),
            "total_endpoints": int(denominator_value) if damage_model == "any_hit" else denominator_value,
            "affected_endpoints": int(numerator_value) if damage_model == "any_hit" else numerator_value,
            "disconnection_value": value,
            "damage_model": damage_model,
            "denominator_unit": denominator_unit,
            "numerator_unit": numerator_unit,
            "normalization_denominator": normalization_denominator,
            "normalization_unit": normalization_unit,
        })
    return region_map, rows


def _dilate_labels(atlas, zooms, radius_mm):
    if radius_mm <= 0:
        return atlas
    distance, indices = distance_transform_edt(atlas == 0, sampling=zooms, return_indices=True)
    nearest = atlas[tuple(indices)]
    output = atlas.copy()
    fill = (atlas == 0) & (distance <= radius_mm)
    output[fill] = nearest[fill]
    return output


def _deduplicated_ratio_weights(operator, lesion):
    """Return k-1(k>0) and k using unique positive-length lesion voxels per track."""
    hit_operator = operator.copy()
    hit_operator.sum_duplicates()
    hit_operator.data = (hit_operator.data > 0).astype(np.float32)
    counts = np.asarray(hit_operator @ (np.asarray(lesion).ravel() > 0).astype(np.float32)).ravel()
    return counts - (counts > 0), counts


def _write_network_csv(path, numerator, denominator, names, compact=False):
    label_ids = sorted(label for label in names if label > 0)
    selected_numerator = numerator if compact else numerator[np.ix_(label_ids, label_ids)]
    selected_denominator = denominator if compact else denominator[np.ix_(label_ids, label_ids)]
    matrix = _ratio(selected_numerator, selected_denominator)
    headings = [f"{label}:{names.get(label, f'label-{label}')}" for label in label_ids]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["region", *headings])
        for heading, values in zip(headings, matrix):
            writer.writerow([heading, *(f"{float(value):.10g}" for value in values)])


def _make_qc(background, lesion, maps, path, damage_model, network_csvs=None, atlas_names=None):
    z = int(np.rint(np.argwhere(lesion).mean(0))[2]) if np.any(lesion) else lesion.shape[2] // 2
    model_title = {
        "any_hit": "Any-hit",
        "absolute_length": "Mean affected length (mm)",
        "fractional_length": "Mean affected fraction",
        "streamline_mean": "Equal-streamline mean",
        "total_length_ratio": "Length-pooled ratio",
        "lesion_voxel_mean": "Lesion-voxel mean connectivity",
    }[damage_model]
    titles = (f"{model_title}: traversal", f"{model_title}: endpoints", f"{model_title}: atlas regions")
    networks = []
    for csv_path in network_csvs or []:
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        labels = rows[0][1:]
        matrix = np.asarray([row[1:] for row in rows[1:]], dtype=float)
        if matrix.shape != (len(labels), len(labels)) or [row[0] for row in rows[1:]] != labels:
            raise ValueError(f"Invalid network CSV: {csv_path}")
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError(f"Invalid network values: {csv_path}")
        networks.append(matrix)
    if len(networks) != len(atlas_names or []):
        raise ValueError("QC requires one atlas name per network")
    nrows = 1 + (len(networks) + 2) // 3
    figure, grid = plt.subplots(nrows, 3, figsize=(15, 5 * nrows), facecolor="#111318", squeeze=False)
    figure.subplots_adjust(hspace=0.35, wspace=0.32)
    axes = grid[0]
    image = None
    if damage_model in {"lesion_voxel_mean", "absolute_length"}:
        vmax_values = [float(np.percentile(data[data > 0], 99)) if np.any(data > 0) else 1.0
                       for data in maps]
        mask_threshold = 0.0
        colorbar_label = "Mean affected length (mm)" if damage_model == "absolute_length" else "Mean streamline connectivity per lesion voxel"
    else:
        vmax_values = [1.0] * len(maps)
        mask_threshold = 0.005
        colorbar_label = "Disconnection value"
    for axis, data, title, vmax in zip(axes, maps, titles, vmax_values):
        axis.imshow(background[:, :, z].T, cmap="gray", origin="lower")
        shown = np.ma.masked_where(data[:, :, z].T <= mask_threshold, data[:, :, z].T)
        image = axis.imshow(shown, cmap="inferno", origin="lower", vmin=0, vmax=vmax, alpha=0.78)
        axis.contour(lesion[:, :, z].T.astype(float), [0.5], colors=["#22d3ee"], linewidths=1)
        axis.set_title(title, color="white")
        axis.axis("off")
        if damage_model in {"lesion_voxel_mean", "absolute_length"}:
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
            colorbar.set_label(colorbar_label, color="white", fontsize=8)
            colorbar.ax.tick_params(colors="white", labelsize=8)
    if damage_model not in {"lesion_voxel_mean", "absolute_length"}:
        colorbar = figure.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02)
        colorbar.set_label(colorbar_label, color="white")
        colorbar.ax.tick_params(colors="white")
    network_max = max((float(matrix.max()) for matrix in networks), default=0.0)
    network_max = (network_max or 1.0) if damage_model == "absolute_length" else 1.0
    for index, (matrix, name) in enumerate(zip(networks, atlas_names or [])):
        axis = grid[1 + index // 3, index % 3]
        # IIT exports undirected edges once, in the upper triangle; do not imply absent reverse edges.
        shown = np.ma.array(matrix, mask=np.tri(len(matrix), dtype=bool))
        axis.set_facecolor("#30343b")
        image = axis.imshow(shown, cmap="inferno", vmin=0, vmax=network_max, interpolation="nearest")
        ticks = np.unique(np.linspace(0, len(matrix) - 1, min(5, len(matrix)), dtype=int))
        axis.set_xticks(ticks, labels=ticks + 1)
        axis.set_yticks(ticks, labels=ticks + 1)
        axis.tick_params(colors="white", labelsize=9)
        axis.set_xlabel("ROI order in CSV", color="white")
        axis.set_ylabel("ROI order in CSV", color="white")
        axis.set_title(f"{name}: disconnection network\nUpper triangle; {len(matrix)} ROIs", color="white", fontsize=11)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
        colorbar.set_label("Mean affected length (mm)" if damage_model == "absolute_length" else "Disconnection fraction (0-1)", color="white", fontsize=9)
        colorbar.ax.tick_params(colors="white", labelsize=9)
    for axis in grid[1:].ravel()[len(networks):]:
        axis.axis("off")
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _tck_count(path):
    with Path(path).open("rb") as stream:
        if not stream.readline().startswith(b"mrtrix tracks"):
            raise ValueError(f"Not an MRtrix TCK: {path}")
        for line in stream:
            if line.startswith(b"count:"):
                return int(line.split(b":", 1)[1].strip())
            if line.strip() == b"END":
                return 0
    raise ValueError(f"Invalid TCK header: {path}")


def _load_vector(path, expected_count, label):
    values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    if values.size != expected_count:
        raise RuntimeError(f"Expected {expected_count} {label} values, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Non-finite values in {label}")
    return values


def _load_u24(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 3:
        raise RuntimeError(f"Invalid uint24 cache file: {path}")
    values = raw.reshape(-1, 3).astype(np.uint32)
    return values[:, 0] | (values[:, 1] << 8) | (values[:, 2] << 16)


def _cache_file(cache_dir, filename):
    path = (cache_dir / filename).resolve()
    if cache_dir.resolve() not in path.parents:
        raise RuntimeError(f"Cache manifest contains an unsafe path: {filename}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing operator-cache file: {path}")
    return path


class _DisconnectionBase(BaseInterface):
    output_spec = DisconnectionOutputSpec

    @staticmethod
    def _resolve_output(value, cwd):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else Path(cwd) / path).resolve()

    def _requested_outputs(self, cwd):
        outputs = {
            "disconnection_probability": self._resolve_output(self.inputs.output_disconnection_probability, cwd),
            "chacovol_endpoint_voxelwise": self._resolve_output(self.inputs.output_chacovol_endpoint_voxelwise, cwd),
            "qc": self._resolve_output(self.inputs.output_qc, cwd),
            "chacovol_endpoint_surface_lh": self._resolve_output(self.inputs.output_chacovol_endpoint_surface_lh, cwd),
            "chacovol_endpoint_surface_rh": self._resolve_output(self.inputs.output_chacovol_endpoint_surface_rh, cwd),
            "jhu_regionwise_csv": self._resolve_output(self.inputs.output_jhu_regionwise_csv, cwd),
        }
        region_csvs = ([self._resolve_output(path, cwd) for path in self.inputs.output_chacovol_regionwise_csvs]
                       if isdefined(self.inputs.output_chacovol_regionwise_csvs) else [])
        outputs["chacovol_regionwise_csvs"] = region_csvs
        network_csvs = ([self._resolve_output(path, cwd) for path in self.inputs.output_chacoconn_regionwise_csvs]
                        if isdefined(self.inputs.output_chacoconn_regionwise_csvs) else [])
        outputs["chacoconn_regionwise_csvs"] = network_csvs
        flat_outputs = [value for key, value in outputs.items()
                        if key not in {"chacovol_regionwise_csvs", "chacoconn_regionwise_csvs"}]
        flat_outputs += region_csvs + network_csvs
        if len(set(flat_outputs)) != len(flat_outputs):
            raise ValueError("Each output must have a distinct filename")
        for key in ("disconnection_probability", "chacovol_endpoint_voxelwise"):
            if not str(outputs[key]).lower().endswith((".nii", ".nii.gz")):
                raise ValueError(f"{key} must end in .nii or .nii.gz")
        if any(path.suffix.lower() != ".csv" for path in region_csvs):
            raise ValueError("All output_chacovol_regionwise_csvs must end in .csv")
        if any(path.suffix.lower() != ".csv" for path in network_csvs):
            raise ValueError("All output_chacoconn_regionwise_csvs must end in .csv")
        if outputs["jhu_regionwise_csv"].suffix.lower() != ".csv":
            raise ValueError("output_jhu_regionwise_csv must end in .csv")
        if outputs["qc"].suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            raise ValueError("output_qc has an unsupported extension")
        for hemisphere in ("lh", "rh"):
            if not str(outputs[f"chacovol_endpoint_surface_{hemisphere}"]).endswith(".func.gii"):
                raise ValueError("Endpoint surface outputs must end in .func.gii")
        return outputs

    def _prepare_outputs(self, outputs):
        paths = [value for key, value in outputs.items()
                 if key not in {"chacovol_regionwise_csvs", "chacoconn_regionwise_csvs"}]
        paths.extend(outputs["chacovol_regionwise_csvs"])
        paths.extend(outputs["chacoconn_regionwise_csvs"])
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not self.inputs.overwrite:
                raise FileExistsError(f"Output already exists: {path}")

    def _finish_outputs(self, runtime, outputs, lesion_image, lesion_mni, background, maps, network_sums,
                        lesion_voxel_count, surface_volume=None, damage_model=None):
        from cvdproc.pipelines.dmri.nemo.nemo_postprocess_nipype import NemoCorticalMetrics

        damage_model = self.inputs.damage_model if damage_model is None else damage_model
        denominator_mni, numerator_mni, endpoint_denominator_mni, endpoint_numerator_mni = maps
        atlas_files = list(self.inputs.atlas_files)
        atlas_label_files = list(self.inputs.atlas_label_files)
        atlas_names = list(self.inputs.atlas_names)
        region_outputs = outputs["chacovol_regionwise_csvs"]
        network_outputs = outputs["chacoconn_regionwise_csvs"]
        traversal_mni = _ratio(numerator_mni, denominator_mni)
        endpoint_mni = _ratio(endpoint_numerator_mni, endpoint_denominator_mni)
        descriptions = {
            "any_hit": "Affected streamline fraction (0-1)",
            "absolute_length": "Mean affected streamline length (mm)",
            "fractional_length": "Mean affected streamline proportion (0-1)",
        }
        description = descriptions[damage_model]
        _save_like(traversal_mni, lesion_image, outputs["disconnection_probability"], description=description)
        _save_like(endpoint_mni, lesion_image, outputs["chacovol_endpoint_voxelwise"], description=description)
        # Use voxel means (including zeros), not the endpoint-weighted ChaCo definition.
        jhu_names = {int(label.get("index")): label.text.strip() for label in ET.parse(JHU_LABELS).getroot().findall(".//label")}
        with tempfile.TemporaryDirectory(prefix="disconnection_jhu_", dir=runtime.cwd) as directory:
            raw_csv = str(Path(directory) / "voxel_means.csv")
            CalculateScalarMaps(data_files=[str(outputs["disconnection_probability"])], mask_file=str(JHU_ATLAS),
                                colnames=["disconnection_value"], output_csv=raw_csv, statistic="mean",
                                ignore_background=False).run(cwd=directory)
            with Path(raw_csv).open(newline="") as stream:
                jhu_rows = list(csv.DictReader(stream))
        with outputs["jhu_regionwise_csv"].open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("label_id", "region_name", "disconnection_value", "damage_model", "statistic", "value_unit"))
            writer.writeheader()
            for row in jhu_rows:
                label = int(row["roi_label"])
                writer.writerow(dict(label_id=label, region_name=jhu_names[label], disconnection_value=row["disconnection_value"],
                                     damage_model=damage_model, statistic="voxel_mean_including_zeros",
                                     value_unit="mm" if damage_model == "absolute_length" else "fraction"))
        mni_maps = [traversal_mni, endpoint_mni]

        qc_region_map = endpoint_mni
        for atlas_path, labels_path, atlas_name, csv_path in zip(
                atlas_files, atlas_label_files, atlas_names, region_outputs):
            atlas_image = nib.load(atlas_path)
            if atlas_image.shape[:3] != lesion_image.shape[:3] or not np.allclose(
                    atlas_image.affine, lesion_image.affine, rtol=0, atol=1e-3):
                raise ValueError(f"Atlas {atlas_name} does not match the MNI lesion grid: {atlas_path}")
            atlas = np.rint(np.asanyarray(atlas_image.dataobj)).astype(np.int32)
            region_map, rows = _atlas_region_values(
                atlas, endpoint_denominator_mni, endpoint_numerator_mni,
                _load_bids_atlas_labels(labels_path), damage_model, lesion_voxel_count)
            _write_csv(rows, csv_path)
            if qc_region_map is endpoint_mni:
                qc_region_map = region_map

        for labels_path, csv_path, (network_denominator, network_numerator) in zip(
                atlas_label_files, network_outputs, network_sums):
            _write_network_csv(
                csv_path, network_numerator, network_denominator,
                _load_bids_atlas_labels(labels_path)
            )

        _make_qc(background, lesion_mni, (mni_maps[0], mni_maps[1], qc_region_map), outputs["qc"], damage_model,
                 network_csvs=network_outputs, atlas_names=atlas_names)
        with tempfile.TemporaryDirectory(prefix="disconnection_endpoint_surface_", dir=runtime.cwd) as directory:
            surface_dir = Path(directory)
            use_freesurfer = (hasattr(self.inputs, "use_freesurfer_transform")
                              and self.inputs.use_freesurfer_transform)
            if use_freesurfer:
                if surface_volume is None:
                    raise ValueError("FreeSurfer surface projection requires a T1w-space endpoint map")
                for hemisphere, wall_file in (("lh", LH_MEDIAL_WALL_TXT), ("rh", RH_MEDIAL_WALL_TXT)):
                    mgh = surface_dir / f"endpoint_{hemisphere}_fsaverage.mgh"
                    MRIvol2surf(volume=str(surface_volume), subjects_dir=self.inputs.freesurfer_subjects_dir,
                                regheader=self.inputs.freesurfer_subject_id, hemi=hemisphere, target="fsaverage",
                                proj_frac=0.5, output_surf=str(mgh)).run(cwd=str(surface_dir))
                    values = np.asarray(nib.load(str(mgh)).dataobj, dtype=np.float32).squeeze()
                    if values.shape != (163842,) or not np.isfinite(values).all():
                        raise RuntimeError(f"Unexpected {hemisphere} fsaverage projection: {values.shape}")
                    values[np.loadtxt(wall_file, dtype=int)] = np.nan
                    gifti = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(values)])
                    nib.save(gifti, str(outputs[f"chacovol_endpoint_surface_{hemisphere}"]))
            else:
                endpoint_name = "endpoint.nii.gz"
                nib.save(nib.load(str(outputs["chacovol_endpoint_voxelwise"])), str(surface_dir / endpoint_name))
                NemoCorticalMetrics(nemo_output_dir=str(surface_dir), nemo_postprocessed_dir=str(surface_dir),
                                    output_csv_dir=str(surface_dir)).run(cwd=str(surface_dir))
                for hemisphere in ("lh", "rh"):
                    shutil.move(str(surface_dir / f"endpoint_{hemisphere}_fsaverage.func.gii"),
                                str(outputs[f"chacovol_endpoint_surface_{hemisphere}"]))
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs.update(getattr(self, "_results", {}))
        return outputs


class IITDisconnection(_DisconnectionBase):
    """Compute traversal, endpoint and external-atlas disconnection from the IIT cache."""
    input_spec = IITDisconnectionInputSpec
    output_spec = IITDisconnectionOutputSpec

    def _data_files(self):
        files = [IIT_T1, IIT_T1_256, IIT_TO_MNI_WARP, MNI_TO_IIT_WARP, JHU_ATLAS, JHU_LABELS]
        if isdefined(self.inputs.atlas_files):
            files.extend(Path(path) for path in self.inputs.atlas_files)
        if isdefined(self.inputs.atlas_label_files):
            files.extend(Path(path) for path in self.inputs.atlas_label_files)
        return tuple(files)

    def _operator_cache(self):
        cache_dir = IIT_OPERATOR_CACHE.resolve()
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"IIT operator cache is required: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("format") != CACHE_FORMAT or manifest.get("grid_shape") != [256, 256, 256]
                or manifest.get("length_encoding") != "float32-little-endian-mm"):
            raise RuntimeError(f"Unsupported IIT operator cache: {manifest_path}")
        chunks = manifest.get("chunks", [])
        if not chunks or any(int(chunk.get("tracks", 0)) <= 0 for chunk in chunks):
            raise RuntimeError("IIT operator cache contains no valid chunks")
        if sum(int(chunk["tracks"]) for chunk in chunks) != int(manifest.get("total_tracks", -1)):
            raise RuntimeError("Operator-cache track count does not match manifest")
        required = ("p_indptr.u32", "p_indices.u24", "a_indptr.u32", "a_indices.u24", "a_lengths.f32",
                    "a_total_lengths.f32", "endpoints.u32")
        for chunk in chunks:
            for name in required:
                if name not in chunk.get("files", {}):
                    raise RuntimeError(f"Missing {name} in cache chunk {chunk.get('id')}")
                _cache_file(cache_dir, chunk["files"][name])
        return cache_dir, manifest

    def _requested_output_groups(self, cwd):
        lesion_files = [Path(path).resolve() for path in self.inputs.lesion_files]
        if not lesion_files:
            raise ValueError("At least one lesion file is required")
        damage_models = list(self.inputs.damage_models) if isdefined(self.inputs.damage_models) else []
        damage_models = damage_models or [self.inputs.damage_model]
        if len(damage_models) != len(set(damage_models)):
            raise ValueError("damage_models must not contain duplicates")
        count, atlas_count = len(lesion_files), len(self.inputs.atlas_files)
        output_count = count * len(damage_models)
        input_fields = {
            "disconnection_probability": "output_disconnection_probability",
            "chacovol_endpoint_voxelwise": "output_chacovol_endpoint_voxelwise",
            "chacovol_endpoint_surface_lh": "output_chacovol_endpoint_surface_lh",
            "chacovol_endpoint_surface_rh": "output_chacovol_endpoint_surface_rh",
            "jhu_regionwise_csv": "output_jhu_regionwise_csv",
            "qc": "output_qc",
        }
        aggregate = {}
        for result_name, input_name in input_fields.items():
            values = [self._resolve_output(path, cwd) for path in getattr(self.inputs, input_name)]
            if len(values) != output_count:
                raise ValueError(f"{input_name} must contain lesion_count * model_count paths")
            aggregate[result_name] = values
        deduplicated_outputs = []
        if isdefined(self.inputs.output_deduplicated_voxel_ratio):
            values = [self._resolve_output(path, cwd) for path in self.inputs.output_deduplicated_voxel_ratio]
            if len(values) != count:
                raise ValueError("output_deduplicated_voxel_ratio must contain one path per lesion")
            if any(not str(path).lower().endswith((".nii", ".nii.gz")) for path in values):
                raise ValueError("output_deduplicated_voxel_ratio must end in .nii or .nii.gz")
            aggregate["deduplicated_voxel_ratio"] = values
            deduplicated_outputs = values
        for result_name, input_name in (
                ("chacovol_regionwise_csvs", "output_chacovol_regionwise_csvs"),
                ("chacoconn_regionwise_csvs", "output_chacoconn_regionwise_csvs")):
            values = [self._resolve_output(path, cwd) for path in getattr(self.inputs, input_name)]
            if len(values) != output_count * atlas_count:
                raise ValueError(
                    f"{input_name} must contain lesion_count * model_count * atlas_count paths")
            aggregate[result_name] = values
        all_paths = [path for values in aggregate.values() for path in values]
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("Each batch output must have a distinct filename")
        groups = []
        for lesion_index in range(count):
            model_groups = {}
            for model_index, damage_model in enumerate(damage_models):
                output_index = lesion_index * len(damage_models) + model_index
                outputs = {key: values[output_index] for key, values in aggregate.items()
                           if key not in {"chacovol_regionwise_csvs", "chacoconn_regionwise_csvs",
                                          "deduplicated_voxel_ratio"}}
                first = output_index * atlas_count
                outputs["chacovol_regionwise_csvs"] = aggregate["chacovol_regionwise_csvs"][first:first + atlas_count]
                outputs["chacoconn_regionwise_csvs"] = aggregate["chacoconn_regionwise_csvs"][first:first + atlas_count]
                for key in ("disconnection_probability", "chacovol_endpoint_voxelwise"):
                    if not str(outputs[key]).lower().endswith((".nii", ".nii.gz")):
                        raise ValueError(f"{key} must end in .nii or .nii.gz")
                if outputs["jhu_regionwise_csv"].suffix.lower() != ".csv":
                    raise ValueError("output_jhu_regionwise_csv must end in .csv")
                if outputs["qc"].suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
                    raise ValueError("output_qc has an unsupported extension")
                if any(path.suffix.lower() != ".csv" for path in
                       outputs["chacovol_regionwise_csvs"] + outputs["chacoconn_regionwise_csvs"]):
                    raise ValueError("Regional and network outputs must end in .csv")
                for hemisphere in ("lh", "rh"):
                    if not str(outputs[f"chacovol_endpoint_surface_{hemisphere}"]).endswith(".func.gii"):
                        raise ValueError("Endpoint surface outputs must end in .func.gii")
                self._prepare_outputs(outputs)
                model_groups[damage_model] = outputs
            groups.append(model_groups)
        for path in deduplicated_outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not self.inputs.overwrite:
                raise FileExistsError(f"Output already exists: {path}")
        return lesion_files, damage_models, groups, deduplicated_outputs, aggregate

    def _process_operator_cache(self, lesions256, damage_models, cache_dir, manifest, endpoint_atlases,
                                compute_denominators, compute_deduplicated_ratio):
        nvox = 256**3
        lesions = [np.asarray(lesion, dtype=np.float32).ravel(order="C") for lesion in lesions256]
        denominator = np.zeros(nvox, np.float64) if compute_denominators else None
        endpoint_denominator = np.zeros(nvox, np.float64) if compute_denominators else None
        numerators = {model: [np.zeros(nvox, np.float64) for _ in lesions] for model in damage_models}
        endpoint_numerators = {model: [np.zeros(nvox, np.float64) for _ in lesions] for model in damage_models}
        deduplicated_numerators = ([np.zeros(nvox, np.float64) for _ in lesions]
                                   if compute_deduplicated_ratio else None)
        deduplicated_denominators = ([np.zeros(nvox, np.float64) for _ in lesions]
                                     if compute_deduplicated_ratio else None)
        network_denominators = ([np.zeros((int(atlas.max()) + 1,) * 2, np.float64)
                                 for atlas in endpoint_atlases] if compute_denominators else None)
        network_numerators = {
            model: [[np.zeros((int(atlas.max()) + 1,) * 2, np.float64) for atlas in endpoint_atlases]
                    for _ in lesions]
            for model in damage_models
        }
        affected_tracks = np.zeros(len(lesions), np.int64)
        total_tracks = 0
        for chunk in manifest.get("chunks", []):
            files, ntracks = chunk["files"], int(chunk["tracks"])
            p_indptr = np.fromfile(_cache_file(cache_dir, files["p_indptr.u32"]), dtype="<u4")
            p_indices = _load_u24(_cache_file(cache_dir, files["p_indices.u24"]))
            if (p_indptr.size != ntracks + 1 or p_indptr[0] != 0 or np.any(p_indptr[1:] < p_indptr[:-1])
                    or int(p_indptr[-1]) != p_indices.size or np.any(p_indices >= nvox)):
                raise RuntimeError(f"Invalid projection operator in cache chunk {chunk.get('id')}")
            projection = sparse.csr_matrix((np.ones(p_indices.size, np.float32), p_indices, p_indptr),
                                           shape=(ntracks, nvox), dtype=np.float32)
            projection.sum_duplicates()
            projection.data.fill(1)
            a_indptr = np.fromfile(_cache_file(cache_dir, files["a_indptr.u32"]), dtype="<u4")
            a_indices = _load_u24(_cache_file(cache_dir, files["a_indices.u24"]))
            a_lengths = np.fromfile(_cache_file(cache_dir, files["a_lengths.f32"]), dtype="<f4")
            if (a_indptr.size != ntracks + 1 or a_indptr[0] != 0 or np.any(a_indptr[1:] < a_indptr[:-1])
                    or int(a_indptr[-1]) != a_indices.size or a_indices.size != a_lengths.size
                    or np.any(a_indices >= nvox) or not np.isfinite(a_lengths).all() or np.any(a_lengths < 0)):
                raise RuntimeError(f"Invalid lesion operator in cache chunk {chunk.get('id')}")
            operator = sparse.csr_matrix((a_lengths, a_indices, a_indptr),
                                         shape=(ntracks, nvox), dtype=np.float32)
            total_cached_lengths = None
            if "fractional_length" in damage_models:
                total_cached_lengths = np.fromfile(
                    _cache_file(cache_dir, files["a_total_lengths.f32"]), dtype="<f4")
                if (total_cached_lengths.size != ntracks or not np.isfinite(total_cached_lengths).all()
                        or np.any(total_cached_lengths < 0)):
                    raise RuntimeError(f"Invalid cached streamline lengths in chunk {chunk.get('id')}")
            track_denominator = np.ones(ntracks, np.float32)
            if compute_denominators:
                denominator += np.asarray(projection.T @ track_denominator).ravel()
            endpoints = np.fromfile(_cache_file(cache_dir, files["endpoints.u32"]), dtype="<u4")
            if endpoints.size != 2*ntracks or np.any((endpoints >= nvox) & (endpoints != np.iinfo(np.uint32).max)):
                raise RuntimeError(f"Invalid endpoint operator in cache chunk {chunk.get('id')}")
            endpoint_indices = endpoints.reshape(-1)
            endpoint_tracks = np.repeat(np.arange(ntracks), 2)
            valid_endpoint = endpoint_indices != np.iinfo(np.uint32).max
            endpoint_indices, endpoint_tracks = endpoint_indices[valid_endpoint], endpoint_tracks[valid_endpoint]
            if compute_denominators:
                endpoint_denominator += np.bincount(
                    endpoint_indices, weights=track_denominator[endpoint_tracks], minlength=nvox)
            pair_data = []
            endpoint_pairs = endpoints.reshape(ntracks, 2)
            endpoint_valid = endpoint_pairs != np.iinfo(np.uint32).max
            for atlas_index, atlas in enumerate(endpoint_atlases):
                pair_labels = np.zeros((ntracks, 2), np.int32)
                pair_labels[endpoint_valid] = atlas.ravel()[endpoint_pairs[endpoint_valid]]
                first, second = pair_labels[:, 0], pair_labels[:, 1]
                valid_pair = (first > 0) & (second > 0) & (first != second)
                low = np.minimum(first[valid_pair], second[valid_pair])
                high = np.maximum(first[valid_pair], second[valid_pair])
                size = int(atlas.max()) + 1
                flat = low * size + high
                pair_data.append((valid_pair, flat, size))
                if compute_denominators:
                    network_denominators[atlas_index] += np.bincount(
                        flat, weights=track_denominator[valid_pair], minlength=size * size
                    ).reshape(size, size)
            for lesion_index, lesion in enumerate(lesions):
                damaged = np.asarray(operator @ lesion).ravel()
                for damage_model in damage_models:
                    if damage_model == "any_hit":
                        weights = (damaged > 0).astype(np.float32)
                    elif damage_model == "absolute_length":
                        weights = damaged
                    else:
                        weights = _fractional_length_weights(damaged, total_cached_lengths)
                    numerators[damage_model][lesion_index] += np.asarray(projection.T @ weights).ravel()
                    endpoint_numerators[damage_model][lesion_index] += np.bincount(
                        endpoint_indices, weights=weights[endpoint_tracks], minlength=nvox)
                    for atlas_index, (valid_pair, flat, size) in enumerate(pair_data):
                        network_numerators[damage_model][lesion_index][atlas_index] += np.bincount(
                            flat, weights=weights[valid_pair], minlength=size * size
                        ).reshape(size, size)
                if compute_deduplicated_ratio:
                    deduplicated_weight, distinct_hit_count = _deduplicated_ratio_weights(operator, lesion)
                    deduplicated_numerators[lesion_index] += np.asarray(
                        projection.T @ deduplicated_weight).ravel()
                    deduplicated_denominators[lesion_index] += np.asarray(
                        projection.T @ distinct_hit_count).ravel()
                affected_tracks[lesion_index] += np.count_nonzero(damaged)
            total_tracks += ntracks
            print(f"Processed operator-cache chunk {chunk.get('id')}: models={','.join(damage_models)}, "
                  f"lesions={len(lesions)}, tracks={ntracks}", flush=True)
        if total_tracks != int(manifest.get("total_tracks", -1)):
            raise RuntimeError("Operator-cache track count does not match manifest")
        reshape = lambda values: None if values is None else values.reshape((256, 256, 256))
        return (reshape(denominator),
                {model: [reshape(value) for value in values] for model, values in numerators.items()},
                reshape(endpoint_denominator),
                {model: [reshape(value) for value in values] for model, values in endpoint_numerators.items()},
                network_denominators, network_numerators, total_tracks, affected_tracks.tolist(),
                None if deduplicated_numerators is None else [reshape(value) for value in deduplicated_numerators],
                None if deduplicated_denominators is None else [reshape(value) for value in deduplicated_denominators])

    def _run_interface(self, runtime):
        cache_dir, manifest = self._operator_cache()
        missing = [str(path) for path in self._data_files() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing IIT data:\n" + "\n".join(missing))
        lesion_files, damage_models, output_groups, deduplicated_outputs, aggregate_outputs = (
            self._requested_output_groups(runtime.cwd))
        iit_image, iit256_image = nib.load(str(IIT_T1)), nib.load(str(IIT_T1_256))
        iit_to_mni, mni_to_iit = nib.load(str(IIT_TO_MNI_WARP)), nib.load(str(MNI_TO_IIT_WARP))
        jhu_image = nib.load(str(JHU_ATLAS))
        if iit_image.shape[:3] != (182, 218, 182) or iit256_image.shape[:3] != (256, 256, 256):
            raise ValueError("Unexpected IIT template grid")
        lesion_images, lesions_mni, lesions256 = [], [], []
        for lesion_file in lesion_files:
            image = nib.load(str(lesion_file))
            if image.shape[:3] != (182, 218, 182) or not CalculateScalarMaps._same_grid(image, jhu_image):
                raise ValueError(f"Lesion must match the 1 mm MNI grid: {lesion_file}")
            values = np.asarray(image.dataobj, dtype=np.float32)
            if values.ndim != 3 or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Lesion must be a finite, nonnegative 3D image: {lesion_file}")
            if "fractional_length" in damage_models and np.any(values > 1):
                raise ValueError(f"fractional_length requires lesion values in [0, 1]: {lesion_file}")
            if not np.allclose(image.affine, mni_to_iit.affine, rtol=0, atol=1e-3):
                raise ValueError(f"Lesion affine does not match the MNI warp grid: {lesion_file}")
            lesion_images.append(image)
            lesions_mni.append(values > 0)
            lesions256.append(_to_iit256(_warp_pullback(image, iit_to_mni, iit_image, 0, np.float32)))
        atlas_files = list(self.inputs.atlas_files)
        atlas_label_files = list(self.inputs.atlas_label_files)
        atlas_names = list(self.inputs.atlas_names)
        if not atlas_files or not (len(atlas_files) == len(atlas_label_files) == len(atlas_names)):
            raise ValueError("atlas_files, atlas_label_files and atlas_names must have equal nonzero lengths")
        endpoint_atlases = []
        for atlas_name, atlas_path in zip(atlas_names, atlas_files):
            atlas_image = nib.load(atlas_path)
            if not CalculateScalarMaps._same_grid(atlas_image, lesion_images[0]):
                raise ValueError(f"Atlas {atlas_name} does not match the MNI lesion grid")
            atlas_iit = _warp_pullback(atlas_image, iit_to_mni, iit_image, 0, np.int32)
            atlas256 = _to_iit256(np.rint(atlas_iit).astype(np.int32))
            endpoint_atlases.append(_dilate_labels(
                atlas256, iit256_image.header.get_zooms()[:3], self.inputs.atlas_assignment_radius_mm))
        reference = lesion_images[0]

        def warp_raw(data256):
            image = nib.Nifti1Image(_from_iit256(np.asarray(data256, np.float32)),
                                    iit_image.affine, iit_image.header)
            return np.maximum(_warp_pullback(image, mni_to_iit, reference, 1, np.float32), 0)

        background = _warp_pullback(iit_image, mni_to_iit, reference, 1, np.float32)
        denominator_mni = endpoint_denominator_mni = None
        network_denominators = None
        compute_deduplicated_ratio = "deduplicated_voxel_ratio" in aggregate_outputs
        for first in range(0, len(lesions256), IIT_LESION_BATCH_SIZE):
            last = min(first + IIT_LESION_BATCH_SIZE, len(lesions256))
            result = self._process_operator_cache(
                lesions256[first:last], damage_models, cache_dir, manifest, endpoint_atlases,
                compute_denominators=denominator_mni is None,
                compute_deduplicated_ratio=compute_deduplicated_ratio)
            denominator, numerators, endpoint_denominator, endpoint_numerators = result[:4]
            current_network_denominators, network_numerators, track_count, affected_counts = result[4:8]
            deduplicated_numerators, deduplicated_denominators = result[8:]
            if denominator_mni is None:
                denominator_mni = warp_raw(denominator)
                endpoint_denominator_mni = warp_raw(endpoint_denominator)
                network_denominators = current_network_denominators
            for offset, affected_count in enumerate(affected_counts):
                index = first + offset
                if compute_deduplicated_ratio:
                    deduplicated_numerator_mni = warp_raw(deduplicated_numerators[offset])
                    deduplicated_denominator_mni = warp_raw(deduplicated_denominators[offset])
                    if np.any(deduplicated_numerator_mni > deduplicated_denominator_mni + 1e-4):
                        raise RuntimeError("Deduplicated numerator exceeds distinct lesion-voxel hit count")
                    _save_like(
                        _ratio(deduplicated_numerator_mni, deduplicated_denominator_mni), lesion_images[index],
                        deduplicated_outputs[index],
                        description="Excess distinct lesion-voxel hit ratio (0-1)")
                for damage_model in damage_models:
                    numerator_mni = warp_raw(numerators[damage_model][offset])
                    endpoint_numerator_mni = warp_raw(endpoint_numerators[damage_model][offset])
                    if damage_model == "any_hit":
                        if np.any(numerator_mni > denominator_mni + 1e-4) or np.any(
                                endpoint_numerator_mni > endpoint_denominator_mni + 1e-4):
                            raise RuntimeError("Affected streamline count exceeds total count")
                    network_sums = list(zip(network_denominators, network_numerators[damage_model][offset]))
                    self._finish_outputs(
                        runtime, output_groups[index][damage_model], lesion_images[index], lesions_mni[index],
                        background, (denominator_mni, numerator_mni, endpoint_denominator_mni,
                                     endpoint_numerator_mni), network_sums,
                        int(np.count_nonzero(lesions256[index])), damage_model=damage_model)
                    print(f"IITDisconnection completed: lesion={index + 1}/{len(lesions256)}, "
                          f"model={damage_model}, tracks={track_count}, nonzero={affected_count}, "
                          f"atlases={len(atlas_files)}", flush=True)
            del result, numerators, endpoint_numerators, network_numerators
            del deduplicated_numerators, deduplicated_denominators
        self._results = {key: [str(path) for path in values] for key, values in aggregate_outputs.items()}
        return runtime




VENTRICLE_LABELS = (4, 5, 14, 15, 24, 43, 44)
SUBCORTICAL_GM_LABELS = (
    8, 10, 11, 12, 13, 17, 18, 26, 28,
    47, 49, 50, 51, 52, 53, 54, 58, 60,
)
DSI_ENDPOINT_RADIUS_MM = 4.0


class IndividualDisconnectionInputSpec(DisconnectionInputSpec):
    lesion_file = File(exists=True, mandatory=True, desc="Nonnegative lesion mask or weights on the T1w grid")
    tractogram_file = File(exists=True, mandatory=True, desc="Native DWI MRtrix .tck or DSI Studio .trk.gz")
    connectome_source = traits.Enum("mrtrix3", "dsistudio", usedefault=True)
    dwi_reference = File(exists=True, mandatory=True)
    mni_reference = File(exists=True, mandatory=True, desc="MNI152NLin6Asym 1 mm target grid")
    t1w_reference = File(exists=True, mandatory=True)
    mni_to_t1w_warp = File(exists=True, mandatory=True)
    t1w_to_mni_warp = File(exists=True, mandatory=True)
    t1w_to_dwi_matrix = File(exists=True, mandatory=True)
    dwi_to_t1w_matrix = File(exists=True, mandatory=True)
    anatomical_segmentation = File(exists=True, desc="DWI-grid aparc+aseg for DSI anatomical filtering")
    cortical_gm_mask = File(exists=True, desc="DWI-grid cortical GM mask for DSI endpoint correction")
    brain_mask = File(exists=True, desc="DWI-grid brain mask for DSI anatomical filtering")
    use_freesurfer_transform = traits.Bool(False, usedefault=True)
    freesurfer_subjects_dir = Directory(exists=True, desc="FreeSurfer SUBJECTS_DIR")
    freesurfer_subject_id = traits.Str(desc="FreeSurfer subject directory name")
    nthreads = traits.Int(0, usedefault=True)


def _trk_header(path):
    with gzip.open(path, "rb") as stream:
        header = stream.read(1000)
    if len(header) != 1000 or header[:6] != b"TRACK\x00":
        raise ValueError(f"Not a gzip-compressed TrackVis file: {path}")
    return {
        "dimensions": tuple(int(x) for x in struct.unpack_from("<3h", header, 6)),
        "voxel_sizes": np.asarray(struct.unpack_from("<3f", header, 12)),
        "voxel_to_rasmm": np.frombuffer(header[440:504], dtype="<f4").reshape(4, 4).astype(float),
        "voxel_order": header[948:952].split(b"\x00", 1)[0].decode("ascii"),
        "streamline_count": int(struct.unpack_from("<i", header, 988)[0]),
    }


def _convert_dsi_trk(trk_path, tck_path, target_affine, trk_affine):
    source = nib.streamlines.TrkFile.load(str(trk_path), lazy_load=True)
    restore = np.eye(4)
    restore[:3, 3] = 0.5
    transform = np.asarray(target_affine) @ restore @ np.linalg.inv(np.asarray(trk_affine))

    def streamlines():
        for streamline in source.tractogram.streamlines:
            yield nib.affines.apply_affine(transform, streamline)

    tractogram = nib.streamlines.LazyTractogram(streamlines=streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.TckFile(tractogram).save(str(tck_path))


def _load_map(path, shape):
    image = nib.load(str(path))
    if image.shape[:3] != shape:
        raise RuntimeError(f"Unexpected map grid {image.shape[:3]}: {path}")
    data = np.asanyarray(image.dataobj).astype(np.float64)
    if not np.all(np.isfinite(data)) or np.any(data < 0):
        raise RuntimeError(f"Invalid map values: {path}")
    return data


def _voxel_indices(points, inverse_affine, shape):
    indices = np.rint(nib.affines.apply_affine(inverse_affine, points)).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
    return indices, valid


def _tissue_safe_line(start, finish, inverse_affine, brain, ventricle):
    distance = float(np.linalg.norm(finish - start))
    samples = np.linspace(start, finish, max(2, int(np.ceil(distance / 0.5)) + 1))
    indices, valid = _voxel_indices(samples, inverse_affine, brain.shape)
    if not valid.all():
        return False
    voxels = tuple(indices.T)
    return bool(np.all(brain[voxels]) and not np.any(ventricle[voxels]))


def _assign_gm_endpoint(point, raw_index, raw_valid, anatomy, inverse_affine, affine):
    if not raw_valid:
        return None, False
    voxel = tuple(raw_index)
    if anatomy["gm"][voxel]:
        return raw_index, False
    if anatomy["gm_distance"][voxel] > DSI_ENDPOINT_RADIUS_MM:
        return None, False
    target = anatomy["gm_nearest"][(slice(None),) + voxel]
    target_world = nib.affines.apply_affine(affine, target)
    if not _tissue_safe_line(
            point, target_world, inverse_affine, anatomy["brain"], anatomy["ventricle"]):
        return None, False
    return target, True


def _audit_streamlines(tck_path, lesion, affine, expected, endpoint_atlases, anatomy=None):
    inverse = np.linalg.inv(affine)
    endpoint_labels = [np.zeros((expected, 2), np.int32) for _ in endpoint_atlases]
    endpoint_indices = np.full((expected, 2), -1, np.int64)
    valid_tracks = np.ones(expected, bool)
    stats = {
        "ventricle_core": 0,
        "internal_outside_brain": 0,
        "valid": 0,
        "direct_gm": 0,
        "snapped": 0,
        "unassigned": 0,
        "both_assigned": 0,
    }
    count = 0
    for count, streamline in enumerate(nib.streamlines.load(str(tck_path), lazy_load=True).streamlines, 1):
        track_index = count - 1
        if track_index >= expected or len(streamline) < 2 or not np.isfinite(streamline).all():
            raise ValueError("Invalid streamline geometry or count")
        points = streamline if anatomy is not None else streamline[[0, -1]]
        all_indices, point_valid = _voxel_indices(points, inverse, lesion.shape)
        in_grid = all_indices[point_valid]

        if anatomy is not None:
            crosses_core = bool(
                in_grid.size and np.any(anatomy["ventricle_core"][tuple(in_grid.T)])
            )
            internal_indices = all_indices[2:-2] if len(all_indices) > 4 else all_indices[:0]
            internal_valid = point_valid[2:-2] if len(point_valid) > 4 else point_valid[:0]
            outside_internal = bool(
                internal_indices.size and (
                    not internal_valid.all()
                    or not np.all(anatomy["brain"][tuple(internal_indices[internal_valid].T)])
                )
            )
            valid_tracks[track_index] = not (crosses_core or outside_internal)
            stats["ventricle_core"] += int(crosses_core)
            stats["internal_outside_brain"] += int(outside_internal)
        stats["valid"] += int(valid_tracks[track_index])

        raw_endpoints = all_indices[[0, -1]]
        raw_endpoint_valid = point_valid[[0, -1]]
        assigned_count = 0
        for endpoint_index, (point, raw_index, raw_valid) in enumerate(zip(
                streamline[[0, -1]], raw_endpoints, raw_endpoint_valid)):
            if not valid_tracks[track_index]:
                continue
            if anatomy is None:
                assigned, snapped = (raw_index, False) if raw_valid else (None, False)
            else:
                assigned, snapped = _assign_gm_endpoint(
                    point, raw_index, raw_valid, anatomy, inverse, affine
                )
            if assigned is None:
                stats["unassigned"] += 1
                continue
            endpoint_indices[track_index, endpoint_index] = np.ravel_multi_index(
                tuple(assigned), lesion.shape
            )
            assigned_count += 1
            stats["snapped" if snapped else "direct_gm"] += 1
            for atlas_index, atlas in enumerate(endpoint_atlases):
                endpoint_labels[atlas_index][track_index, endpoint_index] = atlas[tuple(assigned)]
        stats["both_assigned"] += int(assigned_count == 2)
        if count % 100000 == 0:
            print(f"Assigned endpoints for {count}/{expected} streamlines", flush=True)
    if count != expected:
        raise RuntimeError(f"Expected {expected} streamlines, read {count}")
    return endpoint_labels, endpoint_indices, valid_tracks, stats


def _endpoint_map(endpoint_indices, streamline_weights, shape):
    valid = endpoint_indices >= 0
    weights = np.broadcast_to(np.asarray(streamline_weights)[:, None], endpoint_indices.shape)
    values = np.bincount(
        endpoint_indices[valid], weights=weights[valid], minlength=int(np.prod(shape))
    )
    return values.reshape(shape).astype(np.float64, copy=False)


def _load_dsi_anatomy(segmentation_file, cortical_gm_file, brain_file, reference):
    def load(path, label):
        image = nib.load(str(path))
        if image.shape[:3] != reference.shape[:3] or not np.allclose(
                image.affine, reference.affine, rtol=0, atol=1e-4):
            raise ValueError(f"DSI {label} does not match the DWI reference grid: {path}")
        return np.asanyarray(image.dataobj)

    segmentation = np.rint(load(segmentation_file, "anatomical segmentation")).astype(np.int32)
    brain = load(brain_file, "brain mask") > 0
    ventricle = np.isin(segmentation, VENTRICLE_LABELS)
    gm = ((load(cortical_gm_file, "cortical GM mask") > 0)
          | np.isin(segmentation, SUBCORTICAL_GM_LABELS))
    # Tissue labels take precedence over potentially overlapping derived masks.
    gm &= brain & ~ventricle
    gm_distance, gm_nearest = distance_transform_edt(
        ~gm, sampling=reference.header.get_zooms()[:3], return_indices=True
    )
    return {
        "brain": brain,
        "ventricle": ventricle,
        "ventricle_core": binary_erosion(ventricle, iterations=1),
        "gm": gm,
        "gm_distance": gm_distance,
        "gm_nearest": gm_nearest,
    }


def _network_values(endpoint_labels, numerator_weights, denominator_weights, labels):
    label_ids = sorted(label for label in labels if label > 0)
    id_to_index = {label: index for index, label in enumerate(label_ids)}
    first, second = endpoint_labels[:, 0], endpoint_labels[:, 1]
    valid = (first > 0) & (second > 0) & (first != second)
    first, second = first[valid], second[valid]
    low, high = np.minimum(first, second), np.maximum(first, second)
    row = np.fromiter((id_to_index.get(int(value), -1) for value in low), np.int32, count=low.size)
    col = np.fromiter((id_to_index.get(int(value), -1) for value in high), np.int32, count=high.size)
    known = (row >= 0) & (col >= 0)
    size = len(label_ids)
    flat = row[known] * size + col[known]
    denominator = np.bincount(
        flat, weights=np.asarray(denominator_weights)[valid][known], minlength=size * size
    ).reshape(size, size)
    numerator = np.bincount(
        flat, weights=np.asarray(numerator_weights)[valid][known], minlength=size * size
    ).reshape(size, size)
    ratio = _ratio(numerator, denominator)
    return ratio, numerator, denominator, label_ids, int(known.sum())


class IndividualDisconnection(_DisconnectionBase):
    """Score tracks in native DWI space and warp raw sums before MNI division."""
    input_spec = IndividualDisconnectionInputSpec

    def _mni_to_dwi(self, source, output, temp):
        t1 = str(temp / f"{Path(source).name.split('.')[0]}_t1w.nii.gz")
        MRIConvertApplyWarp(input_image=str(source), output_image=t1, warp_image=self.inputs.mni_to_t1w_warp, interp="nearest").run(cwd=str(temp))
        FLIRT(in_file=t1, reference=self.inputs.dwi_reference, apply_xfm=True, in_matrix_file=self.inputs.t1w_to_dwi_matrix, interp="nearestneighbour", out_file=str(output), out_matrix_file=str(temp / "mni_to_dwi_applied.mat"), output_type="NIFTI_GZ").run(cwd=str(temp))

    def _t1_to_dwi(self, source, output, temp):
        FLIRT(in_file=str(source), reference=self.inputs.dwi_reference, apply_xfm=True,
              in_matrix_file=self.inputs.t1w_to_dwi_matrix, interp="nearestneighbour", out_file=str(output),
              out_matrix_file=str(temp / "t1w_to_dwi_applied.mat"), output_type="NIFTI_GZ").run(cwd=str(temp))

    def _dwi_to_t1(self, source, output, temp, stem):
        FLIRT(in_file=str(source), reference=self.inputs.t1w_reference, apply_xfm=True,
              in_matrix_file=self.inputs.dwi_to_t1w_matrix, interp="trilinear", out_file=str(output),
              out_matrix_file=str(temp / f"{stem}_to_t1w_applied.mat"), output_type="NIFTI_GZ").run(cwd=str(temp))

    def _run_interface(self, runtime):
        outputs = self._requested_outputs(runtime.cwd)
        self._prepare_outputs(outputs)
        atlas_files, label_files = list(self.inputs.atlas_files), list(self.inputs.atlas_label_files)
        if not atlas_files or not (len(atlas_files) == len(label_files) == len(self.inputs.atlas_names)
                                  == len(outputs["chacovol_regionwise_csvs"]) == len(outputs["chacoconn_regionwise_csvs"])):
            raise ValueError("Atlas inputs and ROI/network outputs must have equal lengths")
        dwi_image, mni_image = nib.load(self.inputs.dwi_reference), nib.load(self.inputs.mni_reference)
        lesion_t1_image = nib.load(self.inputs.lesion_file)
        t1w_image = nib.load(self.inputs.t1w_reference)
        if lesion_t1_image.ndim != 3 or not CalculateScalarMaps._same_grid(lesion_t1_image, t1w_image):
            raise ValueError("Lesion and T1w reference must have identical shape and affine")
        lesion_t1 = np.asarray(lesion_t1_image.dataobj, np.float32)
        if not np.isfinite(lesion_t1).all() or np.any(lesion_t1 < 0):
            raise ValueError("Lesion values must be finite and nonnegative")
        if not CalculateScalarMaps._same_grid(mni_image, nib.load(str(JHU_ATLAS))):
            raise ValueError("MNI reference must match the MNI152NLin6Asym 1 mm JHU grid")
        for path in atlas_files:
            if not CalculateScalarMaps._same_grid(nib.load(path), mni_image):
                raise ValueError(f"Atlas does not match the MNI grid: {path}")
        tractogram = Path(self.inputs.tractogram_file).resolve()
        is_dsi = self.inputs.connectome_source == "dsistudio"
        if not str(tractogram).endswith(".trk.gz" if is_dsi else ".tck"):
            raise ValueError("Tractogram extension does not match connectome_source")
        if is_dsi:
            if not all(isdefined(getattr(self.inputs, field)) for field in
                       ("anatomical_segmentation", "cortical_gm_mask", "brain_mask")):
                raise ValueError("DSI Studio requires DWI-space segmentation, cortical GM and brain masks")
            trk = _trk_header(tractogram)
            if (dwi_image.shape[:3] != trk["dimensions"]
                    or not np.allclose(dwi_image.header.get_zooms()[:3], trk["voxel_sizes"], atol=1e-4, rtol=0)
                    or (trk["voxel_order"] and "".join(nib.aff2axcodes(dwi_image.affine)) != trk["voxel_order"])):
                raise ValueError("DSI Studio TRK grid does not match the DWI reference")
            count = trk["streamline_count"]
        else:
            count = _tck_count(tractogram)
        if count <= 0:
            raise ValueError("Tractogram must contain streamlines")
        if self.inputs.use_freesurfer_transform:
            if not (isdefined(self.inputs.freesurfer_subjects_dir)
                    and isdefined(self.inputs.freesurfer_subject_id)):
                raise ValueError("FreeSurfer surface projection requires subjects directory and subject ID")
            fs_subject = Path(self.inputs.freesurfer_subjects_dir) / self.inputs.freesurfer_subject_id
            required = (fs_subject / "mri" / "orig.mgz", fs_subject / "surf" / "lh.white",
                        fs_subject / "surf" / "rh.white", fs_subject / "surf" / "lh.sphere.reg",
                        fs_subject / "surf" / "rh.sphere.reg")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("Incomplete FreeSurfer surface registration:\n" + "\n".join(missing))

        with tempfile.TemporaryDirectory(prefix="individual_disconnection_", dir=runtime.cwd) as directory:
            temp = Path(directory)
            lesion_dwi_file = temp / "lesion_dwi.nii.gz"
            self._t1_to_dwi(self.inputs.lesion_file, lesion_dwi_file, temp)
            lesion_dwi_image = nib.load(str(lesion_dwi_file))
            if not CalculateScalarMaps._same_grid(lesion_dwi_image, dwi_image):
                raise ValueError("Transformed lesion does not match the DWI grid")
            lesion = np.asarray(lesion_dwi_image.dataobj, np.float32)
            if not np.any(lesion > 0):
                raise RuntimeError("Lesion is empty after T1w-to-DWI transformation")
            if self.inputs.damage_model == "fractional_length" and np.any(lesion > 1):
                raise ValueError("fractional_length requires lesion values in [0, 1]")
            endpoint_atlases = []
            for index, atlas_file in enumerate(atlas_files):
                native_atlas = temp / f"atlas_{index}_dwi.nii.gz"
                self._mni_to_dwi(atlas_file, native_atlas, temp)
                atlas_image = nib.load(str(native_atlas))
                if not CalculateScalarMaps._same_grid(atlas_image, dwi_image):
                    raise ValueError("Transformed atlas does not match the DWI grid")
                atlas = np.rint(np.asanyarray(atlas_image.dataobj)).astype(np.int32)
                endpoint_atlases.append(_dilate_labels(atlas, dwi_image.header.get_zooms()[:3], self.inputs.atlas_assignment_radius_mm))
            anatomy = None
            if is_dsi:
                anatomy = _load_dsi_anatomy(self.inputs.anatomical_segmentation, self.inputs.cortical_gm_mask,
                                           self.inputs.brain_mask, dwi_image)
                trk_file, tck_file = temp / "streamlines.trk", temp / "streamlines.tck"
                with gzip.open(tractogram, "rb") as source, trk_file.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=16 << 20)
                _convert_dsi_trk(trk_file, tck_file, dwi_image.affine, trk["voxel_to_rasmm"])
            else:
                # MRtrix coordinates are scanner RAS mm; do not apply TrackVis offsets.
                tck_file = tractogram
            fraction_file, length_file = temp / "fraction.txt", temp / "length.txt"
            TckSampleCommand(tracks=str(tck_file), image=str(lesion_dwi_file), values=str(fraction_file),
                             stat_tck="mean", precise=True, args="-force -quiet").run(cwd=directory)
            fractions = _load_vector(fraction_file, count, "lesion-weighted fractions")
            # Precise MRtrix sampling estimates the integral as a length-weighted mean
            # times streamline length. Subvoxel spline boundary mapping introduces a
            # small numerical difference from the IIT cache's direct length integral.
            if self.inputs.damage_model == "absolute_length":
                result = subprocess.run(["tckstats", str(tck_file), "-dump", str(length_file), "-force", "-quiet"],
                                        cwd=directory, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                damaged = fractions * _load_vector(length_file, count, "streamline lengths")
            else:
                damaged = fractions
            endpoint_labels, endpoint_indices, valid, stats = _audit_streamlines(
                tck_file, lesion, dwi_image.affine, count, endpoint_atlases, anatomy)
            denominator_weights = valid.astype(np.float64)
            numerator_weights = denominator_weights * ((damaged > 0) if self.inputs.damage_model == "any_hit" else damaged)
            maps = []
            for name, weights in (("denominator", denominator_weights), ("numerator", numerator_weights)):
                weight_file, map_file = temp / f"{name}.txt", temp / f"{name}.nii.gz"
                np.savetxt(weight_file, weights, fmt="%.12g")
                ComputeTDI(in_file=str(tck_file), reference=self.inputs.dwi_reference, out_file=str(map_file),
                           tck_weights=str(weight_file), nthreads=self.inputs.nthreads,
                           args="-datatype Float64 -force -quiet").run(cwd=directory)
                maps.append(_load_map(map_file, dwi_image.shape[:3]))
            maps.extend(_endpoint_map(endpoint_indices, weights, dwi_image.shape[:3])
                        for weights in (denominator_weights, numerator_weights))
            # Retain native ratios as intermediates, outside the formal MNI output set.
            native_dir = Path(runtime.cwd) / "native"
            native_dir.mkdir(exist_ok=True)
            unit = ("Mean affected streamline length (mm)" if self.inputs.damage_model == "absolute_length"
                    else "Mean affected streamline proportion (0-1)" if self.inputs.damage_model == "fractional_length"
                    else "Affected streamline fraction (0-1)")
            _save_like(_ratio(maps[1], maps[0]), dwi_image, native_dir / "disconnection.nii.gz", description=unit)
            _save_like(_ratio(maps[3], maps[2]), dwi_image, native_dir / "endpoint_disconnection.nii.gz", description=unit)
            network_sums = []
            for pairs, labels_file in zip(endpoint_labels, label_files):
                labels = _load_bids_atlas_labels(labels_file)
                _, num, den, ids, _ = _network_values(pairs, numerator_weights, denominator_weights, labels)
                full_den, full_num = (np.zeros((max(ids) + 1,) * 2, np.float64) for _ in range(2))
                full_den[np.ix_(ids, ids)], full_num[np.ix_(ids, ids)] = den, num
                network_sums.append((full_den, full_num))
            mni_maps, t1_endpoint_maps = [], {}
            for index, data in enumerate(maps):
                native = temp / f"raw_{index}_dwi.nii.gz"
                t1_target, target = temp / f"raw_{index}_t1w.nii.gz", temp / f"raw_{index}_mni.nii.gz"
                _save_like(data, dwi_image, native)
                self._dwi_to_t1(native, t1_target, temp, f"raw_{index}")
                MRIConvertApplyWarp(input_image=str(t1_target), output_image=str(target),
                                    warp_image=self.inputs.t1w_to_mni_warp, interp="interpolate").run(cwd=directory)
                warped = nib.load(str(target))
                if not CalculateScalarMaps._same_grid(warped, mni_image):
                    raise ValueError("Warp output does not match the requested MNI grid")
                if self.inputs.use_freesurfer_transform and index >= 2:
                    t1_endpoint_maps[index] = _load_map(t1_target, t1w_image.shape[:3])
                mni_maps.append(_load_map(target, mni_image.shape))
            qc_lesion = temp / "lesion_mni.nii.gz"
            MRIConvertApplyWarp(input_image=self.inputs.lesion_file, output_image=str(qc_lesion),
                                warp_image=self.inputs.t1w_to_mni_warp, interp="nearest").run(cwd=directory)
            background_file = temp / "t1w_mni.nii.gz"
            MRIConvertApplyWarp(input_image=self.inputs.t1w_reference, output_image=str(background_file),
                                warp_image=self.inputs.t1w_to_mni_warp, interp="interpolate").run(cwd=directory)
            surface_volume = None
            if self.inputs.use_freesurfer_transform:
                surface_volume = temp / "endpoint_disconnection_t1w.nii.gz"
                _save_like(_ratio(t1_endpoint_maps[3], t1_endpoint_maps[2]), t1w_image, surface_volume)
            self._finish_outputs(runtime, outputs, mni_image, np.asarray(nib.load(str(qc_lesion)).dataobj) > 0,
                                 np.asarray(nib.load(str(background_file)).dataobj), mni_maps, network_sums,
                                 int(np.count_nonzero(lesion)), surface_volume=surface_volume)
            self._results = {key: ([str(path) for path in value] if isinstance(value, list) else str(value))
                             for key, value in outputs.items()}
            print(f"IndividualDisconnection completed: source={self.inputs.connectome_source}, model={self.inputs.damage_model}, "
                  f"tracks={count}, valid={stats['valid']}, affected={np.count_nonzero(numerator_weights)}", flush=True)
        return runtime


__all__ = ["IITDisconnection", "IndividualDisconnection"]
