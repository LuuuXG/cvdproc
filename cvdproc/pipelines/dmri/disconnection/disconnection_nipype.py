"""Nipype interface for IIT atlas-based lesion disconnectome analysis.

The online path uses the existing MRtrix3 installation. All damage models can
use a prebuilt sparse operator cache; interface execution never compiles code.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, File,
                                    InputMultiPath, OutputMultiPath, TraitedSpec, isdefined, traits)
from scipy import sparse
from scipy.ndimage import distance_transform_edt, map_coordinates

from cvdproc.config.paths import get_package_path


IIT_DIR = Path(get_package_path("data", "standard", "IIT"))
MNI_CUSTOM_DIR = Path(get_package_path("data", "standard", "MNI152", "custom"))
IIT_T1 = IIT_DIR / "IITmean_t1.nii.gz"
IIT_T1_256 = IIT_DIR / "IITmean_t1_256.nii.gz"
IIT_TRACTOGRAMS = (
    IIT_DIR / "IIT_HARDI_tractogram_256_part-01.tck",
    IIT_DIR / "IIT_HARDI_tractogram_256_part-02.tck",
)
IIT_OPERATOR_CACHE = IIT_DIR / "weighted_disconnection_operator" / "10m"
CACHE_FORMAT = "cvdproc-iit-weighted-disconnection-v1"
IIT_TO_MNI_WARP = MNI_CUSTOM_DIR / "from-IIT_to-MNI152NLin6ASym_warp.nii.gz"
MNI_TO_IIT_WARP = MNI_CUSTOM_DIR / "from-MNI152NLin6ASym_to-IIT_warp.nii.gz"


class IITDisconnectionInputSpec(BaseInterfaceInputSpec):
    lesion_file = File(exists=True, mandatory=True, desc="MNI152NLin6Asym 1 mm lesion mask")
    output_disconnection_probability = File(
        mandatory=True, desc="Output MNI voxelwise disconnection or lesion-normalized connectivity NIfTI"
    )
    output_chacovol_endpoint_voxelwise = File(
        mandatory=True, desc="Output MNI endpoint ChaCo or lesion-normalized connectivity NIfTI"
    )
    atlas_files = InputMultiPath(File(exists=True), desc="MNI label atlases")
    atlas_label_files = InputMultiPath(File(exists=True), desc="BIDS atlas label TSV files")
    atlas_names = traits.List(traits.Str, desc="Atlas entity labels")
    output_chacovol_regionwise_csvs = InputMultiPath(File(), desc="One regionwise CSV per atlas")
    output_chacoconn_regionwise_csvs = InputMultiPath(File(), desc="One pairwise ChaCo matrix per atlas")
    output_qc = File(mandatory=True, desc="Output QC image")
    lesion_threshold = traits.Float(0.0, usedefault=True)
    force_lesion_probability_one = traits.Bool(
        True,
        usedefault=True,
        desc="Set disconnection_probability to 1 inside the input lesion mask for any_hit only",
    )
    damage_model = traits.Enum(
        "any_hit", "streamline_mean", "total_length_ratio", "lesion_voxel_mean", usedefault=True,
        desc=("Lesion-connectivity model: binary any-hit, equal-streamline mean lesion-length fraction, "
              "total damaged length / total streamline length, or Petersen-style mean connectivity per lesion voxel"),
    )
    tractogram_files = InputMultiPath(File(exists=True), desc="Optional tractograms; defaults to the two 5-million-streamline IIT files")
    mrtrix_bin_dir = traits.Str(
        "",
        usedefault=True,
        desc="Optional MRtrix3 binary directory; empty uses commands installed on PATH",
    )
    nthreads = traits.Int(0, usedefault=True, desc="Zero uses the MRtrix default")
    atlas_assignment_radius_mm = traits.Float(2.0, usedefault=True)
    overwrite = traits.Bool(True, usedefault=True)


class IITDisconnectionOutputSpec(TraitedSpec):
    disconnection_probability = File(exists=True)
    chacovol_endpoint_voxelwise = File(exists=True)
    chacovol_regionwise_csvs = OutputMultiPath(File(exists=True))
    chacoconn_regionwise_csvs = OutputMultiPath(File(exists=True))
    qc = File(exists=True)


def _save_like(data, reference, path: Path, dtype=np.float32):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
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


def _write_network_csv(path, numerator, denominator, names):
    label_ids = sorted(label for label in names if label > 0)
    selected_numerator = numerator[np.ix_(label_ids, label_ids)]
    selected_denominator = denominator[np.ix_(label_ids, label_ids)]
    matrix = _ratio(selected_numerator, selected_denominator)
    headings = [f"{label}:{names.get(label, f'label-{label}')}" for label in label_ids]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["region", *headings])
        for heading, values in zip(headings, matrix):
            writer.writerow([heading, *(f"{float(value):.10g}" for value in values)])


def _make_qc(background, lesion, maps, path, damage_model):
    z = int(np.rint(np.argwhere(lesion).mean(0))[2])
    model_title = {
        "any_hit": "Any-hit",
        "streamline_mean": "Equal-streamline mean",
        "total_length_ratio": "Length-pooled ratio",
        "lesion_voxel_mean": "Lesion-voxel mean connectivity",
    }[damage_model]
    titles = (f"{model_title}: traversal", f"{model_title}: endpoints", f"{model_title}: atlas regions")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#111318")
    image = None
    if damage_model == "lesion_voxel_mean":
        vmax_values = [float(np.percentile(data[data > 0], 99)) if np.any(data > 0) else 1.0
                       for data in maps]
        mask_threshold = 0.0
        colorbar_label = "Mean streamline connectivity per lesion voxel"
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
        if damage_model == "lesion_voxel_mean":
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
            colorbar.set_label(colorbar_label, color="white", fontsize=8)
            colorbar.ax.tick_params(colors="white", labelsize=8)
    if damage_model != "lesion_voxel_mean":
        colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label(colorbar_label, color="white")
        colorbar.ax.tick_params(colors="white")
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


class IITDisconnection(BaseInterface):
    """Compute IIT traversal, endpoint and Desikan lesion disconnection."""
    input_spec = IITDisconnectionInputSpec
    output_spec = IITDisconnectionOutputSpec

    def _tractograms(self):
        if isdefined(self.inputs.tractogram_files) and self.inputs.tractogram_files:
            return tuple(Path(path) for path in self.inputs.tractogram_files)
        return IIT_TRACTOGRAMS

    def _data_files(self):
        files = [IIT_T1, IIT_T1_256, *self._tractograms(), IIT_TO_MNI_WARP, MNI_TO_IIT_WARP]
        if isdefined(self.inputs.atlas_files):
            files.extend(Path(path) for path in self.inputs.atlas_files)
        if isdefined(self.inputs.atlas_label_files):
            files.extend(Path(path) for path in self.inputs.atlas_label_files)
        return tuple(files)

    def _operator_cache(self):
        cache_dir = IIT_OPERATOR_CACHE.resolve()
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != CACHE_FORMAT or manifest.get("grid_shape") != [256, 256, 256]:
            raise RuntimeError(f"Unsupported IIT operator cache: {manifest_path}")
        cached_sources = [(item.get("name"), int(item.get("tracks", -1))) for item in manifest.get("sources", [])]
        actual_sources = [(path.name, _tck_count(path)) for path in self._tractograms()]
        if cached_sources != actual_sources:
            return None
        return cache_dir, manifest

    def _process_operator_cache(self, lesion256, cache_dir, manifest, endpoint_atlases=None):
        nvox = 256**3
        lesion = np.asarray(lesion256, dtype=np.float32).ravel(order="C")
        maps_sum = [np.zeros(nvox, np.float64) for _ in range(4)]
        total_tracks = affected_tracks = 0
        network_sums = [
            (np.zeros((int(atlas.max()) + 1,) * 2, np.float64),
             np.zeros((int(atlas.max()) + 1,) * 2, np.float64))
            for atlas in (endpoint_atlases or [])
        ]
        for chunk in manifest.get("chunks", []):
            files = chunk["files"]
            ntracks = int(chunk["tracks"])
            p_indptr = np.fromfile(_cache_file(cache_dir, files["p_indptr.u32"]), dtype="<u4")
            p_indices = _load_u24(_cache_file(cache_dir, files["p_indices.u24"]))
            if p_indptr.size != ntracks+1 or int(p_indptr[-1]) != p_indices.size:
                raise RuntimeError(f"Invalid projection operator in cache chunk {chunk.get('id')}")
            projection = sparse.csr_matrix((np.ones(p_indices.size, np.float32), p_indices, p_indptr),
                                           shape=(ntracks, nvox), dtype=np.float32)

            if self.inputs.damage_model == "lesion_voxel_mean":
                # Petersen-style voxelwise extension: count each streamline-lesion-voxel
                # intersection, then normalize the accumulated target map by lesion size.
                lesion_hits = np.asarray(projection @ lesion).ravel()
                track_denominator = np.ones(ntracks, np.float32)
                track_numerator = lesion_hits
                nonzero_damage = lesion_hits
            else:
                a_indptr = np.fromfile(_cache_file(cache_dir, files["a_indptr.u32"]), dtype="<u4")
                a_indices = _load_u24(_cache_file(cache_dir, files["a_indices.u24"]))
                a_lengths = np.fromfile(_cache_file(cache_dir, files["a_lengths.f32"]), dtype="<f4")
                a_total = np.fromfile(_cache_file(cache_dir, files["a_total_lengths.f32"]), dtype="<f4")
                lengths = np.fromfile(_cache_file(cache_dir, files["streamline_lengths.f32"]), dtype="<f4")
                if (a_indptr.size != ntracks+1 or a_total.size != ntracks or lengths.size != ntracks or
                        int(a_indptr[-1]) != a_indices.size or a_indices.size != a_lengths.size or np.any(a_total <= 0)):
                    raise RuntimeError(f"Invalid lesion operator in cache chunk {chunk.get('id')}")
                operator = sparse.csr_matrix((a_lengths, a_indices, a_indptr),
                                             shape=(ntracks, nvox), dtype=np.float32)
                damaged = np.asarray(operator @ lesion).ravel()
                fractions = np.divide(damaged, a_total, out=np.zeros_like(damaged), where=a_total > 0)
                if np.any(fractions < -1e-6) or np.any(fractions > 1+1e-6):
                    raise RuntimeError(f"Lesion fractions outside [0, 1] in cache chunk {chunk.get('id')}")
                fractions = np.clip(fractions, 0, 1)
                nonzero_damage = fractions
                if self.inputs.damage_model == "any_hit":
                    track_denominator = np.ones(ntracks, np.float32)
                    track_numerator = (damaged > 0).astype(np.float32)
                elif self.inputs.damage_model == "streamline_mean":
                    track_denominator = np.ones(ntracks, np.float32)
                    track_numerator = fractions
                else:
                    if np.any(lengths <= 0):
                        raise RuntimeError(f"Non-positive streamline lengths in cache chunk {chunk.get('id')}")
                    track_denominator = lengths
                    track_numerator = fractions * lengths

            maps_sum[0] += np.asarray(projection.T @ track_denominator).ravel()
            maps_sum[1] += np.asarray(projection.T @ track_numerator).ravel()

            endpoints = np.fromfile(_cache_file(cache_dir, files["endpoints.u32"]), dtype="<u4")
            if endpoints.size != 2*ntracks:
                raise RuntimeError(f"Invalid endpoint operator in cache chunk {chunk.get('id')}")
            endpoint_indices = endpoints.reshape(-1)
            endpoint_tracks = np.repeat(np.arange(ntracks), 2)
            valid = endpoint_indices != np.iinfo(np.uint32).max
            endpoint_indices, endpoint_tracks = endpoint_indices[valid], endpoint_tracks[valid]
            maps_sum[2] += np.bincount(endpoint_indices, weights=track_denominator[endpoint_tracks], minlength=nvox)
            maps_sum[3] += np.bincount(endpoint_indices, weights=track_numerator[endpoint_tracks], minlength=nvox)
            endpoint_pairs = endpoints.reshape(ntracks, 2)
            endpoint_valid = endpoint_pairs != np.iinfo(np.uint32).max
            for atlas, (network_denominator, network_numerator) in zip(endpoint_atlases or [], network_sums):
                pair_labels = np.zeros((ntracks, 2), np.int32)
                pair_labels[endpoint_valid] = atlas.ravel()[endpoint_pairs[endpoint_valid]]
                first, second = pair_labels[:, 0], pair_labels[:, 1]
                valid_pair = (first > 0) & (second > 0) & (first != second)
                low, high = np.minimum(first[valid_pair], second[valid_pair]), np.maximum(first[valid_pair], second[valid_pair])
                size = network_denominator.shape[0]
                flat = low * size + high
                network_denominator += np.bincount(
                    flat, weights=track_denominator[valid_pair], minlength=size * size
                ).reshape(size, size)
                network_numerator += np.bincount(
                    flat, weights=track_numerator[valid_pair], minlength=size * size
                ).reshape(size, size)
            total_tracks += ntracks
            part_affected = int(np.count_nonzero(nonzero_damage))
            affected_tracks += part_affected
            print(f"Processed operator-cache chunk {chunk.get('id')}: model={self.inputs.damage_model}, "
                  f"total={ntracks}, nonzero={part_affected}")
        if total_tracks != int(manifest.get("total_tracks", -1)):
            raise RuntimeError("Operator-cache track count does not match manifest")
        return (*(values.reshape((256, 256, 256)) for values in maps_sum),
                total_tracks, affected_tracks, network_sums)

    @staticmethod
    def _resolve_output(value, cwd):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else Path(cwd) / path).resolve()

    def _requested_outputs(self, cwd):
        outputs = {
            "disconnection_probability": self._resolve_output(self.inputs.output_disconnection_probability, cwd),
            "chacovol_endpoint_voxelwise": self._resolve_output(self.inputs.output_chacovol_endpoint_voxelwise, cwd),
            "qc": self._resolve_output(self.inputs.output_qc, cwd),
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
        if outputs["qc"].suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            raise ValueError("output_qc has an unsupported extension")
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

    @staticmethod
    def _tool_path(path):
        return str(Path(path).resolve())

    def _mrtrix_command(self, tool, arguments):
        bin_dir = self.inputs.mrtrix_bin_dir.strip()
        binary = Path(bin_dir) / tool if bin_dir else None
        native = str(binary) if binary is not None and binary.is_file() else shutil.which(tool)
        if not native:
            raise RuntimeError(f"MRtrix3 command not found: {tool}")
        return [native, *arguments]

    def _run_mrtrix(self, tool, arguments):
        command = self._mrtrix_command(tool, arguments)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"MRtrix3 {tool} failed.\nCommand: {' '.join(command)}\n"
                               f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    def _tckmap(self, tracks, output, ends_only=False, weights=None, datatype="UInt32"):
        args = [self._tool_path(tracks), self._tool_path(output), "-template", self._tool_path(IIT_T1_256),
                "-datatype", datatype]
        if ends_only:
            args.append("-ends_only")
        if weights is not None:
            args.extend(("-tck_weights_in", self._tool_path(weights)))
        if self.inputs.nthreads > 0:
            args.extend(("-nthreads", str(self.inputs.nthreads)))
        args.extend(("-force", "-quiet"))
        self._run_mrtrix("tckmap", args)

    def _select_affected(self, tracks, lesion, output):
        args = [self._tool_path(tracks), self._tool_path(output), "-include", self._tool_path(lesion)]
        if self.inputs.nthreads > 0:
            args.extend(("-nthreads", str(self.inputs.nthreads)))
        args.extend(("-force", "-quiet"))
        self._run_mrtrix("tckedit", args)

    @staticmethod
    def _load_count_map(path):
        image = nib.load(str(path))
        if image.shape[:3] != (256, 256, 256):
            raise ValueError(f"Unexpected MRtrix count-map grid: {image.shape[:3]}")
        return np.rint(np.asanyarray(image.dataobj)).astype(np.uint64)

    @staticmethod
    def _load_weight_map(path):
        image = nib.load(str(path))
        if image.shape[:3] != (256, 256, 256):
            raise ValueError(f"Unexpected MRtrix weight-map grid: {image.shape[:3]}")
        data = np.asanyarray(image.dataobj).astype(np.float64)
        if not np.all(np.isfinite(data)) or np.any(data < 0):
            raise RuntimeError(f"Invalid values in MRtrix weight map: {path}")
        return data

    def _sample_lesion_fraction(self, tracks, lesion, output):
        args = [self._tool_path(tracks), self._tool_path(lesion), self._tool_path(output),
                "-stat_tck", "mean", "-precise", "-force", "-quiet"]
        self._run_mrtrix("tcksample", args)

    def _streamline_lengths(self, tracks, output):
        args = [self._tool_path(tracks), "-dump", self._tool_path(output), "-force", "-quiet"]
        self._run_mrtrix("tckstats", args)

    @staticmethod
    def _save_weights(values, path):
        np.savetxt(path, values, fmt="%.10g")

    def _process_tractograms(self, lesion256, iit256_image, temp_dir, endpoint_atlases=None):
        cache = self._operator_cache()
        if cache is not None:
            return self._process_operator_cache(lesion256, *cache, endpoint_atlases=endpoint_atlases)
        if endpoint_atlases:
            raise RuntimeError("Pairwise ChaCo output requires the IIT operator cache")
        if self.inputs.damage_model == "lesion_voxel_mean":
            raise RuntimeError(
                "lesion_voxel_mean requires an IIT operator cache containing the "
                "streamline-voxel projection operator"
            )
        lesion_path = temp_dir / "lesion_iit256.nii.gz"
        _save_like(lesion256, iit256_image, lesion_path, np.uint8)
        damage_model = self.inputs.damage_model
        map_dtype = np.uint64 if damage_model == "any_hit" else np.float64
        maps_sum = [np.zeros((256, 256, 256), map_dtype) for _ in range(4)]
        total_tracks = affected_tracks = 0
        for index, tractogram in enumerate(self._tractograms(), 1):
            prefix = temp_dir / f"part-{index:02d}"
            part_total = _tck_count(tractogram)
            if damage_model == "any_hit":
                affected_tck = Path(f"{prefix}_affected.tck")
                maps = [Path(f"{prefix}_{name}.nii") for name in
                        ("total", "affected", "endpoint_total", "endpoint_affected")]
                self._tckmap(tractogram, maps[0])
                self._tckmap(tractogram, maps[2], True)
                self._select_affected(tractogram, lesion_path, affected_tck)
                self._tckmap(affected_tck, maps[1])
                self._tckmap(affected_tck, maps[3], True)
                for destination, path in zip(maps_sum, maps):
                    destination += self._load_count_map(path)
                part_affected = _tck_count(affected_tck)
            else:
                fraction_path = Path(f"{prefix}_lesion_fraction.txt")
                self._sample_lesion_fraction(tractogram, lesion_path, fraction_path)
                fractions = _load_vector(fraction_path, part_total, "lesion-fraction")
                if np.any(fractions < -1e-7) or np.any(fractions > 1 + 1e-7):
                    raise RuntimeError(f"Lesion fractions outside [0, 1] for {tractogram}")
                fractions = np.clip(fractions, 0, 1)
                self._save_weights(fractions, fraction_path)
                part_affected = int(np.count_nonzero(fractions))
                if damage_model == "streamline_mean":
                    maps = [Path(f"{prefix}_{name}.nii") for name in
                            ("total", "fraction_sum", "endpoint_total", "endpoint_fraction_sum")]
                    self._tckmap(tractogram, maps[0])
                    self._tckmap(tractogram, maps[2], True)
                    self._tckmap(tractogram, maps[1], weights=fraction_path, datatype="Float64")
                    self._tckmap(tractogram, maps[3], True, fraction_path, "Float64")
                    loaders = (self._load_count_map, self._load_weight_map,
                               self._load_count_map, self._load_weight_map)
                else:
                    length_path = Path(f"{prefix}_length_mm.txt")
                    damaged_path = Path(f"{prefix}_damaged_length_mm.txt")
                    self._streamline_lengths(tractogram, length_path)
                    lengths = _load_vector(length_path, part_total, "streamline-length")
                    if np.any(lengths <= 0):
                        raise RuntimeError(f"Non-positive streamline lengths for {tractogram}")
                    self._save_weights(lengths, length_path)
                    self._save_weights(fractions * lengths, damaged_path)
                    maps = [Path(f"{prefix}_{name}.nii") for name in
                            ("length_sum", "damaged_length_sum", "endpoint_length_sum", "endpoint_damaged_length_sum")]
                    self._tckmap(tractogram, maps[0], weights=length_path, datatype="Float64")
                    self._tckmap(tractogram, maps[2], True, length_path, "Float64")
                    self._tckmap(tractogram, maps[1], weights=damaged_path, datatype="Float64")
                    self._tckmap(tractogram, maps[3], True, damaged_path, "Float64")
                    loaders = (self._load_weight_map,) * 4
                for destination, path, loader in zip(maps_sum, maps, loaders):
                    destination += loader(path)
            total_tracks += part_total
            affected_tracks += part_affected
            print(f"Processed {tractogram.name}: model={damage_model}, total={part_total}, nonzero={part_affected}")
        return *maps_sum, total_tracks, affected_tracks, []

    def _run_interface(self, runtime):
        missing = [str(path) for path in self._data_files() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing IIT data:\n" + "\n".join(missing))
        outputs = self._requested_outputs(runtime.cwd)
        self._prepare_outputs(outputs)
        lesion_image = nib.load(str(Path(self.inputs.lesion_file).resolve()))
        iit_image, iit256_image = nib.load(str(IIT_T1)), nib.load(str(IIT_T1_256))
        iit_to_mni, mni_to_iit = nib.load(str(IIT_TO_MNI_WARP)), nib.load(str(MNI_TO_IIT_WARP))
        if lesion_image.shape[:3] != (182, 218, 182):
            raise ValueError(f"Expected 1 mm MNI grid (182, 218, 182), got {lesion_image.shape[:3]}")
        if iit_image.shape[:3] != (182, 218, 182) or iit256_image.shape[:3] != (256, 256, 256):
            raise ValueError("Unexpected IIT template grid")
        atlas_files = list(self.inputs.atlas_files) if isdefined(self.inputs.atlas_files) else []
        atlas_label_files = list(self.inputs.atlas_label_files) if isdefined(self.inputs.atlas_label_files) else []
        atlas_names = list(self.inputs.atlas_names) if isdefined(self.inputs.atlas_names) else []
        region_outputs = outputs["chacovol_regionwise_csvs"]
        if not (len(atlas_files) == len(atlas_label_files) == len(atlas_names) == len(region_outputs)):
            raise ValueError("atlas_files, atlas_label_files, atlas_names and region CSV outputs must have equal lengths")
        network_outputs = outputs["chacoconn_regionwise_csvs"]
        if network_outputs and len(network_outputs) != len(atlas_files):
            raise ValueError("Pairwise ChaCo outputs must contain one CSV per atlas")
        lesion_mni = np.asanyarray(lesion_image.dataobj) > self.inputs.lesion_threshold
        if not lesion_mni.any():
            raise ValueError("Lesion is empty after thresholding")
        lesion_iit = _warp_pullback(lesion_image, iit_to_mni, iit_image, 0, np.uint8) > 0
        lesion256 = _to_iit256(lesion_iit.astype(np.uint8))
        endpoint_atlases = []
        if network_outputs:
            for atlas_name, atlas_path in zip(atlas_names, atlas_files):
                atlas_mni_image = nib.load(atlas_path)
                if atlas_mni_image.shape[:3] != lesion_image.shape[:3] or not np.allclose(
                        atlas_mni_image.affine, lesion_image.affine, rtol=0, atol=1e-3):
                    raise ValueError(f"Atlas {atlas_name} does not match the MNI lesion grid")
                atlas_iit = _warp_pullback(atlas_mni_image, iit_to_mni, iit_image, 0, np.int32)
                atlas256 = _to_iit256(np.rint(atlas_iit).astype(np.int32))
                endpoint_atlases.append(_dilate_labels(
                    atlas256, iit256_image.header.get_zooms()[:3],
                    self.inputs.atlas_assignment_radius_mm
                ))

        with tempfile.TemporaryDirectory(prefix="iit_disconnectome_", dir=runtime.cwd) as tmp:
            processed = self._process_tractograms(
                lesion256, iit256_image, Path(tmp), endpoint_atlases=endpoint_atlases
            )
        (denominator, numerator, endpoint_denominator, endpoint_numerator,
         track_count, affected_count, network_sums) = processed
        lesion_voxel_count = int(np.count_nonzero(lesion256))
        if lesion_voxel_count <= 0:
            raise RuntimeError("Lesion is empty in IIT-256 space")
        if self.inputs.damage_model != "lesion_voxel_mean":
            tolerance = np.maximum(1e-7, np.asarray(denominator, dtype=np.float64) * 1e-7)
            endpoint_tolerance = np.maximum(1e-7, np.asarray(endpoint_denominator, dtype=np.float64) * 1e-7)
            if (np.any(numerator > denominator + tolerance) or
                    np.any(endpoint_numerator > endpoint_denominator + endpoint_tolerance)):
                raise RuntimeError("Damage numerator exceeds denominator")
        expected = sum(_tck_count(path) for path in self._tractograms())
        if track_count != expected:
            raise RuntimeError(f"Expected {expected} tracks, got {track_count}")

        def warp_raw(data256):
            image = nib.Nifti1Image(_from_iit256(np.asarray(data256, np.float32)),
                                    iit_image.affine, iit_image.header)
            return np.maximum(_warp_pullback(image, mni_to_iit, lesion_image, 1, np.float32), 0)

        denominator_mni = warp_raw(denominator)
        numerator_mni = warp_raw(numerator)
        endpoint_denominator_mni = warp_raw(endpoint_denominator)
        endpoint_numerator_mni = warp_raw(endpoint_numerator)
        if self.inputs.damage_model == "lesion_voxel_mean":
            traversal_mni = numerator_mni / float(lesion_voxel_count)
            endpoint_mni = endpoint_numerator_mni / float(lesion_voxel_count)
        else:
            traversal_mni = np.clip(_ratio(numerator_mni, denominator_mni), 0, 1)
            endpoint_mni = np.clip(_ratio(endpoint_numerator_mni, endpoint_denominator_mni), 0, 1)
        if self.inputs.damage_model == "any_hit" and self.inputs.force_lesion_probability_one:
            traversal_mni[lesion_mni] = 1
        _save_like(traversal_mni, lesion_image, outputs["disconnection_probability"])
        _save_like(endpoint_mni, lesion_image, outputs["chacovol_endpoint_voxelwise"])
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
                _load_bids_atlas_labels(labels_path), self.inputs.damage_model, lesion_voxel_count)
            _write_csv(rows, csv_path)
            if qc_region_map is endpoint_mni:
                qc_region_map = region_map

        for labels_path, csv_path, (network_denominator, network_numerator) in zip(
                atlas_label_files, network_outputs, network_sums):
            _write_network_csv(
                csv_path, network_numerator, network_denominator,
                _load_bids_atlas_labels(labels_path)
            )

        background = _warp_pullback(iit_image, mni_to_iit, lesion_image, 1, np.float32)
        _make_qc(background, lesion_mni, (mni_maps[0], mni_maps[1], qc_region_map), outputs["qc"], self.inputs.damage_model)
        self._results = {key: ([str(path) for path in value] if isinstance(value, list) else str(value))
                         for key, value in outputs.items()}
        atlas_summary = f", atlases={len(atlas_files)}"
        print(f"IITDisconnection completed: model={self.inputs.damage_model}, tracks={track_count}, "
              f"nonzero={affected_count}{atlas_summary}")
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs.update(getattr(self, "_results", {}))
        return outputs


__all__ = ["IITDisconnection"]
