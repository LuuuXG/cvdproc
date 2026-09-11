"""Nipype interface for subject-specific MRtrix3 or DSI Studio disconnectomes."""
from __future__ import annotations

import gzip
import csv
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, File,
                                    InputMultiPath, OutputMultiPath, TraitedSpec,
                                    isdefined, traits)

from .disconnection_nipype import (_atlas_region_values, _load_bids_atlas_labels,
                                   _make_qc, _ratio, _save_like, _tck_count, _write_csv)


METHODS = ("any_hit", "streamline_mean", "total_length_ratio", "lesion_voxel_mean")
VENTRICLE_LABELS = (4, 5, 14, 15, 24, 43, 44)
SUBCORTICAL_GM_LABELS = (
    8, 10, 11, 12, 13, 17, 18, 26, 28,
    47, 49, 50, 51, 52, 53, 54, 58, 60,
)
DSI_ENDPOINT_RADIUS_MM = 4.0


class IndividualDisconnectionInputSpec(BaseInterfaceInputSpec):
    lesion_file = File(exists=True, mandatory=True, desc="MNI152NLin6Asym lesion mask")
    tractogram_file = File(
        exists=True, mandatory=True,
        desc="DSI Studio .trk.gz or MRtrix .tck tractogram in the DWI analysis space",
    )
    dwi_reference = File(exists=True, mandatory=True, desc="DWI-space reference")
    t1w_reference = File(exists=True, mandatory=True)
    mni_to_t1w_warp = File(exists=True, mandatory=True)
    t1w_to_mni_warp = File(exists=True, mandatory=True)
    t1w_to_dwi_matrix = File(exists=True, mandatory=True)
    dwi_to_t1w_matrix = File(exists=True, mandatory=True)
    anatomical_segmentation = File(
        exists=True, desc="DWI-grid aparc+aseg used for DSI anatomical filtering"
    )
    cortical_gm_mask = File(
        exists=True, desc="DWI-grid cortical GM mask used for DSI endpoint correction"
    )
    brain_mask = File(
        exists=True, desc="DWI-grid brain mask used for DSI anatomical filtering"
    )
    atlas_files = InputMultiPath(File(exists=True), mandatory=True)
    atlas_label_files = InputMultiPath(File(exists=True), mandatory=True)
    atlas_names = traits.List(traits.Str, mandatory=True)
    output_disconnection_probabilities = InputMultiPath(File(), mandatory=True)
    output_chacovol_endpoint_voxelwise = InputMultiPath(File(), mandatory=True)
    output_native_disconnection_probabilities = InputMultiPath(File(), mandatory=True)
    output_native_chacovol_endpoint_voxelwise = InputMultiPath(File(), mandatory=True)
    output_chacovol_regionwise_csvs = InputMultiPath(File(), mandatory=True)
    output_chacoconn_regionwise_csvs = InputMultiPath(
        File(), desc="Method-major pairwise ChaCo matrices for the first three METHODS"
    )
    output_qcs = InputMultiPath(File(), mandatory=True)
    lesion_threshold = traits.Float(0.0, usedefault=True)
    force_lesion_probability_one = traits.Bool(True, usedefault=True)
    mrtrix_bin_dir = traits.Str("", usedefault=True)
    nthreads = traits.Int(0, usedefault=True)
    atlas_assignment_radius_mm = traits.Float(
        2.0, usedefault=True,
        desc="Nearest-label dilation radius after mapping atlas to the DWI grid",
    )
    overwrite = traits.Bool(True, usedefault=True)


class IndividualDisconnectionOutputSpec(TraitedSpec):
    disconnection_probabilities = OutputMultiPath(File(exists=True))
    chacovol_endpoint_voxelwise = OutputMultiPath(File(exists=True))
    native_disconnection_probabilities = OutputMultiPath(File(exists=True))
    native_chacovol_endpoint_voxelwise = OutputMultiPath(File(exists=True))
    chacovol_regionwise_csvs = OutputMultiPath(File(exists=True))
    chacoconn_regionwise_csvs = OutputMultiPath(File(exists=True))
    qcs = OutputMultiPath(File(exists=True))


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


def _load_vector(path, expected, label):
    values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    if values.size != expected or not np.all(np.isfinite(values)):
        raise RuntimeError(f"Invalid {label}: expected {expected} finite values, got {values.size}")
    return values


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
    lesion_flat = lesion.ravel()
    weights = np.zeros(expected, np.float32)
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
        all_indices, point_valid = _voxel_indices(streamline, inverse, lesion.shape)
        in_grid = all_indices[point_valid]
        if in_grid.size:
            indices = np.unique(np.ravel_multi_index(in_grid.T, lesion.shape))
            weights[count - 1] = np.count_nonzero(lesion_flat[indices])

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
            print(f"Audited anatomy and lesion for {count}/{expected} streamlines", flush=True)
    if count != expected:
        raise RuntimeError(f"Expected {expected} streamlines, read {count}")
    return weights, endpoint_labels, endpoint_indices, valid_tracks, stats


def _endpoint_map(endpoint_indices, streamline_weights, shape):
    valid = endpoint_indices >= 0
    weights = np.broadcast_to(np.asarray(streamline_weights)[:, None], endpoint_indices.shape)
    values = np.bincount(
        endpoint_indices[valid], weights=weights[valid], minlength=int(np.prod(shape))
    )
    return values.reshape(shape).astype(np.float64, copy=False)


def _dilate_atlas_labels(atlas, zooms, radius_mm):
    if radius_mm <= 0:
        return atlas
    distance, indices = distance_transform_edt(
        atlas == 0, sampling=zooms, return_indices=True
    )
    output = atlas.copy()
    fill = (atlas == 0) & (distance <= radius_mm)
    nearest = atlas[tuple(indices)]
    output[fill] = nearest[fill]
    return output


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


def _write_network_csv(path, matrix, label_ids, label_names):
    headings = [f"{label}:{label_names.get(label, f'label-{label}')}" for label in label_ids]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["region", *headings])
        for heading, values in zip(headings, matrix):
            writer.writerow([heading, *(f"{float(value):.10g}" for value in values)])


class IndividualDisconnection(BaseInterface):
    """Calculate four models in native DWI space, then map final maps to MNI."""
    input_spec = IndividualDisconnectionInputSpec
    output_spec = IndividualDisconnectionOutputSpec

    @staticmethod
    def _resolve(values, cwd):
        return [(Path(value) if Path(value).is_absolute() else Path(cwd) / value).resolve()
                for value in values]

    def _command(self, name):
        bin_dir = self.inputs.mrtrix_bin_dir.strip()
        candidate = Path(bin_dir) / name if bin_dir else None
        command = str(candidate) if candidate is not None and candidate.is_file() else shutil.which(name)
        if not command:
            raise RuntimeError(f"Required command not found on PATH: {name}")
        return command

    @staticmethod
    def _run(command):
        result = subprocess.run([str(value) for value in command], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Command failed: {' '.join(map(str, command))}\n{result.stdout}\n{result.stderr}")

    def _tckmap(self, tck, output, template, weights, ends=False):
        command = [self._command("tckmap"), tck, output, "-template", template,
                   "-datatype", "Float64", "-tck_weights_in", weights]
        if ends:
            command.append("-ends_only")
        if self.inputs.nthreads > 0:
            command.extend(("-nthreads", self.inputs.nthreads))
        command.extend(("-force", "-quiet"))
        self._run(command)

    def _mni_to_dwi(self, source, output, temp):
        t1 = temp / "lesion_t1w.nii.gz"
        self._run([self._command("mri_convert"), "-at", self.inputs.mni_to_t1w_warp,
                   source, t1, "-rt", "nearest"])
        self._run([self._command("flirt"), "-in", t1, "-ref", self.inputs.dwi_reference,
                   "-applyxfm", "-init", self.inputs.t1w_to_dwi_matrix,
                   "-interp", "nearestneighbour", "-out", output])

    def _dwi_to_mni(self, source, output, temp, stem):
        t1 = temp / f"{stem}_t1w.nii.gz"
        self._run([self._command("flirt"), "-in", source, "-ref", self.inputs.t1w_reference,
                   "-applyxfm", "-init", self.inputs.dwi_to_t1w_matrix,
                   "-interp", "trilinear", "-out", t1])
        self._run([self._command("mri_convert"), "-at", self.inputs.t1w_to_mni_warp,
                   t1, output, "-rt", "interpolate"])

    def _run_interface(self, runtime):
        traversal_outputs = self._resolve(self.inputs.output_disconnection_probabilities, runtime.cwd)
        endpoint_outputs = self._resolve(self.inputs.output_chacovol_endpoint_voxelwise, runtime.cwd)
        native_traversal_outputs = self._resolve(
            self.inputs.output_native_disconnection_probabilities, runtime.cwd
        )
        native_endpoint_outputs = self._resolve(
            self.inputs.output_native_chacovol_endpoint_voxelwise, runtime.cwd
        )
        region_outputs = self._resolve(self.inputs.output_chacovol_regionwise_csvs, runtime.cwd)
        network_outputs = (self._resolve(self.inputs.output_chacoconn_regionwise_csvs, runtime.cwd)
                           if isdefined(self.inputs.output_chacoconn_regionwise_csvs) else [])
        qc_outputs = self._resolve(self.inputs.output_qcs, runtime.cwd)
        atlas_files = list(self.inputs.atlas_files)
        label_files = list(self.inputs.atlas_label_files)
        atlas_names = list(self.inputs.atlas_names)
        if any(len(values) != 4 for values in (
                traversal_outputs, endpoint_outputs, native_traversal_outputs,
                native_endpoint_outputs, qc_outputs)):
            raise ValueError("MNI/native traversal, endpoint and QC output lists must follow the four METHODS")
        if not (len(atlas_files) == len(label_files) == len(atlas_names)):
            raise ValueError("Atlas file, TSV and name lists must have equal lengths")
        if len(region_outputs) != 4 * len(atlas_files):
            raise ValueError("Region outputs must be ordered method-major and contain four files per atlas")
        if network_outputs and len(network_outputs) != 3 * len(atlas_files):
            raise ValueError("Network outputs must be method-major for any_hit, streamline_mean and total_length_ratio")
        all_outputs = (traversal_outputs + endpoint_outputs + native_traversal_outputs +
                       native_endpoint_outputs + region_outputs + network_outputs + qc_outputs)
        if len(set(all_outputs)) != len(all_outputs):
            raise ValueError("Every output filename must be distinct")
        for path in all_outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not self.inputs.overwrite:
                raise FileExistsError(path)

        mni_image = nib.load(self.inputs.lesion_file)
        lesion_mni = np.asanyarray(mni_image.dataobj) > self.inputs.lesion_threshold
        if mni_image.shape[:3] != (182, 218, 182) or not lesion_mni.any():
            raise ValueError("The MNI lesion must be nonempty on the 182x218x182 grid")
        tractogram_path = Path(self.inputs.tractogram_file)
        is_dsi_trk = tractogram_path.name.lower().endswith(".trk.gz")
        is_mrtrix_tck = tractogram_path.suffix.lower() == ".tck"
        if not (is_dsi_trk or is_mrtrix_tck):
            raise ValueError("tractogram_file must end in .trk.gz or .tck")
        dwi_image = nib.load(self.inputs.dwi_reference)
        if is_dsi_trk:
            anatomy_inputs = (
                self.inputs.anatomical_segmentation,
                self.inputs.cortical_gm_mask,
                self.inputs.brain_mask,
            )
            if not all(isdefined(value) and value for value in anatomy_inputs):
                raise ValueError(
                    "DSI Studio input requires anatomical_segmentation, cortical_gm_mask, "
                    "and brain_mask for post-hoc anatomical filtering"
                )
            trk = _trk_header(tractogram_path)
            if (dwi_image.shape[:3] != trk["dimensions"] or
                    not np.allclose(dwi_image.header.get_zooms()[:3], trk["voxel_sizes"], atol=1e-4, rtol=0) or
                    (trk["voxel_order"] and "".join(nib.aff2axcodes(dwi_image.affine)) != trk["voxel_order"])):
                raise ValueError("DSI Studio TRK grid does not match the DWI reference")
            n_streamlines = trk["streamline_count"]
        else:
            trk = None
            n_streamlines = _tck_count(tractogram_path)
        if n_streamlines <= 0:
            raise ValueError("Tractogram header has no streamline count")

        with tempfile.TemporaryDirectory(prefix="individual_disconnection_", dir=runtime.cwd) as tmp_name:
            temp = Path(tmp_name)
            lesion_dwi_file = temp / "lesion_dwi.nii.gz"
            self._mni_to_dwi(self.inputs.lesion_file, lesion_dwi_file, temp)
            lesion_dwi_image = nib.load(str(lesion_dwi_file))
            lesion_dwi = np.asanyarray(lesion_dwi_image.dataobj) > self.inputs.lesion_threshold
            lesion_voxels = int(lesion_dwi.sum())
            if lesion_voxels == 0:
                raise RuntimeError("Lesion is empty after MNI-to-DWI transformation")
            endpoint_atlases = []
            for atlas_index, atlas_file in enumerate(atlas_files):
                atlas_dwi_file = temp / f"atlas_{atlas_index}_dwi.nii.gz"
                self._mni_to_dwi(atlas_file, atlas_dwi_file, temp)
                atlas_dwi = np.rint(np.asanyarray(nib.load(str(atlas_dwi_file)).dataobj)).astype(np.int32)
                endpoint_atlases.append(_dilate_atlas_labels(
                    atlas_dwi, dwi_image.header.get_zooms()[:3], self.inputs.atlas_assignment_radius_mm
                ))
            if is_dsi_trk:
                anatomy = _load_dsi_anatomy(
                    self.inputs.anatomical_segmentation,
                    self.inputs.cortical_gm_mask,
                    self.inputs.brain_mask,
                    dwi_image,
                )
                trk_file, tck_file = temp / "streamlines.trk", temp / "streamlines.tck"
                with gzip.open(tractogram_path, "rb") as source, trk_file.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=16 << 20)
                _convert_dsi_trk(trk_file, tck_file, dwi_image.affine, trk["voxel_to_rasmm"])
            else:
                anatomy = None
                # MRtrix TCK points are already stored in scanner/world RAS millimetres.
                # No TrackVis half-voxel correction or coordinate rewrite is appropriate.
                tck_file = tractogram_path.resolve()

            fraction_file, length_file = temp / "fraction.txt", temp / "length.txt"
            self._run([self._command("tcksample"), tck_file, lesion_dwi_file, fraction_file,
                       "-stat_tck", "mean", "-precise", "-force", "-quiet"])
            self._run([self._command("tckstats"), tck_file, "-dump", length_file, "-force", "-quiet"])
            fractions = np.clip(_load_vector(fraction_file, n_streamlines, "lesion fractions"), 0, 1)
            lengths = _load_vector(length_file, n_streamlines, "streamline lengths")
            if np.any(lengths <= 0):
                raise RuntimeError("Non-positive streamline length")
            lesion_hits, endpoint_labels, endpoint_indices, valid_tracks, anatomy_stats = _audit_streamlines(
                tck_file, lesion_dwi, dwi_image.affine, n_streamlines,
                endpoint_atlases, anatomy
            )
            if anatomy is not None:
                valid_count = anatomy_stats["valid"]
                assigned = anatomy_stats["direct_gm"] + anatomy_stats["snapped"]
                print(
                    "DSI post-hoc anatomy: "
                    f"valid={valid_count}/{n_streamlines}, "
                    f"ventricle_core_rejected={anatomy_stats['ventricle_core']}, "
                    f"internal_outside_brain={anatomy_stats['internal_outside_brain']}, "
                    f"GM_endpoints={assigned}/{2 * valid_count}, "
                    f"both_endpoints_assigned={anatomy_stats['both_assigned']}/{valid_count}",
                    flush=True,
                )
            valid_weights = valid_tracks.astype(float)
            weights = {
                "ones": valid_weights,
                "any_hit": valid_weights * (fractions > 0),
                "streamline_mean": valid_weights * fractions,
                "length": valid_weights * lengths,
                "total_length_ratio": valid_weights * fractions * lengths,
                "lesion_voxel_mean": valid_weights * lesion_hits,
            }
            weight_files = {}
            for name, values in weights.items():
                weight_files[name] = temp / f"{name}.txt"
                np.savetxt(weight_files[name], values, fmt="%.10g")
            dwi_maps = {}
            for name in weights:
                dwi_maps[name] = temp / f"{name}_traversal.nii.gz"
                self._tckmap(
                    tck_file, dwi_maps[name], lesion_dwi_file, weight_files[name]
                )
                if anatomy is None:
                    dwi_maps[f"{name}_endpoint"] = temp / f"{name}_endpoint.nii.gz"
                    self._tckmap(
                        tck_file, dwi_maps[f"{name}_endpoint"], lesion_dwi_file,
                        weight_files[name], ends=True
                    )

            total_dwi = _load_map(dwi_maps["ones"], dwi_image.shape[:3])
            endpoint_total_dwi = (
                _endpoint_map(endpoint_indices, weights["ones"], dwi_image.shape[:3])
                if anatomy is not None
                else _load_map(dwi_maps["ones_endpoint"], dwi_image.shape[:3])
            )
            for method_index, method in enumerate(METHODS):
                if method == "total_length_ratio":
                    denominator = _load_map(dwi_maps["length"], dwi_image.shape[:3])
                    endpoint_denominator = (
                        _endpoint_map(endpoint_indices, weights["length"], dwi_image.shape[:3])
                        if anatomy is not None
                        else _load_map(dwi_maps["length_endpoint"], dwi_image.shape[:3])
                    )
                else:
                    denominator, endpoint_denominator = total_dwi, endpoint_total_dwi
                numerator = _load_map(dwi_maps[method], dwi_image.shape[:3])
                endpoint_numerator = (
                    _endpoint_map(endpoint_indices, weights[method], dwi_image.shape[:3])
                    if anatomy is not None
                    else _load_map(dwi_maps[f"{method}_endpoint"], dwi_image.shape[:3])
                )
                if method == "lesion_voxel_mean":
                    traversal = numerator / float(lesion_voxels)
                    endpoint = endpoint_numerator / float(lesion_voxels)
                else:
                    traversal = np.clip(_ratio(numerator, denominator), 0, 1)
                    endpoint = np.clip(_ratio(endpoint_numerator, endpoint_denominator), 0, 1)
                if method == "any_hit" and self.inputs.force_lesion_probability_one:
                    traversal[lesion_dwi] = 1
                _save_like(traversal, dwi_image, native_traversal_outputs[method_index])
                _save_like(endpoint, dwi_image, native_endpoint_outputs[method_index])
                self._dwi_to_mni(
                    native_traversal_outputs[method_index], traversal_outputs[method_index],
                    temp, f"{method}_traversal"
                )
                self._dwi_to_mni(
                    native_endpoint_outputs[method_index], endpoint_outputs[method_index],
                    temp, f"{method}_endpoint"
                )
                traversal_mni = np.maximum(
                    np.asanyarray(nib.load(str(traversal_outputs[method_index])).dataobj).astype(np.float32), 0
                )
                endpoint_mni = np.maximum(
                    np.asanyarray(nib.load(str(endpoint_outputs[method_index])).dataobj).astype(np.float32), 0
                )
                if method != "lesion_voxel_mean":
                    traversal_mni = np.clip(traversal_mni, 0, 1)
                    endpoint_mni = np.clip(endpoint_mni, 0, 1)
                if method == "any_hit" and self.inputs.force_lesion_probability_one:
                    traversal_mni[lesion_mni] = 1
                _save_like(traversal_mni, mni_image, traversal_outputs[method_index])
                _save_like(endpoint_mni, mni_image, endpoint_outputs[method_index])
                region_map = endpoint
                for atlas_index, labels_path in enumerate(label_files):
                    atlas = endpoint_atlases[atlas_index]
                    current_map, rows = _atlas_region_values(
                        atlas, endpoint_denominator, endpoint_numerator,
                        _load_bids_atlas_labels(labels_path), method, lesion_voxels)
                    _write_csv(rows, region_outputs[method_index * len(atlas_files) + atlas_index])
                    if atlas_index == 0:
                        region_map = current_map
                    if network_outputs and method != "lesion_voxel_mean":
                        if method == "total_length_ratio":
                            network_denominator_weights = weights["length"]
                        else:
                            network_denominator_weights = weights["ones"]
                        network, _, _, label_ids, included = _network_values(
                            endpoint_labels[atlas_index], weights[method],
                            network_denominator_weights, _load_bids_atlas_labels(labels_path)
                        )
                        network_index = method_index * len(atlas_files) + atlas_index
                        _write_network_csv(
                            network_outputs[network_index], network, label_ids,
                            _load_bids_atlas_labels(labels_path)
                        )
                        print(f"Pairwise {method}/{atlas_names[atlas_index]}: "
                              f"included inter-region streamlines={included}", flush=True)
                background = np.log1p(np.maximum(total_dwi, 0))
                _make_qc(background, lesion_dwi, (traversal, endpoint, region_map),
                         qc_outputs[method_index], method)

        self._results = {
            "disconnection_probabilities": [str(path) for path in traversal_outputs],
            "chacovol_endpoint_voxelwise": [str(path) for path in endpoint_outputs],
            "native_disconnection_probabilities": [str(path) for path in native_traversal_outputs],
            "native_chacovol_endpoint_voxelwise": [str(path) for path in native_endpoint_outputs],
            "chacovol_regionwise_csvs": [str(path) for path in region_outputs],
            "chacoconn_regionwise_csvs": [str(path) for path in network_outputs],
            "qcs": [str(path) for path in qc_outputs],
        }
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs.update(getattr(self, "_results", {}))
        return outputs


__all__ = ["IndividualDisconnection"]
