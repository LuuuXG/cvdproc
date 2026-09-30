import os
import re
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    InputMultiPath,
    Directory,
    isdefined,
    traits,
)
from neuromaps import transforms
from traits.api import Either, List
import pickle
from typing import List as TList, Optional

from cvdproc.config.paths import get_package_path
from cvdproc.config.paths import LH_MEDIAL_WALL_TXT, RH_MEDIAL_WALL_TXT, find_qcache_metric_pairs


_TRACTOGRAPHY_METHODS = ("ifod2act", "sdstream")


def _bids_value(value):
    return re.sub(r"[^A-Za-z0-9]+", "", str(value))


def _nemo_bids_name(prefix, atlas=None, method=None, description=None, statistic="mean", suffix="metrics", extension=".csv", **entities):
    parts = [prefix]
    for key in ("hemi", "space", "res", "den"):
        value = entities.pop(key, None)
        if value:
            parts.append(f"{key}-{_bids_value(value)}")
    if atlas:
        parts.append(f"atlas-{_bids_value(atlas)}")
    if method:
        parts.append(f"model-{_bids_value(method)}")
    for key, value in entities.items():
        if value:
            parts.append(f"{key}-{_bids_value(value)}")
    if description:
        parts.append(f"desc-{_bids_value(description)}")
    if statistic:
        parts.append(f"stat-{_bids_value(statistic)}")
    return "_".join(parts) + f"_{suffix}{extension}"


def _parse_nemo_mean_map(filename):
    pattern = re.compile(
        rf"^(?P<source>.*?)(?P<method>{'|'.join(_TRACTOGRAPHY_METHODS)})_chacovol_"
        r"res(?P<resolution>[0-9]+(?:p[0-9]+)?)mm_mean\.nii\.gz$",
        re.IGNORECASE,
    )
    return pattern.match(filename)


def _mni152_to_fsaverage_164k(mni_image, lh_output, rh_output):
    fsaverage_lh, fsaverage_rh = transforms.mni152_to_fsaverage(mni_image, '164k')
    lh_medial_wall = np.loadtxt(LH_MEDIAL_WALL_TXT, dtype=int)
    rh_medial_wall = np.loadtxt(RH_MEDIAL_WALL_TXT, dtype=int)

    fsaverage_lh.darrays[0].data[lh_medial_wall] = np.nan
    fsaverage_rh.darrays[0].data[rh_medial_wall] = np.nan

    lh_output = os.path.abspath(lh_output)
    rh_output = os.path.abspath(rh_output)
    os.makedirs(os.path.dirname(lh_output), exist_ok=True)
    os.makedirs(os.path.dirname(rh_output), exist_ok=True)
    fsaverage_lh.to_filename(lh_output)
    fsaverage_rh.to_filename(rh_output)
    return lh_output, rh_output


class MNI152ToFsaverage164kInputSpec(BaseInterfaceInputSpec):
    mni_image = File(exists=True, mandatory=True, desc='MNI152 scalar image')
    lh_output = File(mandatory=True, desc='Left fsaverage 164k metric GIFTI')
    rh_output = File(mandatory=True, desc='Right fsaverage 164k metric GIFTI')


class MNI152ToFsaverage164kOutputSpec(TraitedSpec):
    lh_output = File(exists=True, desc='Left fsaverage 164k metric GIFTI')
    rh_output = File(exists=True, desc='Right fsaverage 164k metric GIFTI')


class MNI152ToFsaverage164k(BaseInterface):
    input_spec = MNI152ToFsaverage164kInputSpec
    output_spec = MNI152ToFsaverage164kOutputSpec

    def _run_interface(self, runtime):
        self._lh_output, self._rh_output = _mni152_to_fsaverage_164k(
            self.inputs.mni_image, self.inputs.lh_output, self.inputs.rh_output
        )
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['lh_output'] = getattr(self, '_lh_output', os.path.abspath(self.inputs.lh_output))
        outputs['rh_output'] = getattr(self, '_rh_output', os.path.abspath(self.inputs.rh_output))
        return outputs


class NemoCorticalMetricsInputSpec(BaseInterfaceInputSpec):
    nemo_output_dir = Directory(exists=True, mandatory=True, desc='Directory containing Nemo output files')
    nemo_postprocessed_dir = Directory(mandatory=True, desc='Directory to save postprocessed Nemo output files')
    freesurfer_output_dirs = InputMultiPath(Directory(exists=True), mandatory=False, desc='List of Freesurfer output directories')
    output_csv_dir = Directory(mandatory=False, desc='Directory to save one CSV per chacovol file')
    bids_prefix = traits.Str(desc='BIDS subject/session prefix for postprocessed outputs')
    mean_only = traits.Bool(False, usedefault=True, desc='Transform only unsmoothed mean ChacoVol maps')


class NemoCorticalMetricsOutputSpec(TraitedSpec):
    nemo_postprocessed_dir = Directory(exists=True, desc='Directory containing postprocessed Nemo output files')
    output_csv_dir = Directory(desc='Directory containing one CSV per chacovol file')
    surface_files = traits.List(File(exists=True), desc='Generated fsaverage 164k surface maps')
    metric_csvs = traits.List(File(exists=True), desc='Generated FreeSurfer weighted metric CSV files')


class NemoCorticalMetrics(BaseInterface):
    input_spec = NemoCorticalMetricsInputSpec
    output_spec = NemoCorticalMetricsOutputSpec

    def _run_interface(self, runtime):
        nemo_output_dir = self.inputs.nemo_output_dir
        nemo_postprocessed_dir = self.inputs.nemo_postprocessed_dir

        if not os.path.exists(nemo_postprocessed_dir):
            os.makedirs(nemo_postprocessed_dir)

        mni_images = sorted(f for f in os.listdir(nemo_output_dir) if f.endswith('.nii.gz'))
        if self.inputs.mean_only:
            mni_images = [f for f in mni_images if _parse_nemo_mean_map(f)]
            if len(mni_images) == 0:
                raise FileNotFoundError(
                    f"No unsmoothed mean ChacoVol maps for ifod2act or sdstream found in: {nemo_output_dir}"
                )

        surface_pairs = []
        for mni_image in mni_images:
            mni_image_path = os.path.join(nemo_output_dir, mni_image)
            match = _parse_nemo_mean_map(mni_image)
            if isdefined(self.inputs.bids_prefix) and self.inputs.bids_prefix and match:
                method = match.group('method').lower()
                common = dict(prefix=self.inputs.bids_prefix, method=method, description='NemoChacovol', statistic='mean',
                              suffix='map', extension='.func.gii', space='fsaverage', den='164k')
                lh_filename = _nemo_bids_name(**common, hemi='L')
                rh_filename = _nemo_bids_name(**common, hemi='R')
            else:
                method = match.group('method').lower() if match else None
                lh_filename = mni_image.replace('.nii.gz', '_lh_fsaverage.func.gii')
                rh_filename = mni_image.replace('.nii.gz', '_rh_fsaverage.func.gii')
            _mni152_to_fsaverage_164k(
                mni_image_path,
                os.path.join(nemo_postprocessed_dir, lh_filename),
                os.path.join(nemo_postprocessed_dir, rh_filename)
            )
            surface_pairs.append((method, os.path.join(nemo_postprocessed_dir, lh_filename), os.path.join(nemo_postprocessed_dir, rh_filename)))
            print(f"Transformed {mni_image} to fsaverage space and saved as {lh_filename} and {rh_filename}")

        metric_csvs = []
        if isdefined(self.inputs.freesurfer_output_dirs) and len(self.inputs.freesurfer_output_dirs) > 0:
            fs_dirs = list(self.inputs.freesurfer_output_dirs)
            for method, lh_path, rh_path in surface_pairs:
                lh_data = nib.load(lh_path).darrays[0].data
                rh_data = nib.load(rh_path).darrays[0].data
                weight = np.concatenate([lh_data, rh_data])

                rows = []
                for fs_dir in fs_dirs:
                    metric_pairs = find_qcache_metric_pairs(fs_dir)
                    row = {'freesurfer_dir': os.path.basename(fs_dir.rstrip('/'))}
                    for metric_key, hemis in metric_pairs.items():
                        lh_metric = nib.load(hemis['lh']).get_fdata().squeeze()
                        rh_metric = nib.load(hemis['rh']).get_fdata().squeeze()
                        metric_data = np.concatenate([lh_metric, rh_metric])

                        valid = (~np.isnan(weight)) & (~np.isnan(metric_data))
                        wsum = np.nansum(weight[valid])
                        wmean = np.nan if wsum == 0 else np.nansum(weight[valid] * metric_data[valid]) / wsum
                        row[metric_key] = wmean
                    rows.append(row)

                if isdefined(self.inputs.output_csv_dir) and self.inputs.output_csv_dir:
                    os.makedirs(self.inputs.output_csv_dir, exist_ok=True)
                    if isdefined(self.inputs.bids_prefix) and self.inputs.bids_prefix:
                        filename = _nemo_bids_name(self.inputs.bids_prefix, method=method, description='NemoChacovolWeighted',
                                                   statistic='mean', suffix='metrics', extension='.csv', space='fsaverage', den='164k')
                    else:
                        base_id = os.path.basename(lh_path).replace('_lh_fsaverage.func.gii', '')
                        filename = f"{base_id}_cortical_metrics.csv"
                    output_path = os.path.join(self.inputs.output_csv_dir, filename)
                    df = pd.DataFrame(rows)
                    df.to_csv(output_path, index=False)
                    metric_csvs.append(output_path)
                    print(f"Saved weighted cortical metrics to {output_path}")

        self._surface_files = [path for pair in surface_pairs for path in pair[1:]]
        self._metric_csvs = metric_csvs
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['nemo_postprocessed_dir'] = self.inputs.nemo_postprocessed_dir
        if isdefined(self.inputs.output_csv_dir) and self.inputs.output_csv_dir:
            outputs['output_csv_dir'] = self.inputs.output_csv_dir
        outputs['surface_files'] = getattr(self, '_surface_files', [])
        outputs['metric_csvs'] = getattr(self, '_metric_csvs', [])
        return outputs

# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
_ATLAS_TO_FNAME = {
    "fs86subj": "fs86subj.csv",
    "fs86avg": "fs86subj.csv",
    "aal": "aal116.csv",
    "aal116": "aal116.csv",
    "shen268": "tpl-MNI152NLin6Asym_res-1_atlas-Shen268_dseg.tsv",
}


def _load_labels_csv(label_csv: str) -> TList[str]:
    """
    Load label CSV and return a list of ROI names.

    Supported formats:
      - Two columns: [id, label] (with or without header)
      - One column: [label]
    """
    with open(label_csv, "r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
    delimiter = "\t" if "\t" in first_line else ","
    first_fields = [field.strip().lower() for field in first_line.split(delimiter)]
    has_header = any(field in {"index", "id", "label", "name", "region", "roi"} for field in first_fields)
    df = pd.read_csv(label_csv, sep=delimiter, header=0 if has_header else None)

    preferred_columns = [column for column in df.columns if str(column).strip().lower() in {"name", "label", "region", "roi"}]
    if preferred_columns:
        labels = df[preferred_columns[0]].astype(str).tolist()
    elif df.shape[1] >= 2:
        labels = df.iloc[:, 1].astype(str).tolist()
    else:
        labels = df.iloc[:, 0].astype(str).tolist()

    labels = [x.strip() for x in labels if str(x).strip() != ""]
    return labels


def _resolve_label_csv(
    atlas: str,
    nemo_output_dir: str,
    nemo_postprocessed_dir: str,
) -> Optional[str]:
    atlas_key = atlas.lower()
    candidates = []
    if atlas_key in _ATLAS_TO_FNAME:
        fname = _ATLAS_TO_FNAME[atlas_key]
        candidates.extend([
            get_package_path("data", "atlas", "nemo", fname),
            os.path.join(nemo_output_dir, fname),
            os.path.join(nemo_postprocessed_dir, fname),
        ])

    search_roots = [nemo_output_dir, nemo_postprocessed_dir, get_package_path("data", "atlas", "nemo")]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "*")):
            basename = os.path.basename(path).lower()
            if os.path.isfile(path) and basename.endswith((".csv", ".tsv")) and atlas_key in basename:
                candidates.append(path)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return None


def _labels_for_atlas(atlas, expected_count, nemo_output_dir, nemo_postprocessed_dir):
    label_csv = _resolve_label_csv(atlas, nemo_output_dir, nemo_postprocessed_dir)
    if label_csv is None:
        print(f"No label table found for atlas '{atlas}'; using numeric ROI labels.")
        return [f"ROI-{index}" for index in range(1, expected_count + 1)]
    labels = _load_labels_csv(label_csv)
    if len(labels) != expected_count:
        raise ValueError(
            f"Label count mismatch for atlas='{atlas}': n_labels={len(labels)} vs data_dim={expected_count}. "
            f"label_csv={label_csv}"
        )
    return labels


def _nemo_result_files(nemo_output_dir, measure, extension, structural_connectivity=False):
    method_pattern = "|".join(_TRACTOGRAPHY_METHODS)
    if structural_connectivity:
        regex = re.compile(
            rf".*(?P<method>{method_pattern})_{measure}_(?P<atlas>.+?)_(?P<variant>nemoSC.*?)_mean\.{extension}$",
            re.IGNORECASE,
        )
    else:
        regex = re.compile(
            rf".*(?P<method>{method_pattern})_{measure}_(?P<atlas>.+?)_mean\.{extension}$",
            re.IGNORECASE,
        )
    matches = []
    for path in sorted(glob.glob(os.path.join(nemo_output_dir, f"*.{extension}"))):
        match = regex.match(os.path.basename(path))
        if match:
            matches.append((path, match))
    return matches


def _load_single_2d_matrix_from_mat(mat_path: str) -> np.ndarray:
    """
    Load a single 2D numeric matrix from a .mat file.
    Picks the first valid 2D ndarray among non-__ keys.
    """
    mat = loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    if len(keys) == 0:
        raise ValueError(f"No valid variables found in mat file: {mat_path}")

    for k in keys:
        v = mat[k]
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return np.asarray(v)

    raise ValueError(f"No valid 2D matrix found in mat file: {mat_path}")


def _ensure_out_dir(root: str, subdir: str) -> str:
    out_dir = os.path.join(root, subdir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------
# NemoChacovol
# ---------------------------------------------------------------------
class NemoChacovolInputSpec(BaseInterfaceInputSpec):
    nemo_output_dir = Directory(exists=True, mandatory=True, desc="Directory containing Nemo output files")
    nemo_postprocessed_dir = Directory(exists=True, mandatory=True, desc="Directory containing Nemo postprocessed files")
    bids_prefix = traits.Str(desc="BIDS subject/session prefix for output filenames")


class NemoChacovolOutputSpec(TraitedSpec):
    output_csvs = traits.List(File(exists=True), desc="List of output CSV files")


class NemoChacovol(BaseInterface):
    input_spec = NemoChacovolInputSpec
    output_spec = NemoChacovolOutputSpec

    def _run_interface(self, runtime):
        nemo_output_dir = os.path.abspath(self.inputs.nemo_output_dir)
        nemo_post_dir = os.path.abspath(self.inputs.nemo_postprocessed_dir)

        out_dir = _ensure_out_dir(nemo_post_dir, "chacovol_csv")

        result_files = _nemo_result_files(nemo_output_dir, "chacovol", "pkl")
        if len(result_files) == 0:
            raise FileNotFoundError(
                f"No ifod2act or sdstream ChacoVol pkl files found under: {nemo_output_dir}"
            )

        output_csvs: TList[str] = []

        for pkl_path, match in result_files:
            base = os.path.basename(pkl_path)
            atlas = match.group("atlas")
            method = match.group("method").lower()

            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            arr = np.asarray(data)

            # Normalize shapes
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            if arr.ndim != 2:
                raise ValueError(
                    f"Invalid chacovol data in '{pkl_path}': expected 1D or 2D, got shape={arr.shape}"
                )

            n0, n1 = arr.shape
            if n0 == n1:
                expected_count = n0
            elif n0 == 1:
                expected_count = n1
            elif n1 == 1:
                expected_count = n0
            else:
                raise ValueError(
                    f"Invalid chacovol array in '{pkl_path}': expected square, row-vector, or column-vector; got shape={arr.shape}"
                )
            labels = _labels_for_atlas(atlas, expected_count, nemo_output_dir, nemo_post_dir)
            if isdefined(self.inputs.bids_prefix) and self.inputs.bids_prefix:
                filename = _nemo_bids_name(self.inputs.bids_prefix, atlas=atlas, method=method,
                                           description="NemoChacovol", statistic="mean", suffix="metrics")
            else:
                filename = base.replace(".pkl", ".csv")
            out_csv = os.path.join(out_dir, filename)

            # (N, N) -> treat as matrix
            if n0 == n1:
                if len(labels) != n0:
                    raise ValueError(
                        f"Label count mismatch for atlas='{atlas}' in '{pkl_path}': "
                        f"n_labels={len(labels)} vs matrix_dim={n0}."
                    )
                df = pd.DataFrame(arr, index=labels, columns=labels)
                df.to_csv(out_csv, index=True, header=True)
                output_csvs.append(out_csv)
                continue

            # (1, N) -> vector row
            if n0 == 1 and n1 == len(labels):
                df = pd.DataFrame(arr, columns=labels, index=["mean"])
                df.to_csv(out_csv, index=True, header=True)
                output_csvs.append(out_csv)
                continue

            # (N, 1) -> vector col
            if n1 == 1 and n0 == len(labels):
                df = pd.DataFrame(arr[:, 0], index=labels, columns=["mean"])
                df.to_csv(out_csv, index=True, header=True)
                output_csvs.append(out_csv)
                continue

            raise ValueError(
                f"Invalid chacovol array in '{pkl_path}': expected (N,N), (1,N), or (N,1) with N=labels, "
                f"got shape={arr.shape}, N_labels={len(labels)}"
            )

        if len(output_csvs) == 0:
            raise RuntimeError(
                f"Found pkl files but none matched atlas regex. Directory: {nemo_output_dir}"
            )

        self._output_csvs = output_csvs
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_csvs"] = getattr(self, "_output_csvs", [])
        return outputs


# ---------------------------------------------------------------------
# NemoChacoconn
# ---------------------------------------------------------------------
class NemoChacoconnSCInputSpec(BaseInterfaceInputSpec):
    nemo_output_dir = Directory(exists=True, mandatory=True, desc="Directory containing Nemo output files")
    nemo_postprocessed_dir = Directory(exists=True, mandatory=True, desc="Directory containing Nemo postprocessed files")
    bids_prefix = traits.Str(desc="BIDS subject/session prefix for output filenames")


class NemoChacoconnSCOutputSpec(TraitedSpec):
    output_csvs = traits.List(File(exists=True), desc="List of output chacoconn CSV files")


class NemoChacoconnSC(BaseInterface):
    input_spec = NemoChacoconnSCInputSpec
    output_spec = NemoChacoconnSCOutputSpec

    def _run_interface(self, runtime):
        nemo_output_dir = os.path.abspath(self.inputs.nemo_output_dir)
        nemo_post_dir = os.path.abspath(self.inputs.nemo_postprocessed_dir)

        out_dir = _ensure_out_dir(nemo_post_dir, "chacoconn_csv")

        result_files = _nemo_result_files(nemo_output_dir, "chacoconn", "mat", structural_connectivity=True)
        if len(result_files) == 0:
            print(f"No ifod2act or sdstream ChacoConn SC mat files found in: {nemo_output_dir}")
            self._output_csvs = []
            return runtime

        output_csvs: TList[str] = []

        for mat_path, match in result_files:
            base = os.path.basename(mat_path)
            atlas = match.group("atlas")
            method = match.group("method").lower()
            variant = match.group("variant")

            mat = _load_single_2d_matrix_from_mat(mat_path)

            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"Invalid chacoconn matrix in '{mat_path}': expected square ROI×ROI, got shape={mat.shape}"
                )

            labels = _labels_for_atlas(atlas, mat.shape[0], nemo_output_dir, nemo_post_dir)
            
            mat = np.asarray(mat, dtype=np.float64)

            # If matrix looks like upper-triangular-only, symmetrize it
            lower = np.tril(mat, k=-1)
            upper = np.triu(mat, k=1)

            if np.allclose(lower, 0) and not np.allclose(upper, 0):
                mat = mat + mat.T - np.diag(np.diag(mat))

            df = pd.DataFrame(mat, index=labels, columns=labels)

            if isdefined(self.inputs.bids_prefix) and self.inputs.bids_prefix:
                variant_desc = re.sub(r"^nemo", "", variant, flags=re.IGNORECASE)
                filename = _nemo_bids_name(self.inputs.bids_prefix, atlas=atlas, method=method,
                                           description=f"NemoChacoconn{variant_desc}", statistic="mean",
                                           suffix="connectivity")
            else:
                filename = base.replace(".mat", ".csv")
            out_csv = os.path.join(out_dir, filename)
            df.to_csv(out_csv, index=True, header=True)
            output_csvs.append(out_csv)

        if len(output_csvs) == 0:
            raise RuntimeError(
                f"Found mat files but none matched atlas regex. Directory: {nemo_output_dir}"
            )

        self._output_csvs = output_csvs
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_csvs"] = getattr(self, "_output_csvs", [])
        return outputs

class NemoChacoconnInputSpec(BaseInterfaceInputSpec):
    nemo_output_dir = Directory(exists=True, mandatory=True, desc="Directory containing Nemo output files")
    nemo_postprocessed_dir = Directory(exists=True, mandatory=True, desc="Directory containing Nemo postprocessed files")
    bids_prefix = traits.Str(desc="BIDS subject/session prefix for output filenames")


class NemoChacoconnOutputSpec(TraitedSpec):
    output_csvs = traits.List(File(exists=True), desc="List of output chacoconn CSV files")


class NemoChacoconn(BaseInterface):
    """
    Convert Nemo chacoconn (connection loss / disconnection) matrices from sparse .pkl to dense .csv.

    Expected input patterns under nemo_output_dir:
        *ifod2act_chacoconn_<atlas>_mean.pkl
        *sdstream_chacoconn_<atlas>_mean.pkl

    Output folder:
        <nemo_postprocessed_dir>/chacoconn_csv/

    Output filename:
        <input_basename>.csv
    """

    input_spec = NemoChacoconnInputSpec
    output_spec = NemoChacoconnOutputSpec

    def _run_interface(self, runtime):
        nemo_output_dir = os.path.abspath(self.inputs.nemo_output_dir)
        nemo_post_dir = os.path.abspath(self.inputs.nemo_postprocessed_dir)

        out_dir = os.path.join(nemo_post_dir, "chacoconn_csv")
        os.makedirs(out_dir, exist_ok=True)

        result_files = _nemo_result_files(nemo_output_dir, "chacoconn", "pkl")
        if len(result_files) == 0:
            print(f"No ifod2act or sdstream ChacoConn pkl files found in: {nemo_output_dir}")
            self._output_csvs = []
            return runtime

        output_csvs: TList[str] = []

        for pkl_path, match in result_files:
            base = os.path.basename(pkl_path)
            atlas = match.group("atlas")
            method = match.group("method").lower()

            data = pickle.load(open(pkl_path, "rb"))

            # chacoconn is typically stored as scipy.sparse; must convert to dense before saving
            if hasattr(data, "toarray"):
                arr = data.toarray()
            else:
                arr = np.asarray(data)

            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    f"Invalid chacoconn matrix in '{pkl_path}': expected square 2D matrix, got shape={arr.shape}"
                )

            labels = _labels_for_atlas(atlas, arr.shape[0], nemo_output_dir, nemo_post_dir)

            # If only upper triangle is populated (common for some connectome exports), symmetrize
            lower = np.tril(arr, k=-1)
            upper = np.triu(arr, k=1)
            if np.allclose(lower, 0) and not np.allclose(upper, 0):
                arr = arr + arr.T - np.diag(np.diag(arr))

            df = pd.DataFrame(arr, index=labels, columns=labels)

            if isdefined(self.inputs.bids_prefix) and self.inputs.bids_prefix:
                filename = _nemo_bids_name(self.inputs.bids_prefix, atlas=atlas, method=method,
                                           description="NemoChacoconn", statistic="mean", suffix="connectivity")
            else:
                filename = base.replace(".pkl", ".csv")
            out_csv = os.path.join(out_dir, filename)
            df.to_csv(out_csv, index=True, header=True)

            output_csvs.append(out_csv)

        if len(output_csvs) == 0:
            raise RuntimeError(
                f"Found pkl files but none matched atlas regex. Directory: {nemo_output_dir}"
            )

        self._output_csvs = output_csvs
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_csvs"] = getattr(self, "_output_csvs", [])
        return outputs
