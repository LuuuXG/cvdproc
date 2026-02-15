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
    traits,
)
from neuromaps import transforms
from traits.api import Either, List
import pickle
from typing import List as TList, Optional

from cvdproc.config.paths import get_package_path
from cvdproc.config.paths import LH_MEDIAL_WALL_TXT, RH_MEDIAL_WALL_TXT, find_qcache_metric_pairs

class NemoCorticalMetricsInputSpec(BaseInterfaceInputSpec):
    nemo_output_dir = Directory(exists=True, mandatory=True, desc='Directory containing Nemo output files')
    nemo_postprocessed_dir = Directory(mandatory=True, desc='Directory to save postprocessed Nemo output files')
    freesurfer_output_dirs = InputMultiPath(Directory(exists=True), mandatory=False, desc='List of Freesurfer output directories')
    output_csv_dir = Directory(mandatory=False, desc='Directory to save one CSV per chacovol file')

class NemoCorticalMetricsOutputSpec(TraitedSpec):
    nemo_postprocessed_dir = Directory(exists=True, desc='Directory containing postprocessed Nemo output files')
    output_csv_dir = Directory(desc='Directory containing one CSV per chacovol file')

class NemoCorticalMetrics(BaseInterface):
    input_spec = NemoCorticalMetricsInputSpec
    output_spec = NemoCorticalMetricsOutputSpec

    def _run_interface(self, runtime):
        nemo_output_dir = self.inputs.nemo_output_dir
        nemo_postprocessed_dir = self.inputs.nemo_postprocessed_dir

        if not os.path.exists(nemo_postprocessed_dir):
            os.makedirs(nemo_postprocessed_dir)

        lh_medial_wall = np.loadtxt(LH_MEDIAL_WALL_TXT, dtype=int)
        rh_medial_wall = np.loadtxt(RH_MEDIAL_WALL_TXT, dtype=int)

        # Step 1: transform all MNI nii.gz files to fsaverage.func.gii
        mni_images = [f for f in os.listdir(nemo_output_dir) if f.endswith('.nii.gz')]
        for mni_image in mni_images:
            mni_image_path = os.path.join(nemo_output_dir, mni_image)
            fsaverage_lh, fsaverage_rh = transforms.mni152_to_fsaverage(mni_image_path, '164k')

            lh_data = fsaverage_lh.darrays[0].data
            rh_data = fsaverage_rh.darrays[0].data
            lh_data[lh_medial_wall] = None
            rh_data[rh_medial_wall] = None
            fsaverage_lh.darrays[0].data[:] = lh_data
            fsaverage_rh.darrays[0].data[:] = rh_data

            lh_filename = mni_image.replace('.nii.gz', '_lh_fsaverage.func.gii')
            rh_filename = mni_image.replace('.nii.gz', '_rh_fsaverage.func.gii')
            fsaverage_lh.to_filename(os.path.join(nemo_postprocessed_dir, lh_filename))
            fsaverage_rh.to_filename(os.path.join(nemo_postprocessed_dir, rh_filename))
            print(f"Transformed {mni_image} to fsaverage space and saved as {lh_filename} and {rh_filename}")

        if self.inputs.freesurfer_output_dirs:
            fs_dirs = self.inputs.freesurfer_output_dirs
            fsavg_files = os.listdir(nemo_postprocessed_dir)
            lh_pattern = re.compile(r'(.+_chacovol_res\d+mm_smooth\d+mm_mean)_lh_fsaverage\.func\.gii')
            lh_prob_maps = [f for f in fsavg_files if lh_pattern.match(f)]

            for lh_file in lh_prob_maps:
                match = lh_pattern.match(lh_file)
                base_id = match.group(1)
                rh_file = f"{base_id}_rh_fsaverage.func.gii"
                lh_path = os.path.join(nemo_postprocessed_dir, lh_file)
                rh_path = os.path.join(nemo_postprocessed_dir, rh_file)
                if not os.path.exists(rh_path):
                    continue

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

                if self.inputs.output_csv_dir:
                    os.makedirs(self.inputs.output_csv_dir, exist_ok=True)
                    output_path = os.path.join(self.inputs.output_csv_dir, f"{base_id}_cortical_metrics.csv")
                    df = pd.DataFrame(rows)
                    df.to_csv(output_path, index=False)
                    print(f"Saved {base_id} results to {output_path}")

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['nemo_postprocessed_dir'] = self.inputs.nemo_postprocessed_dir
        outputs['output_csv_dir'] = self.inputs.output_csv_dir if self.inputs.output_csv_dir else None
        return outputs

# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
_ATLAS_TO_FNAME = {
    "fs86subj": "fs86subj.csv",
    "aal": "aal116.csv",
}


def _load_labels_csv(label_csv: str) -> TList[str]:
    """
    Load label CSV and return a list of ROI names.

    Supported formats:
      - Two columns: [id, label] (with or without header)
      - One column: [label]
    """
    df = pd.read_csv(label_csv, header=None)

    if df.shape[1] >= 2:
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
    if atlas not in _ATLAS_TO_FNAME:
        return None

    fname = _ATLAS_TO_FNAME[atlas]

    # 1) packaged path
    try:
        pkg_path = get_package_path("data", "atlas", "nemo", fname)
        if os.path.isfile(pkg_path):
            return pkg_path
    except Exception:
        pass

    # 2) nemo_output_dir
    cand = os.path.join(nemo_output_dir, fname)
    if os.path.isfile(cand):
        return cand

    # 3) nemo_postprocessed_dir
    cand = os.path.join(nemo_postprocessed_dir, fname)
    if os.path.isfile(cand):
        return cand

    return None


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


class NemoChacovolOutputSpec(TraitedSpec):
    output_csvs = traits.List(File(exists=True), desc="List of output CSV files")


class NemoChacovol(BaseInterface):
    input_spec = NemoChacovolInputSpec
    output_spec = NemoChacovolOutputSpec

    def _run_interface(self, runtime):
        nemo_output_dir = os.path.abspath(self.inputs.nemo_output_dir)
        nemo_post_dir = os.path.abspath(self.inputs.nemo_postprocessed_dir)

        out_dir = _ensure_out_dir(nemo_post_dir, "chacovol_csv")

        pkl_glob = os.path.join(nemo_output_dir, "*ifod2act_chacovol_*_mean.pkl")
        pkl_files = sorted(glob.glob(pkl_glob))
        if len(pkl_files) == 0:
            raise FileNotFoundError(
                f"No chacovol pkl files found under: {nemo_output_dir} "
                f"(pattern: *ifod2act_chacovol_*_mean.pkl)"
            )

        rx = re.compile(r".*ifod2act_chacovol_(?P<atlas>.+?)_mean\.pkl$")

        output_csvs: TList[str] = []

        for pkl_path in pkl_files:
            base = os.path.basename(pkl_path)
            m = rx.match(base)
            if m is None:
                continue

            atlas = m.group("atlas")

            label_csv = _resolve_label_csv(atlas, nemo_output_dir, nemo_post_dir)
            if label_csv is None:
                raise FileNotFoundError(
                    f"Cannot resolve label CSV for atlas='{atlas}' from file '{base}'. "
                    f"Supported atlases: {', '.join(sorted(_ATLAS_TO_FNAME.keys()))}"
                )

            labels = _load_labels_csv(label_csv)

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
            out_csv = os.path.join(out_dir, base.replace(".pkl", ".csv"))

            # (N, N) -> treat as matrix
            if n0 == n1:
                if len(labels) != n0:
                    raise ValueError(
                        f"Label count mismatch for atlas='{atlas}' in '{pkl_path}': "
                        f"n_labels={len(labels)} vs matrix_dim={n0}. label_csv={label_csv}"
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


class NemoChacoconnSCOutputSpec(TraitedSpec):
    output_csvs = traits.List(File(exists=True), desc="List of output chacoconn CSV files")


class NemoChacoconnSC(BaseInterface):
    input_spec = NemoChacoconnSCInputSpec
    output_spec = NemoChacoconnSCOutputSpec

    def _run_interface(self, runtime):
        nemo_output_dir = os.path.abspath(self.inputs.nemo_output_dir)
        nemo_post_dir = os.path.abspath(self.inputs.nemo_postprocessed_dir)

        out_dir = _ensure_out_dir(nemo_post_dir, "chacoconn_csv")

        mat_glob = os.path.join(nemo_output_dir, "*ifod2act_chacoconn_*_nemoSC*_mean.mat")
        mat_files = sorted(glob.glob(mat_glob))
        if len(mat_files) == 0:
            raise FileNotFoundError(
                f"No chacoconn mat files found under: {nemo_output_dir} "
                f"(pattern: *ifod2act_chacoconn_*_nemoSC*_mean.mat)"
            )

        rx = re.compile(r".*ifod2act_chacoconn_(?P<atlas>.+?)_nemoSC.*_mean\.mat$")

        output_csvs: TList[str] = []

        for mat_path in mat_files:
            base = os.path.basename(mat_path)
            m = rx.match(base)
            if m is None:
                continue

            atlas = m.group("atlas")

            label_csv = _resolve_label_csv(atlas, nemo_output_dir, nemo_post_dir)
            if label_csv is None:
                raise FileNotFoundError(
                    f"Cannot resolve label CSV for atlas='{atlas}' from file '{base}'. "
                    f"Supported atlases: {', '.join(sorted(_ATLAS_TO_FNAME.keys()))}"
                )

            labels = _load_labels_csv(label_csv)

            mat = _load_single_2d_matrix_from_mat(mat_path)

            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"Invalid chacoconn matrix in '{mat_path}': expected square ROI×ROI, got shape={mat.shape}"
                )

            if mat.shape[0] != len(labels):
                raise ValueError(
                    f"Label count mismatch for atlas='{atlas}' in '{mat_path}': "
                    f"N_labels={len(labels)} vs matrix_dim={mat.shape[0]}. label_csv={label_csv}"
                )
            
            mat = np.asarray(mat, dtype=np.float64)

            # If matrix looks like upper-triangular-only, symmetrize it
            lower = np.tril(mat, k=-1)
            upper = np.triu(mat, k=1)

            if np.allclose(lower, 0) and not np.allclose(upper, 0):
                mat = mat + mat.T - np.diag(np.diag(mat))

            df = pd.DataFrame(mat, index=labels, columns=labels)

            out_csv = os.path.join(out_dir, base.replace(".mat", ".csv"))
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
    nemo_output_dir = Directory(
        exists=True,
        mandatory=True,
        desc="Directory containing Nemo output files",
    )
    nemo_postprocessed_dir = Directory(
        exists=True,
        mandatory=True,
        desc="Directory containing Nemo postprocessed files",
    )


class NemoChacoconnOutputSpec(TraitedSpec):
    output_csvs = traits.List(
        File(exists=True),
        desc="List of output chacoconn CSV files",
    )


class NemoChacoconn(BaseInterface):
    """
    Convert Nemo chacoconn (connection loss / disconnection) matrices from sparse .pkl to dense .csv.

    Expected input pattern under nemo_output_dir:
        *ifod2act_chacoconn_<atlas>_mean.pkl

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

        pkl_glob = os.path.join(nemo_output_dir, "*ifod2act_chacoconn_*_mean.pkl")
        pkl_files = sorted(glob.glob(pkl_glob))

        if len(pkl_files) == 0:
            raise FileNotFoundError(
                f"No chacoconn pkl files found under: {nemo_output_dir} "
                f"(pattern: *ifod2act_chacoconn_*_mean.pkl)"
            )

        rx = re.compile(r".*ifod2act_chacoconn_(?P<atlas>.+?)_mean\.pkl$")

        output_csvs: TList[str] = []

        for pkl_path in pkl_files:
            base = os.path.basename(pkl_path)
            m = rx.match(base)
            if m is None:
                continue

            atlas = m.group("atlas")

            label_csv = _resolve_label_csv(
                atlas=atlas,
                nemo_output_dir=nemo_output_dir,
                nemo_postprocessed_dir=nemo_post_dir,
            )
            if label_csv is None:
                raise FileNotFoundError(
                    f"Cannot resolve label CSV for atlas='{atlas}' from file '{base}'."
                )

            labels = _load_labels_csv(label_csv)

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

            if arr.shape[0] != len(labels):
                raise ValueError(
                    f"Label count mismatch for atlas='{atlas}' in '{pkl_path}': "
                    f"n_labels={len(labels)} vs matrix_dim={arr.shape[0]}. Label file: {label_csv}"
                )

            # If only upper triangle is populated (common for some connectome exports), symmetrize
            lower = np.tril(arr, k=-1)
            upper = np.triu(arr, k=1)
            if np.allclose(lower, 0) and not np.allclose(upper, 0):
                arr = arr + arr.T - np.diag(np.diag(arr))

            df = pd.DataFrame(arr, index=labels, columns=labels)

            out_csv = os.path.join(out_dir, base.replace(".pkl", ".csv"))
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