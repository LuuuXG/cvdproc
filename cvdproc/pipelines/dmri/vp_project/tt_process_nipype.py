import os
import gzip
import numpy as np
import nibabel as nib
from scipy.io import loadmat
from nibabel.streamlines import Tractogram, TckFile

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
)
from traits.api import Bool, Str


# -------------------------
# TT utilities
# -------------------------

def parse_tt(track_bytes):
    """Parse a DSI Studio TinyTrack byte stream into voxel-space streamlines."""
    buf1 = np.asarray(track_bytes, dtype=np.uint8).ravel()
    buf2 = buf1.view(np.int8)

    streamlines = []
    i = 0
    total_len = len(buf1)

    while i + 16 <= total_len:
        size = np.frombuffer(buf1[i:i + 4].tobytes(), dtype=np.uint32)[0]

        if size == 0 or (size % 3) != 0:
            break

        n_points = int(size // 3)
        record_len = int(size + 13)

        if n_points < 2 or i + record_len > total_len:
            break

        x = np.frombuffer(buf1[i + 4:i + 8].tobytes(), dtype=np.int32)[0]
        y = np.frombuffer(buf1[i + 8:i + 12].tobytes(), dtype=np.int32)[0]
        z = np.frombuffer(buf1[i + 12:i + 16].tobytes(), dtype=np.int32)[0]

        coords = np.zeros((n_points, 3), dtype=np.float64)
        coords[0] = [x, y, z]

        q = i + 16
        for j in range(1, n_points):
            x += int(buf2[q])
            y += int(buf2[q + 1])
            z += int(buf2[q + 2])
            q += 3
            coords[j] = [x, y, z]

        streamlines.append(coords / 32.0)
        i += record_len

    return streamlines


def _as_dim3(dim):
    dim = np.asarray(dim).astype(np.int64).reshape(-1)
    if dim.size < 3:
        raise ValueError(f"Invalid TT dimension: {dim}")
    return dim[:3]


def flip_streamlines_y(streamlines, dim):
    """Flip y between DSI Studio voxel convention and nibabel voxel convention."""
    dim = _as_dim3(dim)
    flipped = []

    for s in streamlines:
        s2 = np.asarray(s, dtype=np.float64).copy()
        if s2.ndim != 2 or s2.shape[1] != 3:
            continue
        s2[:, 1] = (dim[1] - 1) - s2[:, 1]
        flipped.append(s2)

    return flipped


def load_tt_streamlines(tt_gz_path, flip_y_axis=False):
    """Load TT streamlines and optionally convert them to nibabel voxel coordinates."""
    if not os.path.isfile(tt_gz_path):
        raise FileNotFoundError(f"Input TT file not found: {tt_gz_path}")

    with gzip.open(tt_gz_path, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)

    if "track" not in mat:
        raise KeyError("No 'track' field found in TT file.")

    track_bytes = np.asarray(mat["track"], dtype=np.uint8).ravel()
    streamlines = parse_tt(track_bytes)

    if len(streamlines) == 0:
        raise RuntimeError("No streamlines parsed from TT.")

    dimension = mat.get("dimension", None)
    if flip_y_axis:
        if dimension is None:
            raise KeyError("No 'dimension' field found in TT file. Cannot flip y axis.")
        streamlines = flip_streamlines_y(streamlines, dimension)

    return streamlines, mat


# -------------------------
# Coordinate utilities
# -------------------------

def vox_to_mm(points_ijk, affine):
    """Convert voxel coordinates to world/mm coordinates using a NIfTI affine."""
    pts = np.asarray(points_ijk, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    return (affine @ pts_h.T).T[:, :3]


def mm_to_vox(points_mm, inv_affine):
    """Convert world/mm coordinates to voxel coordinates using an inverse NIfTI affine."""
    pts = np.asarray(points_mm, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    return (inv_affine @ pts_h.T).T[:, :3]


# -------------------------
# TT.GZ to TCK
# -------------------------

def tt_gz_to_tck_mm(
    tt_gz_path,
    ref_nii,
    out_tck,
    flip_y_axis=False,
):
    """Convert a DSI Studio TT.GZ file to a TCK file in RAS/mm coordinates."""
    if not os.path.isfile(ref_nii):
        raise FileNotFoundError(f"Reference NIfTI not found: {ref_nii}")

    img = nib.load(ref_nii)
    affine = img.affine

    streamlines_vox, _ = load_tt_streamlines(
        tt_gz_path,
        flip_y_axis=flip_y_axis,
    )

    streamlines_mm = [vox_to_mm(s, affine) for s in streamlines_vox]

    tractogram = Tractogram(streamlines_mm, affine_to_rasmm=np.eye(4))

    out_dir = os.path.dirname(os.path.abspath(out_tck))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    TckFile(tractogram).save(out_tck)

    if not os.path.isfile(out_tck):
        raise RuntimeError(f"Failed to write TCK: {out_tck}")

    return out_tck


class TTGZToTCKInputSpec(BaseInterfaceInputSpec):
    tt_gz_path = File(exists=True, mandatory=True, desc="Input DSI Studio TT.GZ file")
    ref_nii = File(exists=True, mandatory=True, desc="Reference NIfTI defining target affine")
    out_tck = File(mandatory=True, desc="Output TCK file path")
    flip_y_axis = Bool(
        False,
        usedefault=True,
        desc="Flip y axis from DSI Studio voxel convention to nibabel voxel convention before affine conversion",
    )


class TTGZToTCKOutputSpec(TraitedSpec):
    out_tck = File(exists=True, desc="Output TCK file")


class TTGZToTCK(BaseInterface):
    input_spec = TTGZToTCKInputSpec
    output_spec = TTGZToTCKOutputSpec

    def _run_interface(self, runtime):
        out_tck = os.path.abspath(self.inputs.out_tck)
        tt_gz_to_tck_mm(
            tt_gz_path=self.inputs.tt_gz_path,
            ref_nii=self.inputs.ref_nii,
            out_tck=out_tck,
            flip_y_axis=bool(self.inputs.flip_y_axis),
        )
        self._out_tck = out_tck
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_tck"] = getattr(
            self,
            "_out_tck",
            os.path.abspath(self.inputs.out_tck),
        )
        return outputs


# -------------------------
# TT.GZ to probabilistic TDI
# -------------------------

def tt_gz_to_prob_tdi(
    tt_gz_path,
    ref_nii,
    out_tdi,
    flip_y_axis=False,
):
    """Create a probabilistic tract-density image from a DSI Studio TT.GZ file."""
    if not os.path.isfile(ref_nii):
        raise FileNotFoundError(f"Reference NIfTI not found: {ref_nii}")

    img = nib.load(ref_nii)
    affine = img.affine
    inv_affine = np.linalg.inv(affine)
    shape = img.shape[:3]

    streamlines_vox, _ = load_tt_streamlines(
        tt_gz_path,
        flip_y_axis=flip_y_axis,
    )

    tdi = np.zeros(shape, dtype=np.float32)
    n_streamlines = len(streamlines_vox)

    for s in streamlines_vox:
        pts = np.asarray(s, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
            continue

        pts_mm = vox_to_mm(pts, affine)
        ijk = mm_to_vox(pts_mm, inv_affine)
        ijk = np.rint(ijk).astype(np.int64)

        valid = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )
        ijk = ijk[valid]

        if ijk.shape[0] == 0:
            continue

        ijk_unique = np.unique(ijk, axis=0)
        tdi[ijk_unique[:, 0], ijk_unique[:, 1], ijk_unique[:, 2]] += 1.0

    if n_streamlines > 0:
        tdi /= float(n_streamlines)

    out_dir = os.path.dirname(os.path.abspath(out_tdi))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_img = nib.Nifti1Image(tdi, affine, img.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, out_tdi)

    if not os.path.isfile(out_tdi):
        raise RuntimeError(f"Failed to write TDI: {out_tdi}")

    return out_tdi


class TTGZToTDIInputSpec(BaseInterfaceInputSpec):
    tt_gz_path = File(exists=True, mandatory=True, desc="Input DSI Studio TT.GZ file")
    ref_nii = File(exists=True, mandatory=True, desc="Reference NIfTI defining target grid")
    out_tdi = File(mandatory=True, desc="Output probabilistic TDI NIfTI")
    flip_y_axis = Bool(
        False,
        usedefault=True,
        desc="Flip y axis from DSI Studio voxel convention to nibabel voxel convention before TDI rasterization",
    )


class TTGZToTDIOutputSpec(TraitedSpec):
    out_tdi = File(exists=True, desc="Output TDI NIfTI")


class TTGZToTDI(BaseInterface):
    input_spec = TTGZToTDIInputSpec
    output_spec = TTGZToTDIOutputSpec

    def _run_interface(self, runtime):
        out_tdi = os.path.abspath(self.inputs.out_tdi)
        tt_gz_to_prob_tdi(
            tt_gz_path=self.inputs.tt_gz_path,
            ref_nii=self.inputs.ref_nii,
            out_tdi=out_tdi,
            flip_y_axis=bool(self.inputs.flip_y_axis),
        )
        self._out_tdi = out_tdi
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_tdi"] = getattr(
            self,
            "_out_tdi",
            os.path.abspath(self.inputs.out_tdi),
        )
        return outputs

import os

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
)

from cvdproc.pipelines.dmri.dsistudio.tt_utils import TinyTrackIO


class FlipTTInputSpec(BaseInterfaceInputSpec):

    in_tt = File(
        exists=True,
        mandatory=True,
        desc="Input DSI Studio TinyTrack file",
    )

    out_tt = File(
        mandatory=True,
        desc="Output flipped TinyTrack file",
    )

    flip_x_axis = traits.Bool(
        False,
        usedefault=True,
        desc="Permanently flip streamline coordinates along x axis",
    )

    flip_y_axis = traits.Bool(
        False,
        usedefault=True,
        desc="Permanently flip streamline coordinates along y axis",
    )

    flip_z_axis = traits.Bool(
        False,
        usedefault=True,
        desc="Permanently flip streamline coordinates along z axis",
    )


class FlipTTOutputSpec(TraitedSpec):

    out_tt = File(
        exists=True,
        desc="Output flipped TinyTrack file",
    )


class FlipTT(BaseInterface):

    input_spec = FlipTTInputSpec
    output_spec = FlipTTOutputSpec

    def _run_interface(self, runtime):

        out_tt = os.path.abspath(self.inputs.out_tt)
        os.makedirs(os.path.dirname(out_tt), exist_ok=True)

        loader = TinyTrackIO(
            flip_x_axis=bool(self.inputs.flip_x_axis),
            flip_y_axis=bool(self.inputs.flip_y_axis),
            flip_z_axis=bool(self.inputs.flip_z_axis),
        )

        data = loader.load(
            self.inputs.in_tt,
            preserve_metadata=True,
        )

        saver = TinyTrackIO(
            flip_x_axis=False,
            flip_y_axis=False,
            flip_z_axis=False,
        )

        saver.save(
            streamlines=data.streamlines,
            output_file=out_tt,
            dimension=data.dimension,
            voxel_size=data.voxel_size,
            metadata=data.metadata,
        )

        self._out_tt = out_tt

        return runtime

    def _list_outputs(self):

        outputs = self.output_spec().get()
        outputs["out_tt"] = getattr(
            self,
            "_out_tt",
            os.path.abspath(self.inputs.out_tt),
        )

        return outputs