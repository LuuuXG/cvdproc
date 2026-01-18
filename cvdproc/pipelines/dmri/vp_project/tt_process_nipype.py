import os
import gzip
import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, TckFile

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
)


def locate_tt_data_offset(track_bytes, min_consecutive=3, max_scan_bytes=8 * 1024 * 1024):
    buf = memoryview(track_bytes)
    L = len(buf)
    scan_limit = min(L, max_scan_bytes)

    def valid_record_at(p):
        if p + 4 > L:
            return None
        size = np.frombuffer(buf[p:p+4], dtype="<u4", count=1)[0]
        if size == 0 or (size % 3) != 0:
            return None
        npts = int(size // 3)
        if npts < 2:
            return None
        total_len = int(size + 13)
        if p + total_len > L:
            return None
        payload = buf[p+4:p+total_len]
        if len(payload) != (12 + (npts - 1) * 3):
            return None
        return total_len

    p = 0
    while p + 4 < scan_limit:
        rec_len = valid_record_at(p)
        if rec_len is None:
            p += 1
            continue

        ok = 1
        q = p + rec_len
        while ok < min_consecutive:
            rec_len2 = valid_record_at(q)
            if rec_len2 is None:
                break
            ok += 1
            q += rec_len2

        if ok >= min_consecutive:
            return p

        p += 1

    raise RuntimeError("Cannot locate TT streamline data offset.")


def parse_tt(track_bytes, start_offset=0):
    buf1 = np.frombuffer(track_bytes, dtype=np.uint8)
    buf2 = buf1.view(np.int8)

    streamlines = []
    i = int(start_offset)
    L = len(buf1)

    while i + 4 <= L:
        size = np.frombuffer(buf1[i:i+4].tobytes(), dtype=np.uint32)[0]
        if size == 0 or (size % 3) != 0:
            break

        npts = int(size // 3)
        rec_len = int(size + 13)
        if i + rec_len > L or npts < 2:
            break

        x = np.frombuffer(buf1[i+4:i+8].tobytes(), dtype=np.int32)[0]
        y = np.frombuffer(buf1[i+8:i+12].tobytes(), dtype=np.int32)[0]
        z = np.frombuffer(buf1[i+12:i+16].tobytes(), dtype=np.int32)[0]

        coords = np.zeros((npts, 3), dtype=np.float64)
        coords[0] = [x, y, z]

        q = i + 16
        for j in range(1, npts):
            x += int(buf2[q])
            y += int(buf2[q + 1])
            z += int(buf2[q + 2])
            q += 3
            coords[j] = [x, y, z]

        streamlines.append(coords / 32.0)
        i += rec_len

    return streamlines


def vox_to_mm(points_ijk, affine):
    pts = np.asarray(points_ijk, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    mm = (affine @ pts_h.T).T[:, :3]
    return mm


def tt_gz_to_tck_mm(tt_gz_path, ref_nii, out_tck):
    if not os.path.isfile(tt_gz_path):
        raise FileNotFoundError(f"Input TT file not found: {tt_gz_path}")
    if not os.path.isfile(ref_nii):
        raise FileNotFoundError(f"Reference NIfTI not found: {ref_nii}")

    img = nib.load(ref_nii)
    affine = img.affine

    with gzip.open(tt_gz_path, "rb") as f:
        track_bytes = f.read()

    start_offset = locate_tt_data_offset(track_bytes)
    streamlines_vox = parse_tt(track_bytes, start_offset=start_offset)

    if len(streamlines_vox) == 0:
        raise RuntimeError("No streamlines parsed from TT.")

    streamlines_mm = [vox_to_mm(s, affine) for s in streamlines_vox]

    tractogram = Tractogram(streamlines_mm, affine_to_rasmm=np.eye(4))

    out_dir = os.path.dirname(out_tck)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    TckFile(tractogram).save(out_tck)

    if not os.path.isfile(out_tck):
        raise RuntimeError(f"Failed to write TCK: {out_tck}")

    return out_tck


class TTGZToTCKInputSpec(BaseInterfaceInputSpec):
    tt_gz_path = File(exists=True, mandatory=True, desc="Input DSI Studio TT.GZ file")
    ref_nii = File(exists=True, mandatory=True, desc="Reference NIfTI (defines target space and affine)")
    out_tck = File(mandatory=True, desc="Output TCK file path")


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
        )
        self._out_tck = out_tck
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_tck"] = getattr(self, "_out_tck", os.path.abspath(self.inputs.out_tck))
        return outputs

# -------------------------
# Core TDI logic
# -------------------------
def tt_gz_to_prob_tdi(tt_gz_path, ref_nii, out_tdi):
    if not os.path.isfile(tt_gz_path):
        raise FileNotFoundError(tt_gz_path)
    if not os.path.isfile(ref_nii):
        raise FileNotFoundError(ref_nii)

    img = nib.load(ref_nii)
    affine = img.affine
    inv_affine = np.linalg.inv(affine)
    shape = img.shape[:3]

    with gzip.open(tt_gz_path, "rb") as f:
        track_bytes = f.read()

    start_offset = locate_tt_data_offset(track_bytes)
    streamlines_vox = parse_tt(track_bytes, start_offset=start_offset)

    if len(streamlines_vox) == 0:
        raise RuntimeError("No streamlines parsed from TT.")

    tdi = np.zeros(shape, dtype=np.float32)
    n_streamlines = len(streamlines_vox)

    for s in streamlines_vox:
        # voxel -> mm
        pts = np.asarray(s, dtype=np.float64)
        ones = np.ones((pts.shape[0], 1))
        pts_h = np.concatenate([pts, ones], axis=1)
        pts_mm = (affine @ pts_h.T).T[:, :3]

        # mm -> voxel (ref grid)
        ones = np.ones((pts_mm.shape[0], 1))
        pts_mm_h = np.concatenate([pts_mm, ones], axis=1)
        ijk = (inv_affine @ pts_mm_h.T).T[:, :3]

        ijk = np.rint(ijk).astype(np.int64)

        valid = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )
        ijk = ijk[valid]

        if ijk.shape[0] == 0:
            continue

        # unique voxels per streamline
        ijk_unique = np.unique(ijk, axis=0)
        tdi[ijk_unique[:, 0], ijk_unique[:, 1], ijk_unique[:, 2]] += 1.0

    tdi /= float(n_streamlines)

    out_img = nib.Nifti1Image(tdi, affine, img.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, out_tdi)

    return out_tdi

# -------------------------
# Nipype Interface
# -------------------------
class TTGZToTDIInputSpec(BaseInterfaceInputSpec):
    tt_gz_path = File(exists=True, mandatory=True, desc="Input DSI Studio TT.GZ file")
    ref_nii = File(exists=True, mandatory=True, desc="Reference NIfTI defining target grid")
    out_tdi = File(mandatory=True, desc="Output probabilistic TDI NIfTI")

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
        )
        self._out_tdi = out_tdi
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_tdi"] = getattr(self, "_out_tdi", os.path.abspath(self.inputs.out_tdi))
        return outputs
    
if __name__ == "__main__":
    from nipype import Node

    # node = Node(TTGZToTCK(), name="ttgz_to_tck")
    # node.inputs.tt_gz_path = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_hemi-R_space-ACPC_bundle-OT_streamlines.tt.gz"
    # node.inputs.ref_nii = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/dtifit/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_model-tensor_param-fa_dwimap.nii.gz"
    # node.inputs.out_tck = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_hemi-R_space-ACPC_bundle-OT_streamlines.tck"
    # res = node.run()

    node = Node(TTGZToTDI(), name="ttgz_to_tdi")
    node.inputs.tt_gz_path = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_hemi-R_space-ACPC_bundle-OR_streamlines.tt.gz"
    node.inputs.ref_nii = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/dtifit/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_model-tensor_param-fa_dwimap.nii.gz"
    node.inputs.out_tdi = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-AFib0241/ses-baseline/visual_pathway_analysis/sub-AFib0241_ses-baseline_acq-DSIb4000_dir-AP_hemi-R_space-ACPC_bundle-OR_tdi.nii.gz"
    res = node.run()