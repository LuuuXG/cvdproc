import gzip
import numpy as np
import nibabel as nib
from scipy.io import loadmat, savemat

# ---------- TT 解析 ----------
def parse_tt(track_bytes):
    buf1 = np.array(track_bytes, dtype=np.uint8)
    buf2 = buf1.view(np.int8)
    pos, i, L = [], 0, len(buf1)
    while i < L:
        pos.append(i)
        size = np.frombuffer(buf1[i:i+4].tobytes(), dtype=np.uint32)[0]
        i += size + 13
    streamlines = []
    for p in pos:
        npts = np.frombuffer(buf1[p:p+4].tobytes(), dtype=np.uint32)[0] // 3
        x = np.frombuffer(buf1[p+4:p+8].tobytes(),  dtype=np.int32)[0]
        y = np.frombuffer(buf1[p+8:p+12].tobytes(), dtype=np.int32)[0]
        z = np.frombuffer(buf1[p+12:p+16].tobytes(),dtype=np.int32)[0]
        coords = np.zeros((npts, 3), dtype=np.float32)
        coords[0] = [x, y, z]
        q = p + 16
        for j in range(1, npts):
            x += int(buf2[q]); y += int(buf2[q+1]); z += int(buf2[q+2]); q += 3
            coords[j] = [x, y, z]
        streamlines.append(coords / 32.0)  # 回到体素坐标
    return streamlines

# ---------- TT 写回 ----------
def encode_tt(streamlines):
    out = bytearray()
    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        if s.shape[0] < 2:
            continue
        s_i = np.round(s * 32).astype('<i4', copy=False)      # int32 LE
        dif = np.diff(s_i, axis=0).astype(np.int8, copy=False) # int8
        N = s_i.shape[0]
        out += np.array([3 * N], dtype='<u4').tobytes()       # 头：3*N（不是字节数）
        out += s_i[0].tobytes()
        out += dif.tobytes()
    return np.frombuffer(bytes(out), dtype=np.uint8)

# ---------- 主函数 ----------
def refine_and_orient_tt_by_roi(tt_file, roi_file, end_roi_file, output_file,
                                distance_limit=None, direction="roi_to_end"):
    """
    For each streamline in a TT file:
      - Find the points closest to the start and end ROI centers.
      - Keep the segment between them.
      - Orient the streamline according to 'direction' parameter.
      - Optionally discard fibers too far from either ROI center.

    Parameters
    ----------
    tt_file : str
        Input .tt.gz file (TinyTrack format)
    roi_file : str
        Start ROI NIfTI mask (1 = start region of interest)
    end_roi_file : str
        End ROI NIfTI mask (1 = end region of interest)
    output_file : str
        Output .tt.gz file path
    distance_limit : float or None
        If provided, fibers farther than this (in voxel units) from either ROI center will be discarded
    direction : {"roi_to_end", "end_to_roi"}
        Desired orientation of the output fibers.
    """

    # Load TT file
    with gzip.open(tt_file, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)
    vsz = mat.get("voxel_size", np.array([1, 1, 1]))
    dim = mat.get("dimension", np.array([1, 1, 1]))
    streamlines = parse_tt(mat["track"])
    print(f"[TT] Loaded {len(streamlines)} streamlines from {tt_file}")

    # Load ROIs and compute centers
    start_roi = nib.load(roi_file).get_fdata() > 0
    end_roi = nib.load(end_roi_file).get_fdata() > 0
    start_idx = np.argwhere(start_roi)
    end_idx = np.argwhere(end_roi)
    if start_idx.size == 0 or end_idx.size == 0:
        raise ValueError("One or both ROI masks are empty.")
    start_center = start_idx.mean(axis=0).astype(np.float32)
    end_center = end_idx.mean(axis=0).astype(np.float32)
    print(f"[ROI] Start center: {start_center}, End center: {end_center}")

    refined = []
    kept = 0

    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        if s.shape[0] < 2:
            continue

        # Find closest points to start and end ROI centers
        d_start = np.linalg.norm(s - start_center, axis=1)
        d_end = np.linalg.norm(s - end_center, axis=1)
        i_start = int(np.argmin(d_start))
        i_end = int(np.argmin(d_end))

        # Apply distance threshold if needed
        if distance_limit is not None and (d_start[i_start] > distance_limit or d_end[i_end] > distance_limit):
            continue

        # Ensure valid ordering (always slice from smaller to larger index)
        if i_start < i_end:
            seg = s[i_start:i_end + 1]
        else:
            seg = s[i_end:i_start + 1][::-1].copy()

        # Orient according to user request
        if direction == "roi_to_end":
            if np.linalg.norm(seg[0] - start_center) > np.linalg.norm(seg[-1] - start_center):
                seg = seg[::-1].copy()
        elif direction == "end_to_roi":
            if np.linalg.norm(seg[0] - end_center) > np.linalg.norm(seg[-1] - end_center):
                seg = seg[::-1].copy()
        else:
            raise ValueError("direction must be 'roi_to_end' or 'end_to_roi'")

        if seg.shape[0] >= 2:
            refined.append(seg)
            kept += 1

    print(f"[Refine] Kept {kept}/{len(streamlines)} ({100 * kept / len(streamlines):.1f}%)")

    # Save new TT file
    encoded = encode_tt(refined)
    out = {"dimension": dim, "voxel_size": vsz, "track": encoded}
    with gzip.open(output_file, "wb") as f:
        savemat(f, out, format="4")
    print(f"[Save] {output_file} (streamlines={len(refined)})")


refine_and_orient_tt_by_roi(
    tt_file="/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0378/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OT.tt.gz",
    roi_file="/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0378/ses-baseline/visual_pathway_analysis/rois/optic_chiasm_in_DWI.nii.gz",
    end_roi_file="/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0378/ses-baseline/visual_pathway_analysis/rois/lh_LGN_in_DWI.nii.gz",
    output_file="/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0378/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OT_refine1.tt.gz",
    distance_limit=None,
    direction="roi_to_end"
)
