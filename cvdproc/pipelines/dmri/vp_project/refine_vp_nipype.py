import os
import nibabel as nib
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str
import gzip
from scipy.io import loadmat, savemat
from scipy.interpolate import splprep, splev

from cvdproc.bids_data.rename_bids_file import rename_bids_file

# Parse TT file (DSI Studio format)
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
        streamlines.append(coords / 32.0)
    return streamlines

# Write TT file (DSI Studio format)
def encode_tt(streamlines):
    out = bytearray()
    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        if s.shape[0] < 2:
            continue
        s_i = np.round(s * 32).astype('<i4', copy=False)      # int32 LE
        dif = np.diff(s_i, axis=0).astype(np.int8, copy=False) # int8
        N = s_i.shape[0]
        out += np.array([3 * N], dtype='<u4').tobytes()       
        out += s_i[0].tobytes()
        out += dif.tobytes()
    return np.frombuffer(bytes(out), dtype=np.uint8)

def split_or_and_merge_to_ot(
    or_tt_file,
    lgn_roi,
    chiasm_roi,
    ot_tt_file,
    output_ot_file
):
    """
    Detect and extract LGN–Chiasm segment from OR fibers that also pass through both ROIs,
    then merge it with existing OT fibers.

    Parameters
    ----------
    or_tt_file : str
        Path to the optic radiation .tt.gz file (already refined).
    lgn_roi : str
        LGN ROI NIfTI file (binary mask).
    chiasm_roi : str
        Optic chiasm ROI NIfTI file (binary mask).
    ot_tt_file : str
        Existing optic tract .tt.gz file (if missing, create a new one).
    output_ot_file : str
        Path to save the updated optic tract .tt.gz file (overwrite).
    """

    if not os.path.exists(or_tt_file) or not os.path.exists(lgn_roi) or not os.path.exists(chiasm_roi):
        print(f"[SplitMerge] Missing required inputs — skipping ({or_tt_file})")
        return

    # Load ROIs and get voxel centers
    lgn_data = nib.load(lgn_roi).get_fdata() > 0
    chiasm_data = nib.load(chiasm_roi).get_fdata() > 0
    if not lgn_data.any() or not chiasm_data.any():
        print("[SplitMerge] One or both ROI masks are empty — skipping.")
        return
    lgn_coords = np.argwhere(lgn_data)
    chiasm_coords = np.argwhere(chiasm_data)
    lgn_center = lgn_coords.mean(axis=0)
    chiasm_center = chiasm_coords.mean(axis=0)
    print(f"[SplitMerge] LGN center: {lgn_center}, Chiasm center: {chiasm_center}")

    # Helper: distance map check
    def min_distance_to_mask(points, mask_coords):
        # return array of min distances for each point
        return np.min(np.linalg.norm(points[:, None, :] - mask_coords[None, :, :], axis=2), axis=1)

    # Load OR streamlines
    with gzip.open(or_tt_file, "rb") as f:
        mat_or = loadmat(f, squeeze_me=True, struct_as_record=False)
    vsz = mat_or.get("voxel_size", np.array([1, 1, 1]))
    dim = mat_or.get("dimension", np.array([1, 1, 1]))
    or_streamlines = parse_tt(mat_or["track"])
    print(f"[SplitMerge] Loaded {len(or_streamlines)} OR fibers")

    # Load OT if exists
    if os.path.exists(ot_tt_file):
        with gzip.open(ot_tt_file, "rb") as f:
            mat_ot = loadmat(f, squeeze_me=True, struct_as_record=False)
        ot_streamlines = parse_tt(mat_ot["track"])
        print(f"[SplitMerge] Loaded {len(ot_streamlines)} OT fibers")
    else:
        ot_streamlines = []
        print("[SplitMerge] No existing OT file found — will create a new one.")

    extracted_segments = []
    for s in or_streamlines:
        s = np.asarray(s, dtype=np.float32)
        if s.ndim != 2 or s.shape[0] < 5:
            continue

        # Check whether fiber passes both LGN and chiasm
        dist_lgn = min_distance_to_mask(s, lgn_coords)
        dist_chiasm = min_distance_to_mask(s, chiasm_coords)
        lgn_hits = np.where(dist_lgn < 2.0)[0]       # within 2 voxels
        chiasm_hits = np.where(dist_chiasm < 2.0)[0] # within 2 voxels
        if len(lgn_hits) == 0 or len(chiasm_hits) == 0:
            continue

        # Find closest point indices to ROI centers
        i_lgn = int(np.argmin(dist_lgn))
        i_chiasm = int(np.argmin(dist_chiasm))

        # Require spatial order (chiasm <-> LGN)
        if abs(i_lgn - i_chiasm) < 3:
            continue

        # Extract segment and orient from chiasm -> LGN
        if i_chiasm < i_lgn:
            seg = s[i_chiasm:i_lgn + 1]
        else:
            seg = s[i_lgn:i_chiasm + 1][::-1].copy()

        if seg.shape[0] >= 5:
            extracted_segments.append(seg)

    print(f"[SplitMerge] Extracted {len(extracted_segments)} LGN–Chiasm segments from OR fibers")

    # Merge with OT and save
    merged = ot_streamlines + extracted_segments
    if len(merged) == 0:
        print("[SplitMerge] Nothing to save — merged fiber set is empty.")
        return

    encoded = encode_tt(merged)
    out = {"dimension": dim, "voxel_size": vsz, "track": encoded}
    with gzip.open(output_ot_file, "wb") as f:
        savemat(f, out, format="4")

    print(f"[SplitMerge] Saved merged OT fibers to {output_ot_file}  (total={len(merged)})")

def filter_split_by_meyersloop_keep_lgn_segment(
    tt_file,
    lgn_roi,
    meyersloop_roi,
    output_file,
    min_points=5,
    keep_original_if_ambiguous=True,
    drop_if_not_both_hit=False,
    probe_n=64
):
    """
    Voxel-hit-only filter/split.

    For each streamline:
      - Determine whether it hits LGN ROI and Meyer's loop ROI (voxel membership).
      - Only if it hits BOTH, split at the point closest to Meyer's loop ROI center.
      - Keep only the segment that hits LGN ROI if exactly one segment hits LGN.

    IMPORTANT:
      Streamline coordinates may be either:
        (A) voxel indices already, OR
        (B) world coordinates (mm).
      This function auto-detects the better mapping by probing a subset of streamlines,
      then uses the selected mode globally.

    Parameters
    ----------
    tt_file : str
        Input .tt.gz file.
    lgn_roi : str
        LGN ROI NIfTI mask (binary).
    meyersloop_roi : str
        Meyer's loop ROI NIfTI mask (binary).
    output_file : str
        Output .tt.gz file.
    min_points : int
        Minimum number of points required for a kept streamline/segment.
    keep_original_if_ambiguous : bool
        If True, keep original streamline when both segments hit LGN (or neither hits).
        If False, drop those ambiguous streamlines.
    drop_if_not_both_hit : bool
        If True, drop streamlines that do not hit BOTH LGN and Meyer's loop.
        If False, keep them unchanged.
    probe_n : int
        Number of streamlines to probe for auto-detection of coordinate mode.
    """

    import os
    import gzip
    import numpy as np
    import nibabel as nib
    from scipy.io import loadmat, savemat

    if not (os.path.exists(tt_file) and os.path.exists(lgn_roi) and os.path.exists(meyersloop_roi)):
        raise FileNotFoundError("One or more input files do not exist.")

    # -------------------------
    # Load ROI masks + affine
    # -------------------------
    lgn_img = nib.load(lgn_roi)
    ml_img = nib.load(meyersloop_roi)

    lgn_mask = lgn_img.get_fdata() > 0
    ml_mask = ml_img.get_fdata() > 0

    if not (lgn_mask.any() and ml_mask.any()):
        raise ValueError("LGN ROI or Meyer's loop ROI mask is empty.")

    ml_coords = np.argwhere(ml_mask)
    ml_center = ml_coords.mean(axis=0).astype(np.float32)

    inv_affine = np.linalg.inv(ml_img.affine)

    # -------------------------
    # Helpers: convert points -> voxel indices (two candidate modes)
    # -------------------------
    def points_to_vox_direct(points):
        # Treat points as voxel coordinates directly
        return np.rint(points).astype(np.int32)

    def points_to_vox_world(points):
        # Treat points as world (mm), convert via inverse affine to voxel
        n = points.shape[0]
        ones = np.ones((n, 1), dtype=np.float32)
        xyz1 = np.concatenate([points.astype(np.float32), ones], axis=1)  # (n,4)
        ijk1 = (inv_affine @ xyz1.T).T  # (n,4)
        return np.rint(ijk1[:, :3]).astype(np.int32)

    def in_bounds(ijk, shape):
        return (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )

    def hit_mask_from_ijk(ijk, mask):
        ok = in_bounds(ijk, mask.shape)
        if not np.any(ok):
            return False
        ijk_ok = ijk[ok]
        return bool(np.any(mask[ijk_ok[:, 0], ijk_ok[:, 1], ijk_ok[:, 2]]))

    def choose_mode(streamlines, n_probe):
        # Choose global mode by maximizing "both-hit" count on a probe subset
        n_probe = min(n_probe, len(streamlines))
        if n_probe == 0:
            return "direct"

        both_direct = 0
        both_world = 0
        valid_direct = 0
        valid_world = 0

        for s in streamlines[:n_probe]:
            s = np.asarray(s, dtype=np.float32)
            if s.ndim != 2 or s.shape[0] < 2:
                continue

            ijk_d = points_to_vox_direct(s)
            ijk_w = points_to_vox_world(s)

            # valid ratio proxy: any point falls in bounds of ml_mask
            if np.any(in_bounds(ijk_d, ml_mask.shape)):
                valid_direct += 1
            if np.any(in_bounds(ijk_w, ml_mask.shape)):
                valid_world += 1

            hit_lgn_d = hit_mask_from_ijk(ijk_d, lgn_mask)
            hit_ml_d = hit_mask_from_ijk(ijk_d, ml_mask)
            if hit_lgn_d and hit_ml_d:
                both_direct += 1

            hit_lgn_w = hit_mask_from_ijk(ijk_w, lgn_mask)
            hit_ml_w = hit_mask_from_ijk(ijk_w, ml_mask)
            if hit_lgn_w and hit_ml_w:
                both_world += 1

        # Primary criterion: more "both-hit"
        # Secondary: more "valid"
        if both_world > both_direct:
            return "world"
        if both_direct > both_world:
            return "direct"
        return "world" if valid_world > valid_direct else "direct"

    # -------------------------
    # Load TT streamlines
    # -------------------------
    with gzip.open(tt_file, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)

    vsz = mat.get("voxel_size", np.array([1, 1, 1]))
    dim = mat.get("dimension", np.array([1, 1, 1]))
    streamlines = parse_tt(mat["track"])

    print(f"[FilterSplit] Loaded {len(streamlines)} streamlines")
    print(f"[FilterSplit] Meyer's loop center (voxel index): {ml_center}")

    mode = choose_mode(streamlines, probe_n)
    print(f"[FilterSplit] Coordinate mode selected: {mode}")

    def points_to_vox(points):
        return points_to_vox_world(points) if mode == "world" else points_to_vox_direct(points)

    # -------------------------
    # Main loop + debug counts
    # -------------------------
    n_hit_lgn = 0
    n_hit_ml = 0
    n_hit_both = 0
    n_modified = 0
    n_ambiguous_dropped = 0
    n_dropped_not_both = 0

    kept = []

    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        if s.ndim != 2 or s.shape[0] < min_points:
            continue

        ijk = points_to_vox(s)

        hit_lgn = hit_mask_from_ijk(ijk, lgn_mask)
        hit_ml = hit_mask_from_ijk(ijk, ml_mask)

        if hit_lgn:
            n_hit_lgn += 1
        if hit_ml:
            n_hit_ml += 1
        if hit_lgn and hit_ml:
            n_hit_both += 1

        # Only process split/crop when BOTH hit
        if not (hit_lgn and hit_ml):
            if drop_if_not_both_hit:
                n_dropped_not_both += 1
                continue
            kept.append(s)
            continue

        # Split at point closest to ML center (in the same coordinate space as s)
        # We compute distance using voxel coordinates for stability.
        # Map ML center into the same coordinate space as ijk for comparison: ijk already voxel.
        d = np.linalg.norm(ijk.astype(np.float32) - ml_center[None, :], axis=1)
        i_split = int(np.argmin(d))

        seg1 = s[: i_split + 1]
        seg2 = s[i_split: ]

        if seg1.shape[0] < min_points or seg2.shape[0] < min_points:
            kept.append(s)
            continue

        seg1_ijk = points_to_vox(seg1)
        seg2_ijk = points_to_vox(seg2)

        seg1_lgn = hit_mask_from_ijk(seg1_ijk, lgn_mask)
        seg2_lgn = hit_mask_from_ijk(seg2_ijk, lgn_mask)

        if seg1_lgn and (not seg2_lgn):
            kept.append(seg1)
            n_modified += 1
        elif seg2_lgn and (not seg1_lgn):
            kept.append(seg2)
            n_modified += 1
        else:
            if keep_original_if_ambiguous:
                kept.append(s)
            else:
                n_ambiguous_dropped += 1

    print(f"[FilterSplit] Hit LGN: {n_hit_lgn}")
    print(f"[FilterSplit] Hit MeyersLoop: {n_hit_ml}")
    print(f"[FilterSplit] Hit BOTH: {n_hit_both}")
    print(f"[FilterSplit] Modified (cropped): {n_modified}")
    if drop_if_not_both_hit:
        print(f"[FilterSplit] Dropped (not both-hit): {n_dropped_not_both}")
    if not keep_original_if_ambiguous:
        print(f"[FilterSplit] Dropped (ambiguous after split): {n_ambiguous_dropped}")
    print(f"[FilterSplit] Output streamlines: {len(kept)}")

    # -------------------------
    # Save TT
    # -------------------------
    encoded = encode_tt(kept)
    out = {"dimension": dim, "voxel_size": vsz, "track": encoded}
    with gzip.open(output_file, "wb") as f:
        savemat(f, out, format="4")

    print(f"[FilterSplit] Saved: {output_file}")


def refine_and_orient_tt_by_roi(
    tt_file,
    roi_file,
    end_roi_file,
    output_file,
    distance_limit=None,
    direction="roi_to_end",
    keep_full_end=False,
    must_in_roi=False,
):
    """
    For each streamline in a TT file:
      - Find the points closest to the start and end ROI centers.
      - Keep the segment between them (optionally keep the full end side).
      - Orient the streamline according to 'direction' parameter.
      - Optionally discard fibers too far from either ROI center.
      - Optionally require the streamline TERMINAL endpoint to lie inside end ROI (must_in_roi).

    must_in_roi behavior (UPDATED):
      - direction="roi_to_end": seg[-1] must be in end_roi_file
      - direction="end_to_roi": seg[0] must be in end_roi_file
    """

    def _in_mask(pt_xyz, mask_bool):
        ijk = np.rint(pt_xyz).astype(int)
        if ijk.shape[0] != 3:
            return False
        x, y, z = ijk
        if x < 0 or y < 0 or z < 0:
            return False
        if (
            x >= mask_bool.shape[0]
            or y >= mask_bool.shape[1]
            or z >= mask_bool.shape[2]
        ):
            return False
        return bool(mask_bool[x, y, z])

    # --------------------
    # Load TT
    # --------------------
    with gzip.open(tt_file, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)

    vsz = mat.get("voxel_size", np.array([1, 1, 1]))
    dim = mat.get("dimension", np.array([1, 1, 1]))
    streamlines = parse_tt(mat["track"])
    print(f"[TT] Loaded {len(streamlines)} streamlines from {tt_file}")

    # --------------------
    # Load ROIs
    # --------------------
    start_roi_mask = nib.load(roi_file).get_fdata() > 0
    end_roi_mask = nib.load(end_roi_file).get_fdata() > 0

    start_idx = np.argwhere(start_roi_mask)
    end_idx = np.argwhere(end_roi_mask)
    if start_idx.size == 0 or end_idx.size == 0:
        raise ValueError("One or both ROI masks are empty.")

    start_center = start_idx.mean(axis=0).astype(np.float32)
    end_center = end_idx.mean(axis=0).astype(np.float32)
    print(f"[ROI] Start center: {start_center}, End center: {end_center}")

    refined = []
    kept = 0
    dropped_must_in_roi = 0

    # --------------------
    # Main loop
    # --------------------
    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        if s.shape[0] < 2:
            continue

        # Closest points to ROI centers
        d_start = np.linalg.norm(s - start_center, axis=1)
        d_end = np.linalg.norm(s - end_center, axis=1)
        i_start = int(np.argmin(d_start))
        i_end = int(np.argmin(d_end))

        # Distance threshold
        if distance_limit is not None and (
            d_start[i_start] > distance_limit
            or d_end[i_end] > distance_limit
        ):
            continue

        # Segment extraction
        if keep_full_end:
            if i_start < i_end:
                seg = s[i_start:]
            else:
                seg = s[: i_start + 1][::-1].copy()
        else:
            if i_start < i_end:
                seg = s[i_start : i_end + 1]
            else:
                seg = s[i_end : i_start + 1][::-1].copy()

        # Orientation
        if direction == "roi_to_end":
            if np.linalg.norm(seg[0] - start_center) > np.linalg.norm(seg[-1] - start_center):
                seg = seg[::-1].copy()
        elif direction == "end_to_roi":
            if np.linalg.norm(seg[0] - end_center) > np.linalg.norm(seg[-1] - end_center):
                seg = seg[::-1].copy()
        else:
            raise ValueError("direction must be 'roi_to_end' or 'end_to_roi'")

        if seg.shape[0] < 2:
            continue

        # --------------------
        # Must-in-end-ROI constraint (UPDATED)
        # --------------------
        if must_in_roi:
            if direction == "roi_to_end":
                ok = _in_mask(seg[-1], end_roi_mask)
            else:  # end_to_roi
                ok = _in_mask(seg[0], end_roi_mask)

            if not ok:
                dropped_must_in_roi += 1
                continue

        refined.append(seg)
        kept += 1

    # --------------------
    # Report & save
    # --------------------
    print(f"[Refine] Kept {kept}/{len(streamlines)} ({100 * kept / len(streamlines):.1f}%)")
    if must_in_roi:
        print(f"[Refine] Dropped by must_in_roi (end ROI only): {dropped_must_in_roi}")

    encoded = encode_tt(refined)
    out = {"dimension": dim, "voxel_size": vsz, "track": encoded}
    with gzip.open(output_file, "wb") as f:
        savemat(f, out, format="4")

    print(f"[Save] {output_file} (streamlines={len(refined)})")



def filter_or_tt_by_direction_window(
    input_tt,
    output_tt,

    # X direction (signed mean diff)
    x_start_idx=0,
    x_end_idx=5,
    x_threshold=None,     # None = disable
    x_sign=None,          # "positive" | "negative"

    # Y direction (consecutive decrease)
    y_start_idx=0,
    y_end_idx=5,
    y_drop_threshold=None,   # None = disable
    y_drop_count=1,

    # Z direction (consecutive decrease)
    z_start_idx=0,
    z_end_idx=5,
    z_drop_threshold=None,   # None = disable
    z_drop_count=1,

    v1_roi=None,
):
    """
    Filter OR streamlines by optional X / Y / Z directional rules and optional V1 endpoint constraint.
    """

    # --------------------
    # Load TT
    # --------------------
    with gzip.open(input_tt, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)

    voxel_size = mat.get("voxel_size", np.array([1, 1, 1]))
    dimension = mat.get("dimension", np.array([1, 1, 1]))
    streamlines = parse_tt(mat["track"])

    print(f"[TT] Loaded {len(streamlines)} streamlines")

    # --------------------
    # Load V1 ROI (optional)
    # --------------------
    v1_mask = None
    if v1_roi is not None:
        if not os.path.exists(v1_roi):
            raise FileNotFoundError(v1_roi)
        v1_mask = nib.load(v1_roi).get_fdata() > 0
        if not v1_mask.any():
            raise ValueError("V1 ROI mask is empty")

    def endpoint_in_mask(pt, mask):
        ijk = np.rint(pt).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= mask.shape):
            return False
        return bool(mask[tuple(ijk)])

    def has_consecutive_drop(vals, threshold, n_drop):
        cnt = 0
        for v in vals:
            if v < -threshold:
                cnt += 1
                if cnt >= n_drop:
                    return True
            else:
                cnt = 0
        return False

    kept = []
    removed_x = removed_y = removed_z = removed_v1 = removed_short = 0

    # --------------------
    # Main loop
    # --------------------
    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        n_pts = s.shape[0]

        max_end = max(
            x_end_idx or 0,
            y_end_idx or 0,
            z_end_idx or 0,
        )
        if n_pts <= max_end:
            removed_short += 1
            continue

        # -------- X rule --------
        if x_threshold is not None:
            dx = np.diff(s[x_start_idx:x_end_idx, 0])
            mean_dx = float(np.mean(dx))

            if x_sign == "positive" and mean_dx <= x_threshold:
                removed_x += 1
                continue
            if x_sign == "negative" and mean_dx >= -x_threshold:
                removed_x += 1
                continue

        # -------- Y rule --------
        if y_drop_threshold is not None:
            dy = np.diff(s[y_start_idx:y_end_idx, 1])
            if has_consecutive_drop(dy, y_drop_threshold, y_drop_count):
                removed_y += 1
                continue

        # -------- Z rule --------
        if z_drop_threshold is not None:
            dz = np.diff(s[z_start_idx:z_end_idx, 2])
            if has_consecutive_drop(dz, z_drop_threshold, z_drop_count):
                removed_z += 1
                continue

        # -------- V1 endpoint rule --------
        if v1_mask is not None:
            if not endpoint_in_mask(s[-1], v1_mask):
                removed_v1 += 1
                continue

        kept.append(s)

    # --------------------
    # Save
    # --------------------
    encoded = encode_tt(kept)
    out = {"dimension": dimension, "voxel_size": voxel_size, "track": encoded}
    with gzip.open(output_tt, "wb") as f:
        savemat(f, out, format="4")

    print(f"[Result] {len(streamlines)} → {len(kept)} kept")
    print(
        f"[Removed] short={removed_short}, "
        f"x={removed_x}, y={removed_y}, z={removed_z}, v1={removed_v1}"
    )

    return len(streamlines), len(kept)


def filter_ot(
    tt_file,
    output_file,
    y_drop_threshold=10,
    z_drop_threshold=10,
    eps=0.0,
    chiasm_roi=None,
):
    """
    Remove streamlines where:
      (1) y coordinate decreases consecutively for >= y_drop_threshold points, OR
      (2) z coordinate decreases consecutively for >= z_drop_threshold points, OR
      (3) both endpoints are not on the same side of the chiasm ROI center along x-axis.

    Parameters
    ----------
    tt_file : str
        Input .tt.gz file (TinyTrack format)
    output_file : str
        Output filtered .tt.gz file
    y_drop_threshold : int
        Number of consecutive points with strictly decreasing y before exclusion
    z_drop_threshold : int
        Number of consecutive points with strictly decreasing z before exclusion
    eps : float
        Small tolerance; if diff < -eps, considered decreasing
    chiasm_roi : str or None
        NIfTI ROI for optic chiasm. If provided, only keep streamlines whose both endpoints
        lie on the same side (left or right) of the chiasm center x-coordinate.
    """

    def has_consecutive_drop(values, threshold, eps=0.0):
        """
        Generic consecutive drop detector.
        values: 1D array
        """
        dv = np.diff(values)
        cnt = 0
        for v in dv:
            if v < -eps:
                cnt += 1
                if cnt >= (threshold - 1):
                    return True
            else:
                cnt = 0
        return False

    # Load TT
    with gzip.open(tt_file, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)

    vsz = mat.get("voxel_size", np.array([1, 1, 1]))
    dim = mat.get("dimension", np.array([1, 1, 1]))
    streamlines = parse_tt(mat["track"])

    print(f"[TT] Loaded {len(streamlines)} streamlines")

    # Load chiasm ROI if provided
    chiasm_x_center = None
    if chiasm_roi is not None and os.path.exists(chiasm_roi):
        roi_data = nib.load(chiasm_roi).get_fdata() > 0
        coords = np.argwhere(roi_data)
        if coords.size > 0:
            chiasm_x_center = coords[:, 0].mean()
            print(f"[ROI] Chiasm center x = {chiasm_x_center:.2f}")
        else:
            print("[ROI] Warning: chiasm ROI mask is empty.")
    else:
        print("[ROI] No chiasm ROI provided — skipping side filter.")

    kept_streamlines = []
    removed_y = 0
    removed_z = 0
    removed_side = 0

    for s in streamlines:
        if not isinstance(s, np.ndarray) or s.ndim != 2 or s.shape[1] != 3:
            continue

        # (1) consecutive Y drop
        if has_consecutive_drop(s[:, 1], y_drop_threshold, eps):
            removed_y += 1
            continue

        # (2) consecutive Z drop
        if has_consecutive_drop(s[:, 2], z_drop_threshold, eps):
            removed_z += 1
            continue

        # (3) check both endpoints relative to chiasm x center
        if chiasm_x_center is not None:
            x_start = s[0, 0]
            x_end = s[-1, 0]
            side_start = np.sign(x_start - chiasm_x_center)
            side_end = np.sign(x_end - chiasm_x_center)

            # must be on the same side
            if side_start == 0 or side_end == 0:
                pass
            elif side_start != side_end:
                removed_side += 1
                continue

        kept_streamlines.append(s)

    removed_total = removed_y + removed_z + removed_side
    print(
        f"[Filter] Removed {removed_total}/{len(streamlines)} "
        f"({100 * removed_total / len(streamlines):.1f}%) "
        f"— ({removed_y} Y-drop, {removed_z} Z-drop, {removed_side} chiasm-side)"
    )

    encoded = encode_tt(kept_streamlines)
    out = {"dimension": dim, "voxel_size": vsz, "track": encoded}
    with gzip.open(output_file, "wb") as f:
        savemat(f, out, format="4")

    print(f"[Save] {output_file} (remaining={len(kept_streamlines)})")

def remove_points_in_roi_from_tt(
    input_tt,
    roi_file,
    output_tt,
    min_points_keep=2,
    drop_entire_streamline_if_hit_roi=False,
):
    """
    Remove points or entire streamlines that fall inside a ROI mask.

    Parameters
    ----------
    input_tt : str
        Input TinyTrack .tt.gz file.
    roi_file : str
        Binary ROI NIfTI mask.
    output_tt : str
        Output TinyTrack .tt.gz file.
    min_points_keep : int
        Minimum number of points required to keep a streamline after removal.
        Only used when drop_entire_streamline_if_hit_roi=False.
    drop_entire_streamline_if_hit_roi : bool
        If True, any streamline that has at least one point inside the ROI
        will be entirely discarded.
        If False, only points inside ROI are removed (default behavior).
    """

    def _point_in_mask(pt_xyz, mask_bool):
        ijk = np.rint(pt_xyz).astype(np.int32)
        if ijk.shape[0] != 3:
            return False
        x, y, z = int(ijk[0]), int(ijk[1]), int(ijk[2])
        if x < 0 or y < 0 or z < 0:
            return False
        if x >= mask_bool.shape[0] or y >= mask_bool.shape[1] or z >= mask_bool.shape[2]:
            return False
        return bool(mask_bool[x, y, z])

    if not os.path.exists(input_tt):
        raise FileNotFoundError(f"Input TT not found: {input_tt}")
    if not os.path.exists(roi_file):
        raise FileNotFoundError(f"ROI mask not found: {roi_file}")

    # Load TT
    with gzip.open(input_tt, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)

    vsz = mat.get("voxel_size", np.array([1, 1, 1]))
    dim = mat.get("dimension", np.array([1, 1, 1]))
    streamlines = parse_tt(mat["track"])
    print(f"[TT] Loaded {len(streamlines)} streamlines from {input_tt}")

    # Load ROI mask
    roi_mask = nib.load(roi_file).get_fdata() > 0
    if not roi_mask.any():
        print(f"[ROI] ROI mask is empty: {roi_file} (no streamlines will be affected)")

    kept_streamlines = []
    removed_points_total = 0
    dropped_streamlines = 0
    dropped_by_roi_hit = 0

    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        if s.ndim != 2 or s.shape[0] < 2 or s.shape[1] != 3:
            continue

        hit_roi = False
        keep_mask = np.ones((s.shape[0],), dtype=bool)

        for i in range(s.shape[0]):
            if _point_in_mask(s[i], roi_mask):
                hit_roi = True
                keep_mask[i] = False

        # --- Mode 1: drop entire streamline if any point hits ROI ---
        if drop_entire_streamline_if_hit_roi:
            if hit_roi:
                dropped_streamlines += 1
                dropped_by_roi_hit += 1
                continue
            kept_streamlines.append(s)
            continue

        # --- Mode 2: only remove ROI points ---
        removed_points = int(np.sum(~keep_mask))
        removed_points_total += removed_points

        s_new = s[keep_mask]
        if s_new.shape[0] < int(min_points_keep):
            dropped_streamlines += 1
            continue

        kept_streamlines.append(s_new)

    print(f"[ROI] Drop-entire-streamline mode: {drop_entire_streamline_if_hit_roi}")
    if drop_entire_streamline_if_hit_roi:
        print(f"[TT] Dropped streamlines by ROI hit: {dropped_by_roi_hit}")
    else:
        print(f"[ROI] Removed points total: {removed_points_total}")
        print(f"[TT] Dropped streamlines (too short after removal): {dropped_streamlines}")

    print(f"[TT] Kept streamlines: {len(kept_streamlines)}/{len(streamlines)}")

    # Save output TT
    os.makedirs(os.path.dirname(os.path.abspath(output_tt)), exist_ok=True)
    encoded = encode_tt(kept_streamlines)
    out = {"dimension": dim, "voxel_size": vsz, "track": encoded}
    with gzip.open(output_tt, "wb") as f:
        savemat(f, out, format="4")

    print(f"[Save] {output_tt} (streamlines={len(kept_streamlines)})")



def trk_interp(streamlines, n_points_new=100, spacing=None, tie_at_center=False):
    """Interpolate all streamlines to have exactly n_points_new points."""
    tracks_interp = [None] * len(streamlines)  # Placeholder for inter
    splines = []

    # 1. Fit splines to each streamline
    for idx, coords in enumerate(streamlines):
        coords = np.asarray(coords)
        if coords.shape[0] < 4:
            # If too short, repeat the first point to fill
            tracks_interp[idx] = np.repeat(coords[:1], n_points_new, axis=0)
            continue

        diffs = np.diff(coords, axis=0)
        segs = np.linalg.norm(diffs, axis=1)
        dist = np.insert(np.cumsum(segs), 0, 0.0)

        if len(np.unique(dist)) < 4:
            tracks_interp[idx] = np.repeat(coords[:1], n_points_new, axis=0)
            continue

        tck, _ = splprep(coords.T, u=dist, s=0, k=min(3, len(coords)-1))
        splines.append((idx, tck, dist[-1]))

    # 2. Fixed-point resampling
    for idx, tck, total_len in splines:
        u_new = np.linspace(0, total_len, n_points_new)
        interp_coords = np.array(splev(u_new, tck)).T
        tracks_interp[idx] = interp_coords

    # 3. tie-at-center mode (geometric center alignment)
    if tie_at_center and spacing is None and len(splines) > 0:
        n_points_new_odd = int(np.floor(n_points_new / 2) * 2 + 1)
        mean_track = np.mean(np.stack([
            np.array(splev(np.linspace(0, s[2], n_points_new_odd), s[1])).T
            for s in splines
        ]), axis=0)
        middle = mean_track[len(mean_track)//2]

        for idx, tck, total_len in splines:
            coords = np.array(splev(np.linspace(0, total_len, n_points_new_odd), tck)).T
            dists = np.linalg.norm(coords - middle, axis=1)
            ind = np.argmin(dists)

            first_half = np.array(splev(np.linspace(0, total_len * (ind / n_points_new_odd),
                                                    n_points_new_odd // 2 + 1), tck)).T
            second_half = np.array(splev(np.linspace(total_len * (ind / n_points_new_odd),
                                                     total_len,
                                                     n_points_new_odd // 2 + 1), tck)).T
            merged = np.vstack([first_half, second_half[1:]])

            # Force padding or cropping
            if merged.shape[0] < n_points_new:
                pad = np.repeat(merged[-1][None, :], n_points_new - merged.shape[0], axis=0)
                merged = np.vstack([merged, pad])
            elif merged.shape[0] > n_points_new:
                merged = merged[:n_points_new]

            tracks_interp[idx] = merged

    # 4. Final check and force padding
    for i, t in enumerate(tracks_interp):
        if t is None or t.shape[0] < n_points_new:
            if t is None or t.shape[0] == 0:
                t = np.zeros((1, 3))
            pad = np.repeat(t[-1][None, :], n_points_new - t.shape[0], axis=0)
            tracks_interp[i] = np.vstack([t, pad])
        elif t.shape[0] > n_points_new:
            tracks_interp[i] = t[:n_points_new]

    return tracks_interp

def interpolate_tt(input_tt, output_tt, n_points_new=100, spacing=None, tie_at_center=False):
    """Load .tt.gz, resample streamlines, and save a new .tt.gz."""
    with gzip.open(input_tt, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)

    voxel_size = mat.get("voxel_size", np.array([1, 1, 1]))
    dimension = mat.get("dimension", np.array([1, 1, 1]))
    track_bytes = mat["track"]

    print(f"Loaded TT: {input_tt}")
    streamlines = parse_tt(track_bytes)
    print(f"Found {len(streamlines)} streamlines")

    interp_streams = trk_interp(streamlines,
                                n_points_new=n_points_new,
                                spacing=spacing,
                                tie_at_center=tie_at_center)

    encoded = encode_tt(interp_streams)
    data_out = {"dimension": dimension,
                "voxel_size": voxel_size,
                "track": encoded}

    with gzip.open(output_tt, "wb") as f:
        savemat(f, data_out, format="4")

    print(f"Saved interpolated TT file: {output_tt}")
    print(f"Each streamline now has {n_points_new} points (tie_at_center={tie_at_center})")

class RefineVPInputSpec(BaseInterfaceInputSpec):
    lh_ot = Str(desc="Left hemisphere optic tract .tt.gz file", mandatory=True)
    rh_ot = Str(desc="Right hemisphere optic tract .tt.gz file", mandatory=True)
    lh_or = Str(desc="Left hemisphere optic radiation .tt.gz file", mandatory=True)
    rh_or = Str(desc="Right hemisphere optic radiation .tt.gz file", mandatory=True)
    lh_ml = Str(desc="Left hemisphere Meyers Loop .tt.gz file", mandatory=False)
    rh_ml = Str(desc="Right hemisphere Meyers Loop .tt.gz file", mandatory=False)
    cho_roi = File(exists=True, desc="Chiasm ROI NIfTI file", mandatory=True)
    #cho_dia1_roi = File(exists=True, desc="Chiasm dilated 1 voxel ROI NIfTI file", mandatory=False)
    lh_lgn_roi = File(exists=True, desc="Left LGN ROI NIfTI file", mandatory=True)
    lh_lgn_dia_x_roi = File(exists=True, desc="Left LGN dilated 3 voxel (in x axis) ROI NIfTI file", mandatory=False)
    lh_lgn_extendpart_roi = File(exists=True, desc="Left LGN ROI extended part NIfTI file", mandatory=False)
    rh_lgn_roi = File(exists=True, desc="Right LGN ROI NIfTI file", mandatory=True)
    rh_lgn_dia_x_roi = File(exists=True, desc="Right LGN dilated 3 voxel (in x axis) ROI NIfTI file", mandatory=False)
    rh_lgn_extendpart_roi = File(exists=True, desc="Right LGN ROI extended part NIfTI file", mandatory=False)
    lh_v1_roi = File(exists=True, desc="Left V1 ROI NIfTI file", mandatory=False)
    rh_v1_roi = File(exists=True, desc="Right V1 ROI NIfTI file", mandatory=False)
    lh_meyersloop_roi = File(exists=True, desc="Left Meyers Loop ROI NIfTI file", mandatory=False)
    rh_meyersloop_roi = File(exists=True, desc="Right Meyers Loop ROI NIfTI file", mandatory=False)

    output_dir = Str(desc="Output directory", mandatory=True)
    output_lh_ot = Str("refined_lh_optic_tract.tt.gz", desc="Output filename for refined left optic tract")
    output_rh_ot = Str("refined_rh_optic_tract.tt.gz", desc="Output filename for refined right optic tract")
    output_lh_or = Str("refined_lh_optic_radiation.tt.gz", desc="Output filename for refined left optic radiation")
    output_rh_or = Str("refined_rh_optic_radiation.tt.gz", desc="Output filename for refined right optic radiation")
    output_lh_ml = Str("refined_lh_meyers_loop.tt.gz", desc="Output filename for refined left Meyers Loop")
    output_rh_ml = Str("refined_rh_meyers_loop.tt.gz", desc="Output filename for refined right Meyers Loop")

class RefineVPOutputSpec(TraitedSpec):
    refined_lh_ot = Str(desc="Refined left hemisphere optic tract .tt.gz file")
    refined_rh_ot = Str(desc="Refined right hemisphere optic tract .tt.gz file")
    refined_lh_or = Str(desc="Refined left hemisphere optic radiation .tt.gz file")
    refined_rh_or = Str(desc="Refined right hemisphere optic radiation .tt.gz file")
    refined_lh_ml = Str(desc="Refined left hemisphere Meyers Loop .tt.gz file")
    refined_rh_ml = Str(desc="Refined right hemisphere Meyers Loop .tt.gz file")

class RefineVP(BaseInterface):
    input_spec = RefineVPInputSpec
    output_spec = RefineVPOutputSpec

    def _run_interface(self, runtime):
        os.makedirs(self.inputs.output_dir, exist_ok=True)

        # Refine and orient optic radiation
        if os.path.exists(self.inputs.lh_or):
            output_lh_or_path = os.path.join(self.inputs.output_dir, self.inputs.output_lh_or)

            filter_split_by_meyersloop_keep_lgn_segment(
                self.inputs.lh_or,
                self.inputs.lh_lgn_dia_x_roi,
                self.inputs.lh_meyersloop_roi,
                output_lh_or_path
            )

            refine_and_orient_tt_by_roi(
                output_lh_or_path,
                self.inputs.lh_lgn_extendpart_roi,
                #self.inputs.lh_lgn_dia_x_roi,
                self.inputs.lh_v1_roi,
                output_lh_or_path,
                distance_limit=None,
                direction="roi_to_end",
                keep_full_end=True,
                must_in_roi=True
            )

            remove_points_in_roi_from_tt(
                input_tt=output_lh_or_path,
                output_tt=output_lh_or_path,
                roi_file=self.inputs.lh_lgn_dia_x_roi
            )

            remove_points_in_roi_from_tt(
                input_tt=output_lh_or_path,
                output_tt=output_lh_or_path,
                roi_file=self.inputs.lh_lgn_roi
            )

            remove_points_in_roi_from_tt(
                input_tt=output_lh_or_path,
                output_tt=output_lh_or_path,
                roi_file=self.inputs.lh_meyersloop_roi,
                drop_entire_streamline_if_hit_roi=True
            )

            interpolate_tt(output_lh_or_path, output_lh_or_path, n_points_new=100, tie_at_center=True)

            filter_or_tt_by_direction_window(
                input_tt=output_lh_or_path,
                output_tt=output_lh_or_path,
                x_start_idx=0,
                x_end_idx=5,
                x_threshold=1e-6,     
                x_sign='positive',
                y_start_idx=5,
                y_end_idx=50,
                y_drop_threshold=1e-6,
                y_drop_count=3,
                v1_roi=self.inputs.lh_v1_roi
            )

            self._refined_lh_or = output_lh_or_path
        else:
            print(f"Left optic radiation file {self.inputs.lh_or} does not exist. Skipping refinement.")
            self._refined_lh_or = ''

        if os.path.exists(self.inputs.rh_or):
            output_rh_or_path = os.path.join(self.inputs.output_dir, self.inputs.output_rh_or)

            filter_split_by_meyersloop_keep_lgn_segment(
                self.inputs.rh_or,
                self.inputs.rh_lgn_dia_x_roi,
                self.inputs.rh_meyersloop_roi,
                output_rh_or_path
            )

            refine_and_orient_tt_by_roi(
                output_rh_or_path,
                self.inputs.rh_lgn_extendpart_roi,
                #self.inputs.rh_lgn_dia_x_roi,
                self.inputs.rh_v1_roi,
                output_rh_or_path,
                distance_limit=None,
                direction="roi_to_end",
                keep_full_end=True,
                must_in_roi=True
            )

            remove_points_in_roi_from_tt(
                input_tt=output_rh_or_path,
                output_tt=output_rh_or_path,
                roi_file=self.inputs.rh_lgn_dia_x_roi
            )

            remove_points_in_roi_from_tt(
                input_tt=output_rh_or_path,
                output_tt=output_rh_or_path,
                roi_file=self.inputs.rh_lgn_roi
            )

            remove_points_in_roi_from_tt(
                input_tt=output_rh_or_path,
                output_tt=output_rh_or_path,
                roi_file=self.inputs.rh_meyersloop_roi,
                drop_entire_streamline_if_hit_roi=True
            )

            interpolate_tt(output_rh_or_path, output_rh_or_path, n_points_new=100, tie_at_center=True)

            filter_or_tt_by_direction_window(
                input_tt=output_rh_or_path,
                output_tt=output_rh_or_path,
                x_start_idx=0,
                x_end_idx=5,
                x_threshold=1e-6,     
                x_sign='negative',
                y_start_idx=5,
                y_end_idx=50,
                y_drop_threshold=1e-6,
                y_drop_count=3,
                v1_roi=self.inputs.rh_v1_roi
            )
            
            self._refined_rh_or = output_rh_or_path
        else:
            print(f"Right optic radiation file {self.inputs.rh_or} does not exist. Skipping refinement.")
            self._refined_rh_or = ''
        
        if os.path.exists(self.inputs.lh_ot):
            # Refine and orient optic tract (2 step)
            output_lh_ot_path = os.path.join(self.inputs.output_dir, self.inputs.output_lh_ot)
            refine_and_orient_tt_by_roi(
                self.inputs.lh_ot,
                self.inputs.cho_roi,
                self.inputs.lh_lgn_roi,
                output_lh_ot_path,
                distance_limit=None,
                direction="roi_to_end",
                keep_full_end=False
            )

            remove_points_in_roi_from_tt(
                input_tt=output_lh_ot_path,
                output_tt=output_lh_ot_path,
                roi_file=self.inputs.lh_lgn_roi
            )

            interpolate_tt(output_lh_ot_path, output_lh_ot_path, n_points_new=100, tie_at_center=True)

            filter_ot(
                tt_file=output_lh_ot_path,
                output_file=output_lh_ot_path,
                y_drop_threshold=3,
                z_drop_threshold=5,
                chiasm_roi=self.inputs.cho_roi,
                eps=0.0
            )

            self._refined_lh_ot = output_lh_ot_path
        else:
            print(f"Left optic tract file {self.inputs.lh_ot} does not exist. Skipping refinement.")
            self._refined_lh_ot = ''
        
        if os.path.exists(self.inputs.rh_ot):
            output_rh_ot_path = os.path.join(self.inputs.output_dir, self.inputs.output_rh_ot)
            refine_and_orient_tt_by_roi(
                self.inputs.rh_ot,
                self.inputs.cho_roi,
                self.inputs.rh_lgn_roi,
                output_rh_ot_path,
                distance_limit=None,
                direction="roi_to_end",
                keep_full_end=False
            )

            remove_points_in_roi_from_tt(
                input_tt=output_rh_ot_path,
                output_tt=output_rh_ot_path,
                roi_file=self.inputs.rh_lgn_roi
            )

            interpolate_tt(output_rh_ot_path, output_rh_ot_path, n_points_new=100, tie_at_center=True)

            filter_ot(
                tt_file=output_rh_ot_path,
                output_file=output_rh_ot_path,
                y_drop_threshold=3,
                z_drop_threshold=5,
                chiasm_roi=self.inputs.cho_roi,
                eps=0.0
            )

            self._refined_rh_ot = output_rh_ot_path

            # Refine Meyers Loop part if provided
            output_lh_ml_path = os.path.join(self.inputs.output_dir, self.inputs.output_lh_ml)
            if self.inputs.lh_ml and os.path.exists(self.inputs.lh_ml):
                refine_and_orient_tt_by_roi(
                    self.inputs.lh_ml,
                    self.inputs.lh_meyersloop_roi,
                    self.inputs.lh_v1_roi,
                    output_lh_ml_path,
                    distance_limit=None,
                    direction="roi_to_end",
                    keep_full_end=False
                )
                interpolate_tt(output_lh_ml_path, output_lh_ml_path, n_points_new=100, tie_at_center=True)
            self._refined_lh_ml = output_lh_ml_path
            
            output_rh_ml_path = os.path.join(self.inputs.output_dir, self.inputs.output_rh_ml)
            if self.inputs.rh_ml and os.path.exists(self.inputs.rh_ml):
                refine_and_orient_tt_by_roi(
                    self.inputs.rh_ml,
                    self.inputs.rh_meyersloop_roi,
                    self.inputs.rh_v1_roi,
                    output_rh_ml_path,
                    distance_limit=None,
                    direction="roi_to_end",
                    keep_full_end=False
                )
                interpolate_tt(output_rh_ml_path, output_rh_ml_path, n_points_new=100, tie_at_center=True)
            self._refined_rh_ml = output_rh_ml_path
        else:
            print(f"Right optic tract file {self.inputs.rh_ot} does not exist. Skipping refinement.")
            self._refined_rh_ot = ''

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["refined_lh_ot"] = self._refined_lh_ot
        outputs["refined_rh_ot"] = self._refined_rh_ot
        outputs["refined_lh_or"] = self._refined_lh_or
        outputs["refined_rh_or"] = self._refined_rh_or
        outputs["refined_lh_ml"] = self._refined_lh_ml
        outputs["refined_rh_ml"] = self._refined_rh_ml
        return outputs

if __name__== "__main__":

    filter_split_by_meyersloop_keep_lgn_segment(
        '/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OR.tt.gz',
        '/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/anat/sub-HC0062_ses-baseline_hemi-L_space-ACPC_label-LGN_desc-dilate3x_mask.nii.gz',
        '/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/anat/sub-HC0062_ses-baseline_hemi-L_space-ACPC_desc-roi4meyersloop_mask.nii.gz',
        '/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OR_1.tt.gz'
    )

    refine_and_orient_tt_by_roi(
        '/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OR_1.tt.gz',
        '/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/anat/sub-HC0062_ses-baseline_hemi-L_space-ACPC_label-LGN_desc-dilate3x_mask.nii.gz',
        '/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/anat/sub-HC0062_ses-baseline_hemi-L_space-ACPC_label-V1exvivo_desc-extend2mm_mask.nii.gz',
        '/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OR_2.tt.gz',
        distance_limit=None,
        direction="roi_to_end",
        keep_full_end=True
    )

    interpolate_tt('/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OR_2.tt.gz', '/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OR_3.tt.gz', n_points_new=100, tie_at_center=True)

    filter_or_tt_by_direction_window(
        input_tt='/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OR_3.tt.gz',
        output_tt='/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0062/ses-baseline/visual_pathway_analysis/raw_tracts/lh_OR_4.tt.gz',
        side="right",
        start_idx=20,
        end_idx=40,
        threshold=0.01
    )