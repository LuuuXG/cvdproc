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

def refine_and_orient_tt_by_roi(
    tt_file,
    roi_file,
    end_roi_file,
    output_file,
    distance_limit=None,
    direction="roi_to_end",
    keep_full_end=False
):
    """
    For each streamline in a TT file:
      - Find the points closest to the start and end ROI centers.
      - Keep the segment between them (optionally keep the full end side).
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
    keep_full_end : bool
        If True, keep the entire streamline beyond the end ROI (do not crop at end ROI center).
        Useful for cortical ROIs (e.g., V1), where center lies inside white matter.
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
        if distance_limit is not None and (
            d_start[i_start] > distance_limit or d_end[i_end] > distance_limit
        ):
            continue

        # --- Define segment boundaries ---
        if keep_full_end:
            # Keep from start ROI to the natural end of the streamline
            if i_start < i_end:
                seg = s[i_start:]
            else:
                seg = s[: i_start + 1][::-1].copy()
        else:
            # Standard: keep between start and end ROI centers
            if i_start < i_end:
                seg = s[i_start : i_end + 1]
            else:
                seg = s[i_end : i_start + 1][::-1].copy()

        # --- Orientation correction ---
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

def filter_or_tt_by_direction_window(
    input_tt, output_tt, side="left",
    start_idx=0, end_idx=5, threshold=0.5
):
    """
    Filter optic radiation (OR) streamlines in a TT (.tt.gz) file based on
    the x-direction trend within a specific point window.

    Parameters
    ----------
    input_tt : str
        Input TinyTrack .tt.gz file.
    output_tt : str
        Output TinyTrack .tt.gz file (filtered).
    side : str
        "left" or "right".
        - "left": keep streamlines with mean dx < -threshold
        - "right": keep streamlines with mean dx > threshold
    start_idx : int
        Starting point index for direction check (e.g., 10).
    end_idx : int
        Ending point index for direction check (e.g., 15).
    threshold : float
        Minimum mean delta-x (mm) to define valid direction.
    """

    # Load TT
    with gzip.open(input_tt, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)
    voxel_size = mat.get("voxel_size", np.array([1, 1, 1]))
    dimension = mat.get("dimension", np.array([1, 1, 1]))
    track_bytes = mat["track"]

    streamlines = parse_tt(track_bytes)
    print(f"[TT] Loaded {len(streamlines)} streamlines from {input_tt}")

    filtered = []
    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        n_pts = s.shape[0]
        if n_pts <= end_idx:
            continue
        # Compute x differences between start_idx and end_idx
        dx = np.diff(s[start_idx:end_idx, 0])
        mean_dx = np.mean(dx)

        if side == "left" and mean_dx < -threshold:
            filtered.append(s)
        elif side == "right" and mean_dx > threshold:
            filtered.append(s)

    print(f"[Filter] Original: {len(streamlines)} → Filtered: {len(filtered)} ({side} OR)")
    print(f"[Window] Used points {start_idx}–{end_idx}, threshold={threshold}")

    # Save output
    encoded = encode_tt(filtered)
    out_data = {"dimension": dimension, "voxel_size": voxel_size, "track": encoded}
    with gzip.open(output_tt, "wb") as f:
        savemat(f, out_data, format="4")
    print(f"[Save] {output_tt} (streamlines={len(filtered)})")

    return len(streamlines), len(filtered)

def filter_ot(tt_file, output_file, y_drop_threshold=10, eps=0.0, chiasm_roi=None):
    """
    Remove streamlines where:
      (1) y coordinate decreases consecutively for >= threshold points, OR
      (2) both endpoints are not on the same side of the chiasm ROI center along x-axis.

    Parameters
    ----------
    tt_file : str
        Input .tt.gz file (TinyTrack format)
    output_file : str
        Output filtered .tt.gz file
    y_drop_threshold : int
        Number of consecutive points with strictly decreasing y before exclusion (e.g. 5 or 10)
    eps : float
        Small tolerance; if diff < -eps, considered decreasing
    chiasm_roi : str or None
        NIfTI ROI for optic chiasm. If provided, only keep streamlines whose both endpoints
        lie on the same side (left or right) of the chiasm center x-coordinate.
    """
    def has_consecutive_y_drop(points, threshold, eps=0.0):
        dy = np.diff(points[:, 1])  # y differences
        cnt = 0
        for v in dy:
            if v < -eps:
                cnt += 1
                if cnt >= (threshold - 1):  # e.g., 10 points => 9 diffs
                    return True
            else:
                cnt = 0
        return False

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
    removed_side = 0

    for s in streamlines:
        if not isinstance(s, np.ndarray) or s.ndim != 2 or s.shape[1] != 3:
            continue

        # (1) check consecutive Y drop
        if has_consecutive_y_drop(s, y_drop_threshold, eps=eps):
            removed_y += 1
            continue

        # (2) check both endpoints relative to chiasm x center
        if chiasm_x_center is not None:
            x_start = s[0, 0]
            x_end = s[-1, 0]
            side_start = np.sign(x_start - chiasm_x_center)
            side_end = np.sign(x_end - chiasm_x_center)

            # must be on the same side (both > 0 or both < 0)
            if side_start == 0 or side_end == 0:
                pass  # exactly at center → keep
            elif side_start != side_end:
                removed_side += 1
                continue

        kept_streamlines.append(s)

    removed_total = removed_y + removed_side
    print(f"[Filter] Removed {removed_total}/{len(streamlines)} "
          f"({100 * removed_total / len(streamlines):.1f}%) "
          f"— ({removed_y} by Y-drop, {removed_side} by chiasm-side)")

    encoded = encode_tt(kept_streamlines)
    out = {"dimension": dim, "voxel_size": vsz, "track": encoded}
    with gzip.open(output_file, "wb") as f:
        savemat(f, out, format="4")

    print(f"[Save] {output_file} (remaining={len(kept_streamlines)})")

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
    cho_roi = File(exists=True, desc="Chiasm ROI NIfTI file", mandatory=True)
    #cho_dia1_roi = File(exists=True, desc="Chiasm dilated 1 voxel ROI NIfTI file", mandatory=False)
    lh_lgn_roi = File(exists=True, desc="Left LGN ROI NIfTI file", mandatory=True)
    #lh_lgn_dia1_roi = File(exists=True, desc="Left LGN dilated 1 voxel ROI NIfTI file", mandatory=False)
    rh_lgn_roi = File(exists=True, desc="Right LGN ROI NIfTI file", mandatory=True)
    #rh_lgn_dia1_roi = File(exists=True, desc="Right LGN dilated 1 voxel ROI NIfTI file", mandatory=False)
    lh_v1_roi = File(exists=True, desc="Left V1 ROI NIfTI file", mandatory=False)
    rh_v1_roi = File(exists=True, desc="Right V1 ROI NIfTI file", mandatory=False)
    
    output_dir = Str(desc="Output directory", mandatory=True)
    output_lh_ot = Str("refined_lh_optic_tract.tt.gz", desc="Output filename for refined left optic tract")
    output_rh_ot = Str("refined_rh_optic_tract.tt.gz", desc="Output filename for refined right optic tract")
    output_lh_or = Str("refined_lh_optic_radiation.tt.gz", desc="Output filename for refined left optic radiation")
    output_rh_or = Str("refined_rh_optic_radiation.tt.gz", desc="Output filename for refined right optic radiation")

class RefineVPOutputSpec(TraitedSpec):
    refined_lh_ot = Str(desc="Refined left hemisphere optic tract .tt.gz file")
    refined_rh_ot = Str(desc="Refined right hemisphere optic tract .tt.gz file")
    refined_lh_or = Str(desc="Refined left hemisphere optic radiation .tt.gz file")
    refined_rh_or = Str(desc="Refined right hemisphere optic radiation .tt.gz file")

class RefineVP(BaseInterface):
    input_spec = RefineVPInputSpec
    output_spec = RefineVPOutputSpec

    def _run_interface(self, runtime):
        os.makedirs(self.inputs.output_dir, exist_ok=True)

        # Refine and orient optic radiation
        if os.path.exists(self.inputs.lh_or):
            split_or_and_merge_to_ot(
                or_tt_file=self.inputs.lh_or,
                lgn_roi=self.inputs.lh_lgn_roi,
                chiasm_roi=self.inputs.cho_roi,
                ot_tt_file=self.inputs.lh_ot,
                output_ot_file=self.inputs.lh_ot
            )

            output_lh_or_path = os.path.join(self.inputs.output_dir, self.inputs.output_lh_or)
            refine_and_orient_tt_by_roi(
                self.inputs.lh_or,
                self.inputs.lh_lgn_roi,
                self.inputs.lh_v1_roi,
                output_lh_or_path,
                distance_limit=None,
                direction="roi_to_end",
                keep_full_end=True
            )

            interpolate_tt(output_lh_or_path, output_lh_or_path, n_points_new=100, tie_at_center=True)

            filter_or_tt_by_direction_window(
                input_tt=output_lh_or_path,
                output_tt=output_lh_or_path,
                side="right",
                start_idx=5,
                end_idx=10,
                threshold=0.1
            )

            self._refined_lh_or = output_lh_or_path
        else:
            print(f"Left optic radiation file {self.inputs.lh_or} does not exist. Skipping refinement.")
            self._refined_lh_or = ''

        if os.path.exists(self.inputs.rh_or):
            split_or_and_merge_to_ot(
                or_tt_file=self.inputs.rh_or,
                lgn_roi=self.inputs.rh_lgn_roi,
                chiasm_roi=self.inputs.cho_roi,
                ot_tt_file=self.inputs.rh_ot,
                output_ot_file=self.inputs.rh_ot
            )

            output_rh_or_path = os.path.join(self.inputs.output_dir, self.inputs.output_rh_or)
            refine_and_orient_tt_by_roi(
                self.inputs.rh_or,
                self.inputs.rh_lgn_roi,
                self.inputs.rh_v1_roi,
                output_rh_or_path,
                distance_limit=None,
                direction="roi_to_end",
                keep_full_end=True
            )

            interpolate_tt(output_rh_or_path, output_rh_or_path, n_points_new=100, tie_at_center=True)

            filter_or_tt_by_direction_window(
                input_tt=output_rh_or_path,
                output_tt=output_rh_or_path,
                side="left",
                start_idx=5,
                end_idx=10,
                threshold=0.1
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
            # refine_and_orient_tt_by_roi(
            #     output_lh_ot_path,
            #     self.inputs.lh_lgn_roi,
            #     self.inputs.cho_roi,
            #     output_lh_ot_path,
            #     distance_limit=None,
            #     direction="end_to_roi"
            # )

            interpolate_tt(output_lh_ot_path, output_lh_ot_path, n_points_new=100, tie_at_center=True)

            filter_ot(
                tt_file=output_lh_ot_path,
                output_file=output_lh_ot_path,
                y_drop_threshold=3,
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
            # refine_and_orient_tt_by_roi(
            #     output_rh_ot_path,
            #     self.inputs.rh_lgn_roi,
            #     self.inputs.cho_roi,
            #     output_rh_ot_path,
            #     distance_limit=None,
            #     direction="end_to_roi"
            # )

            interpolate_tt(output_rh_ot_path, output_rh_ot_path, n_points_new=100, tie_at_center=True)

            filter_ot(
                tt_file=output_rh_ot_path,
                output_file=output_rh_ot_path,
                y_drop_threshold=3,
                eps=0.0
            )

            self._refined_rh_ot = output_rh_ot_path
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
        return outputs