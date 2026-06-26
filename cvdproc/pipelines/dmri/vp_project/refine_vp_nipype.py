import os
import numpy as np
import nibabel as nib

from scipy.interpolate import splprep, splev
from dipy.segment.clustering import QuickBundles

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
)
from traits.api import Str, Bool, Int, Float

from cvdproc.pipelines.dmri.dsistudio.tt_utils import TinyTrackIO


def load_mask(mask_file):
    if mask_file is None or not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")

    mask = nib.load(mask_file).get_fdata() > 0
    if not mask.any():
        raise ValueError(f"Mask is empty: {mask_file}")

    return mask


def mask_center(mask_bool):
    coords = np.argwhere(mask_bool)
    if coords.size == 0:
        raise ValueError("ROI mask is empty.")
    return coords.mean(axis=0).astype(np.float32)


def in_bounds(ijk, shape):
    ijk = np.asarray(ijk, dtype=np.int32)

    if ijk.ndim == 1:
        return (
            0 <= ijk[0] < shape[0]
            and 0 <= ijk[1] < shape[1]
            and 0 <= ijk[2] < shape[2]
        )

    return (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0])
        & (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1])
        & (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
    )


def point_in_mask(point, mask_bool):
    ijk = np.rint(point).astype(np.int32)

    if ijk.shape[0] != 3:
        return False

    if not in_bounds(ijk, mask_bool.shape):
        return False

    return bool(mask_bool[ijk[0], ijk[1], ijk[2]])


def streamline_hit_indices(streamline, mask_bool):
    s = np.asarray(streamline, dtype=np.float32)

    if s.ndim != 2 or s.shape[1] != 3:
        return np.array([], dtype=np.int64)

    ijk = np.rint(s).astype(np.int32)
    valid = in_bounds(ijk, mask_bool.shape)

    if not np.any(valid):
        return np.array([], dtype=np.int64)

    valid_idx = np.where(valid)[0]
    ijk_valid = ijk[valid]

    hit = mask_bool[
        ijk_valid[:, 0],
        ijk_valid[:, 1],
        ijk_valid[:, 2],
    ]

    return valid_idx[hit]


def streamline_hits_mask(streamline, mask_bool):
    return streamline_hit_indices(streamline, mask_bool).size > 0


def endpoint_in_mask(point, mask_bool):
    return point_in_mask(point, mask_bool)


def max_step_length(streamline):
    s = np.asarray(streamline, dtype=np.float32)

    if s.ndim != 2 or s.shape[0] < 2 or s.shape[1] != 3:
        return np.inf

    return float(np.max(np.linalg.norm(np.diff(s, axis=0), axis=1)))


def filter_by_max_step(streamlines, max_step):
    kept = []
    removed = 0

    for s in streamlines:
        if max_step_length(s) <= float(max_step):
            kept.append(s)
        else:
            removed += 1

    return kept, removed


def print_endpoint_summary(name, streamlines):
    valid = [
        np.asarray(s, dtype=np.float32)
        for s in streamlines
        if np.asarray(s).ndim == 2 and np.asarray(s).shape[0] >= 2
    ]

    if len(valid) == 0:
        print(f"[{name}] Empty streamline set")
        return

    starts = np.asarray([s[0] for s in valid], dtype=np.float32)
    ends = np.asarray([s[-1] for s in valid], dtype=np.float32)

    print(f"[{name}] Streamlines: {len(valid)}")
    print(f"[{name}] Mean START: {starts.mean(axis=0)}")
    print(f"[{name}] Mean END:   {ends.mean(axis=0)}")


def load_tt_data(tt_file):
    ttio = TinyTrackIO()
    return ttio.load(tt_file, preserve_metadata=True)


def save_tt_data(data, streamlines, output_file):
    ttio = TinyTrackIO()
    ttio.save(
        streamlines=streamlines,
        output_file=output_file,
        dimension=data.dimension,
        voxel_size=data.voxel_size,
        metadata=data.metadata,
    )


def trim_endpoint_roi_points(streamline, roi_mask, min_points=2):
    s = np.asarray(streamline, dtype=np.float32)

    if s.ndim != 2 or s.shape[0] < min_points or s.shape[1] != 3:
        return None

    start = 0
    end = s.shape[0]

    while start < end and point_in_mask(s[start], roi_mask):
        start += 1

    while end > start and point_in_mask(s[end - 1], roi_mask):
        end -= 1

    out = s[start:end]

    if out.shape[0] < min_points:
        return None

    return out


def trim_endpoint_roi_from_streamlines(streamlines, roi_file, min_points=2):
    roi_mask = load_mask(roi_file)

    kept = []
    dropped = 0
    changed = 0

    for s in streamlines:
        s2 = trim_endpoint_roi_points(s, roi_mask, min_points=min_points)

        if s2 is None:
            dropped += 1
            continue

        if s2.shape[0] != np.asarray(s).shape[0]:
            changed += 1

        kept.append(s2)

    print(f"[TrimROI] Changed: {changed}")
    print(f"[TrimROI] Dropped: {dropped}")
    print(f"[TrimROI] Output: {len(kept)}")

    return kept


def drop_streamlines_hitting_roi(streamlines, roi_file):
    roi_mask = load_mask(roi_file)

    kept = []
    dropped = 0

    for s in streamlines:
        if streamline_hits_mask(s, roi_mask):
            dropped += 1
            continue
        kept.append(s)

    print(f"[DropROI] Dropped by ROI hit: {dropped}")
    print(f"[DropROI] Output: {len(kept)}")

    return kept


def split_by_meyersloop_keep_lgn_segment(
    streamlines,
    lgn_roi,
    meyersloop_roi,
    min_points=5,
    keep_original_if_ambiguous=True,
    drop_if_not_both_hit=False,
):
    lgn_mask = load_mask(lgn_roi)
    ml_mask = load_mask(meyersloop_roi)
    ml_center = mask_center(ml_mask)

    kept = []

    n_hit_lgn = 0
    n_hit_ml = 0
    n_hit_both = 0
    n_modified = 0
    n_dropped_short = 0
    n_dropped_not_both = 0
    n_ambiguous_dropped = 0

    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)

        if s.ndim != 2 or s.shape[0] < min_points or s.shape[1] != 3:
            n_dropped_short += 1
            continue

        hit_lgn = streamline_hits_mask(s, lgn_mask)
        hit_ml = streamline_hits_mask(s, ml_mask)

        if hit_lgn:
            n_hit_lgn += 1
        if hit_ml:
            n_hit_ml += 1
        if hit_lgn and hit_ml:
            n_hit_both += 1

        if not (hit_lgn and hit_ml):
            if drop_if_not_both_hit:
                n_dropped_not_both += 1
                continue
            kept.append(s)
            continue

        d_ml = np.linalg.norm(s - ml_center[None, :], axis=1)
        i_split = int(np.argmin(d_ml))

        seg1 = s[:i_split + 1]
        seg2 = s[i_split:]

        if seg1.shape[0] < min_points or seg2.shape[0] < min_points:
            kept.append(s)
            continue

        seg1_lgn = streamline_hits_mask(seg1, lgn_mask)
        seg2_lgn = streamline_hits_mask(seg2, lgn_mask)

        if seg1_lgn and not seg2_lgn:
            kept.append(seg1)
            n_modified += 1
        elif seg2_lgn and not seg1_lgn:
            kept.append(seg2)
            n_modified += 1
        else:
            if keep_original_if_ambiguous:
                kept.append(s)
            else:
                n_ambiguous_dropped += 1

    print(f"[FilterSplit] Input: {len(streamlines)}")
    print(f"[FilterSplit] Hit LGN: {n_hit_lgn}")
    print(f"[FilterSplit] Hit MeyersLoop: {n_hit_ml}")
    print(f"[FilterSplit] Hit BOTH: {n_hit_both}")
    print(f"[FilterSplit] Modified: {n_modified}")
    print(f"[FilterSplit] Dropped short: {n_dropped_short}")
    print(f"[FilterSplit] Dropped not both-hit: {n_dropped_not_both}")
    print(f"[FilterSplit] Dropped ambiguous: {n_ambiguous_dropped}")
    print(f"[FilterSplit] Output: {len(kept)}")

    return kept


def refine_and_orient_by_roi(
    streamlines,
    roi_file,
    end_roi_file,
    distance_limit=None,
    direction="roi_to_end",
    keep_full_end=False,
    must_in_roi=False,
):
    if direction not in ("roi_to_end", "end_to_roi"):
        raise ValueError("direction must be 'roi_to_end' or 'end_to_roi'")

    start_roi_mask = load_mask(roi_file)
    end_roi_mask = load_mask(end_roi_file)

    start_center = mask_center(start_roi_mask)
    end_center = mask_center(end_roi_mask)

    refined = []

    dropped_must_in_roi = 0
    dropped_distance = 0
    dropped_short = 0

    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)

        if s.ndim != 2 or s.shape[0] < 2 or s.shape[1] != 3:
            dropped_short += 1
            continue

        d_start = np.linalg.norm(s - start_center[None, :], axis=1)
        d_end = np.linalg.norm(s - end_center[None, :], axis=1)

        i_start = int(np.argmin(d_start))
        i_end = int(np.argmin(d_end))

        if distance_limit is not None:
            if d_start[i_start] > distance_limit or d_end[i_end] > distance_limit:
                dropped_distance += 1
                continue

        if keep_full_end:
            if i_start < i_end:
                seg = s[i_start:]
            else:
                seg = s[:i_start + 1][::-1].copy()
        else:
            if i_start < i_end:
                seg = s[i_start:i_end + 1]
            else:
                seg = s[i_end:i_start + 1][::-1].copy()

        if seg.shape[0] < 2:
            dropped_short += 1
            continue

        if direction == "roi_to_end":
            if np.linalg.norm(seg[0] - start_center) > np.linalg.norm(seg[-1] - start_center):
                seg = seg[::-1].copy()
        else:
            if np.linalg.norm(seg[0] - end_center) > np.linalg.norm(seg[-1] - end_center):
                seg = seg[::-1].copy()

        if must_in_roi:
            if direction == "roi_to_end":
                ok = endpoint_in_mask(seg[-1], end_roi_mask)
            else:
                ok = endpoint_in_mask(seg[0], end_roi_mask)

            if not ok:
                dropped_must_in_roi += 1
                continue

        refined.append(seg)

    total = len(streamlines)
    kept = len(refined)
    pct = 100.0 * kept / total if total > 0 else 0.0

    print(f"[Refine] Kept {kept}/{total} ({pct:.1f}%)")
    print(f"[Refine] Dropped short: {dropped_short}")
    print(f"[Refine] Dropped by distance_limit: {dropped_distance}")
    print(f"[Refine] Dropped by must_in_roi: {dropped_must_in_roi}")

    return refined


def trk_interp(streamlines, n_points_new=100, spacing=None, tie_at_center=False):
    tracks_interp = [None] * len(streamlines)
    splines = []

    for idx, coords in enumerate(streamlines):
        coords = np.asarray(coords, dtype=np.float32)

        if coords.ndim != 2 or coords.shape[0] < 4 or coords.shape[1] != 3:
            if coords.ndim == 2 and coords.shape[0] > 0 and coords.shape[1] == 3:
                tracks_interp[idx] = np.repeat(coords[:1], n_points_new, axis=0)
            else:
                tracks_interp[idx] = np.zeros((n_points_new, 3), dtype=np.float32)
            continue

        diffs = np.diff(coords, axis=0)
        segs = np.linalg.norm(diffs, axis=1)
        dist = np.insert(np.cumsum(segs), 0, 0.0)

        if len(np.unique(dist)) < 4:
            tracks_interp[idx] = np.repeat(coords[:1], n_points_new, axis=0)
            continue

        try:
            tck, _ = splprep(coords.T, u=dist, s=0, k=min(3, len(coords) - 1))
            splines.append((idx, tck, dist[-1]))
        except Exception:
            tracks_interp[idx] = coords.copy()

    for idx, tck, total_len in splines:
        u_new = np.linspace(0, total_len, n_points_new)
        interp_coords = np.array(splev(u_new, tck)).T
        tracks_interp[idx] = interp_coords.astype(np.float32)

    if tie_at_center and spacing is None and len(splines) > 0:
        n_points_odd = int(np.floor(n_points_new / 2) * 2 + 1)

        mean_track = np.mean(
            np.stack([
                np.array(splev(np.linspace(0, item[2], n_points_odd), item[1])).T
                for item in splines
            ]),
            axis=0,
        )

        middle = mean_track[len(mean_track) // 2]

        for idx, tck, total_len in splines:
            coords = np.array(splev(np.linspace(0, total_len, n_points_odd), tck)).T
            dists = np.linalg.norm(coords - middle, axis=1)
            ind = int(np.argmin(dists))

            first_half = np.array(
                splev(
                    np.linspace(0, total_len * (ind / n_points_odd), n_points_odd // 2 + 1),
                    tck,
                )
            ).T

            second_half = np.array(
                splev(
                    np.linspace(total_len * (ind / n_points_odd), total_len, n_points_odd // 2 + 1),
                    tck,
                )
            ).T

            merged = np.vstack([first_half, second_half[1:]])

            if merged.shape[0] < n_points_new:
                pad = np.repeat(merged[-1][None, :], n_points_new - merged.shape[0], axis=0)
                merged = np.vstack([merged, pad])
            elif merged.shape[0] > n_points_new:
                merged = merged[:n_points_new]

            tracks_interp[idx] = merged.astype(np.float32)

    for i, t in enumerate(tracks_interp):
        if t is None:
            t = np.zeros((1, 3), dtype=np.float32)

        t = np.asarray(t, dtype=np.float32)

        if t.shape[0] < n_points_new:
            pad = np.repeat(t[-1][None, :], n_points_new - t.shape[0], axis=0)
            tracks_interp[i] = np.vstack([t, pad]).astype(np.float32)
        elif t.shape[0] > n_points_new:
            tracks_interp[i] = t[:n_points_new].astype(np.float32)
        else:
            tracks_interp[i] = t.astype(np.float32)

    return tracks_interp


def interpolate_streamlines(streamlines, n_points_new=100, tie_at_center=True):
    print(f"[Interpolate] Input: {len(streamlines)}")
    out = trk_interp(
        streamlines,
        n_points_new=n_points_new,
        spacing=None,
        tie_at_center=tie_at_center,
    )
    print(f"[Interpolate] Output: {len(out)}")
    return out


def cluster_keep_largest_streamline_cluster(streamlines, threshold=10.0):
    print(f"[QB] Input: {len(streamlines)}")
    print(f"[QB] Threshold: {threshold}")

    if len(streamlines) <= 1:
        return streamlines

    qb = QuickBundles(threshold=float(threshold))
    clusters = qb.cluster(streamlines)

    cluster_sizes = [len(cluster) for cluster in clusters]

    print(f"[QB] Number of clusters: {len(clusters)}")

    for i, size in enumerate(cluster_sizes):
        print(f"[QB] Cluster {i}: {size} streamlines")

    largest_cluster_index = int(np.argmax(cluster_sizes))
    largest_cluster = clusters[largest_cluster_index]
    keep_indices = list(largest_cluster.indices)

    out = [streamlines[i] for i in keep_indices]

    print(f"[QB] Keeping cluster {largest_cluster_index}: {len(out)}/{len(streamlines)}")

    return out


def has_consecutive_drop(values, threshold, eps=0.0):
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


def filter_ot_streamlines(streamlines, chiasm_roi=None, y_drop_threshold=3, z_drop_threshold=5, eps=0.0):
    chiasm_x_center = None

    if chiasm_roi is not None and os.path.exists(chiasm_roi):
        chiasm_mask = load_mask(chiasm_roi)
        coords = np.argwhere(chiasm_mask)

        if coords.size > 0:
            chiasm_x_center = float(coords[:, 0].mean())
            print(f"[OTFilter] Chiasm center x = {chiasm_x_center:.2f}")

    kept = []
    removed_y = 0
    removed_z = 0
    removed_side = 0
    removed_short = 0

    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)

        if s.ndim != 2 or s.shape[1] != 3 or s.shape[0] < 2:
            removed_short += 1
            continue

        if has_consecutive_drop(s[:, 1], y_drop_threshold, eps):
            removed_y += 1
            continue

        if has_consecutive_drop(s[:, 2], z_drop_threshold, eps):
            removed_z += 1
            continue

        if chiasm_x_center is not None:
            x_start = s[0, 0]
            x_end = s[-1, 0]

            side_start = np.sign(x_start - chiasm_x_center)
            side_end = np.sign(x_end - chiasm_x_center)

            if side_start != 0 and side_end != 0 and side_start != side_end:
                removed_side += 1
                continue

        kept.append(s)

    removed_total = removed_y + removed_z + removed_side + removed_short
    pct = 100.0 * removed_total / len(streamlines) if len(streamlines) > 0 else 0.0

    print(
        f"[OTFilter] Removed {removed_total}/{len(streamlines)} ({pct:.1f}%) "
        f"- short={removed_short}, y={removed_y}, z={removed_z}, side={removed_side}"
    )
    print(f"[OTFilter] Output: {len(kept)}")

    return kept


def run_or_refinement(
    input_tt,
    output_tt,
    lgn_dia_x_roi,
    lgn_roi,
    lgn_extendpart_roi,
    meyersloop_roi,
    v1_roi,
    min_points=5,
    max_step=10.0,
    n_points_new=100,
    qb_threshold=10.0,
    run_clustering=True,
):
    data = load_tt_data(input_tt)
    streamlines = data.streamlines

    print(f"[OR] Loaded: {input_tt}")
    print_endpoint_summary("OR input", streamlines)

    streamlines = split_by_meyersloop_keep_lgn_segment(
        streamlines,
        lgn_roi=lgn_dia_x_roi,
        meyersloop_roi=meyersloop_roi,
        min_points=min_points,
        keep_original_if_ambiguous=True,
        drop_if_not_both_hit=False,
    )

    streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
    print(f"[OR] Removed by max_step after split: {removed_step}")

    streamlines = refine_and_orient_by_roi(
        streamlines,
        roi_file=lgn_extendpart_roi,
        end_roi_file=v1_roi,
        distance_limit=None,
        direction="roi_to_end",
        keep_full_end=True,
        must_in_roi=True,
    )

    streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
    print(f"[OR] Removed by max_step after refine: {removed_step}")

    streamlines = trim_endpoint_roi_from_streamlines(
        streamlines,
        roi_file=lgn_dia_x_roi,
        min_points=min_points,
    )

    streamlines = trim_endpoint_roi_from_streamlines(
        streamlines,
        roi_file=lgn_roi,
        min_points=min_points,
    )

    streamlines = drop_streamlines_hitting_roi(
        streamlines,
        roi_file=meyersloop_roi,
    )

    streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
    print(f"[OR] Removed by max_step after ROI cleanup: {removed_step}")

    streamlines = interpolate_streamlines(
        streamlines,
        n_points_new=n_points_new,
        tie_at_center=True,
    )

    streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
    print(f"[OR] Removed by max_step after interpolation: {removed_step}")

    if run_clustering:
        streamlines = cluster_keep_largest_streamline_cluster(
            streamlines,
            threshold=qb_threshold,
        )

        streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
        print(f"[OR] Removed by max_step after clustering: {removed_step}")
    else:
        print("[OR] Clustering skipped")

    print_endpoint_summary("OR output", streamlines)

    save_tt_data(data, streamlines, output_tt)
    print(f"[OR] Saved: {output_tt}")

    return output_tt


def run_ot_refinement(
    input_tt,
    output_tt,
    chiasm_roi,
    lgn_roi,
    min_points=5,
    max_step=10.0,
    n_points_new=100,
    qb_threshold=10.0,
    run_clustering=True,
    run_direction_filter=False,
    y_drop_threshold=3,
    z_drop_threshold=5,
):
    data = load_tt_data(input_tt)
    streamlines = data.streamlines

    print(f"[OT] Loaded: {input_tt}")
    print_endpoint_summary("OT input", streamlines)

    streamlines = refine_and_orient_by_roi(
        streamlines,
        roi_file=chiasm_roi,
        end_roi_file=lgn_roi,
        distance_limit=None,
        direction="roi_to_end",
        keep_full_end=False,
        must_in_roi=False,
    )

    streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
    print(f"[OT] Removed by max_step after refine: {removed_step}")

    streamlines = trim_endpoint_roi_from_streamlines(
        streamlines,
        roi_file=lgn_roi,
        min_points=min_points,
    )

    streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
    print(f"[OT] Removed by max_step after LGN trim: {removed_step}")

    streamlines = interpolate_streamlines(
        streamlines,
        n_points_new=n_points_new,
        tie_at_center=True,
    )

    streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
    print(f"[OT] Removed by max_step after interpolation: {removed_step}")

    if run_direction_filter:
        streamlines = filter_ot_streamlines(
            streamlines,
            chiasm_roi=chiasm_roi,
            y_drop_threshold=y_drop_threshold,
            z_drop_threshold=z_drop_threshold,
            eps=0.0,
        )

        streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
        print(f"[OT] Removed by max_step after OT direction filter: {removed_step}")
    else:
        print("[OT] Direction filter skipped")

    if run_clustering:
        streamlines = cluster_keep_largest_streamline_cluster(
            streamlines,
            threshold=qb_threshold,
        )

        streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
        print(f"[OT] Removed by max_step after clustering: {removed_step}")
    else:
        print("[OT] Clustering skipped")

    print_endpoint_summary("OT output", streamlines)

    save_tt_data(data, streamlines, output_tt)
    print(f"[OT] Saved: {output_tt}")

    return output_tt


def run_ml_refinement(
    input_tt,
    output_tt,
    meyersloop_roi,
    v1_roi,
    max_step=10.0,
    n_points_new=100,
):
    data = load_tt_data(input_tt)
    streamlines = data.streamlines

    print(f"[ML] Loaded: {input_tt}")
    print_endpoint_summary("ML input", streamlines)

    streamlines = refine_and_orient_by_roi(
        streamlines,
        roi_file=meyersloop_roi,
        end_roi_file=v1_roi,
        distance_limit=None,
        direction="roi_to_end",
        keep_full_end=False,
        must_in_roi=False,
    )

    streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
    print(f"[ML] Removed by max_step after refine: {removed_step}")

    streamlines = interpolate_streamlines(
        streamlines,
        n_points_new=n_points_new,
        tie_at_center=True,
    )

    streamlines, removed_step = filter_by_max_step(streamlines, max_step=max_step)
    print(f"[ML] Removed by max_step after interpolation: {removed_step}")

    print_endpoint_summary("ML output", streamlines)

    save_tt_data(data, streamlines, output_tt)
    print(f"[ML] Saved: {output_tt}")

    return output_tt


class RefineVPInputSpec(BaseInterfaceInputSpec):
    lh_ot = Str(desc="Left optic tract TT file", mandatory=True)
    rh_ot = Str(desc="Right optic tract TT file", mandatory=True)
    lh_or = Str(desc="Left optic radiation TT file", mandatory=True)
    rh_or = Str(desc="Right optic radiation TT file", mandatory=True)

    lh_ml = Str(desc="Left Meyer's loop TT file", mandatory=False)
    rh_ml = Str(desc="Right Meyer's loop TT file", mandatory=False)

    cho_roi = File(exists=True, desc="Chiasm ROI NIfTI file", mandatory=True)

    lh_lgn_roi = File(exists=True, desc="Left LGN ROI NIfTI file", mandatory=True)
    lh_lgn_dia_x_roi = File(exists=True, desc="Left dilated LGN ROI NIfTI file", mandatory=True)
    lh_lgn_extendpart_roi = File(exists=True, desc="Left extended LGN ROI NIfTI file", mandatory=True)

    rh_lgn_roi = File(exists=True, desc="Right LGN ROI NIfTI file", mandatory=True)
    rh_lgn_dia_x_roi = File(exists=True, desc="Right dilated LGN ROI NIfTI file", mandatory=True)
    rh_lgn_extendpart_roi = File(exists=True, desc="Right extended LGN ROI NIfTI file", mandatory=True)

    lh_v1_roi = File(exists=True, desc="Left V1 ROI NIfTI file", mandatory=True)
    rh_v1_roi = File(exists=True, desc="Right V1 ROI NIfTI file", mandatory=True)

    lh_meyersloop_roi = File(exists=True, desc="Left Meyer's loop ROI NIfTI file", mandatory=True)
    rh_meyersloop_roi = File(exists=True, desc="Right Meyer's loop ROI NIfTI file", mandatory=True)

    output_dir = Str(desc="Output directory", mandatory=True)

    output_lh_ot = Str("refined_lh_optic_tract.tt.gz", usedefault=True)
    output_rh_ot = Str("refined_rh_optic_tract.tt.gz", usedefault=True)
    output_lh_or = Str("refined_lh_optic_radiation.tt.gz", usedefault=True)
    output_rh_or = Str("refined_rh_optic_radiation.tt.gz", usedefault=True)
    output_lh_ml = Str("refined_lh_meyers_loop.tt.gz", usedefault=True)
    output_rh_ml = Str("refined_rh_meyers_loop.tt.gz", usedefault=True)

    min_points = Int(5, usedefault=True)
    max_step = Float(10.0, usedefault=True)
    n_points_new = Int(100, usedefault=True)

    or_qb_threshold = Float(10.0, usedefault=True)
    ot_qb_threshold = Float(10.0, usedefault=True)

    ot_y_drop_threshold = Int(3, usedefault=True)
    ot_z_drop_threshold = Int(5, usedefault=True)

    run_or_clustering = Bool(True, usedefault=True)
    run_ot_clustering = Bool(True, usedefault=True)
    run_ot_direction_filter = Bool(False, usedefault=True)


class RefineVPOutputSpec(TraitedSpec):
    refined_lh_ot = Str(desc="Refined left optic tract TT file")
    refined_rh_ot = Str(desc="Refined right optic tract TT file")
    refined_lh_or = Str(desc="Refined left optic radiation TT file")
    refined_rh_or = Str(desc="Refined right optic radiation TT file")
    refined_lh_ml = Str(desc="Refined left Meyer's loop TT file")
    refined_rh_ml = Str(desc="Refined right Meyer's loop TT file")


class RefineVP(BaseInterface):
    input_spec = RefineVPInputSpec
    output_spec = RefineVPOutputSpec

    def _run_interface(self, runtime):
        os.makedirs(self.inputs.output_dir, exist_ok=True)

        self._refined_lh_or = ""
        self._refined_rh_or = ""
        self._refined_lh_ot = ""
        self._refined_rh_ot = ""
        self._refined_lh_ml = ""
        self._refined_rh_ml = ""

        output_lh_or_path = os.path.abspath(
            os.path.join(self.inputs.output_dir, self.inputs.output_lh_or)
        )
        output_rh_or_path = os.path.abspath(
            os.path.join(self.inputs.output_dir, self.inputs.output_rh_or)
        )
        output_lh_ot_path = os.path.abspath(
            os.path.join(self.inputs.output_dir, self.inputs.output_lh_ot)
        )
        output_rh_ot_path = os.path.abspath(
            os.path.join(self.inputs.output_dir, self.inputs.output_rh_ot)
        )
        output_lh_ml_path = os.path.abspath(
            os.path.join(self.inputs.output_dir, self.inputs.output_lh_ml)
        )
        output_rh_ml_path = os.path.abspath(
            os.path.join(self.inputs.output_dir, self.inputs.output_rh_ml)
        )

        if os.path.exists(self.inputs.lh_or):
            self._refined_lh_or = run_or_refinement(
                input_tt=self.inputs.lh_or,
                output_tt=output_lh_or_path,
                lgn_dia_x_roi=self.inputs.lh_lgn_dia_x_roi,
                lgn_roi=self.inputs.lh_lgn_roi,
                lgn_extendpart_roi=self.inputs.lh_lgn_extendpart_roi,
                meyersloop_roi=self.inputs.lh_meyersloop_roi,
                v1_roi=self.inputs.lh_v1_roi,
                min_points=int(self.inputs.min_points),
                max_step=float(self.inputs.max_step),
                n_points_new=int(self.inputs.n_points_new),
                qb_threshold=float(self.inputs.or_qb_threshold),
                run_clustering=bool(self.inputs.run_or_clustering),
            )
        else:
            print(f"[RefineVP] Missing left OR: {self.inputs.lh_or}")

        if os.path.exists(self.inputs.rh_or):
            self._refined_rh_or = run_or_refinement(
                input_tt=self.inputs.rh_or,
                output_tt=output_rh_or_path,
                lgn_dia_x_roi=self.inputs.rh_lgn_dia_x_roi,
                lgn_roi=self.inputs.rh_lgn_roi,
                lgn_extendpart_roi=self.inputs.rh_lgn_extendpart_roi,
                meyersloop_roi=self.inputs.rh_meyersloop_roi,
                v1_roi=self.inputs.rh_v1_roi,
                min_points=int(self.inputs.min_points),
                max_step=float(self.inputs.max_step),
                n_points_new=int(self.inputs.n_points_new),
                qb_threshold=float(self.inputs.or_qb_threshold),
                run_clustering=bool(self.inputs.run_or_clustering),
            )
        else:
            print(f"[RefineVP] Missing right OR: {self.inputs.rh_or}")

        if os.path.exists(self.inputs.lh_ot):
            self._refined_lh_ot = run_ot_refinement(
                input_tt=self.inputs.lh_ot,
                output_tt=output_lh_ot_path,
                chiasm_roi=self.inputs.cho_roi,
                lgn_roi=self.inputs.lh_lgn_roi,
                min_points=int(self.inputs.min_points),
                max_step=float(self.inputs.max_step),
                n_points_new=int(self.inputs.n_points_new),
                qb_threshold=float(self.inputs.ot_qb_threshold),
                run_clustering=bool(self.inputs.run_ot_clustering),
                run_direction_filter=bool(self.inputs.run_ot_direction_filter),
                y_drop_threshold=int(self.inputs.ot_y_drop_threshold),
                z_drop_threshold=int(self.inputs.ot_z_drop_threshold),
            )
        else:
            print(f"[RefineVP] Missing left OT: {self.inputs.lh_ot}")

        if os.path.exists(self.inputs.rh_ot):
            self._refined_rh_ot = run_ot_refinement(
                input_tt=self.inputs.rh_ot,
                output_tt=output_rh_ot_path,
                chiasm_roi=self.inputs.cho_roi,
                lgn_roi=self.inputs.rh_lgn_roi,
                min_points=int(self.inputs.min_points),
                max_step=float(self.inputs.max_step),
                n_points_new=int(self.inputs.n_points_new),
                qb_threshold=float(self.inputs.ot_qb_threshold),
                run_clustering=bool(self.inputs.run_ot_clustering),
                run_direction_filter=bool(self.inputs.run_ot_direction_filter),
                y_drop_threshold=int(self.inputs.ot_y_drop_threshold),
                z_drop_threshold=int(self.inputs.ot_z_drop_threshold),
            )
        else:
            print(f"[RefineVP] Missing right OT: {self.inputs.rh_ot}")

        if self.inputs.lh_ml and os.path.exists(self.inputs.lh_ml):
            self._refined_lh_ml = run_ml_refinement(
                input_tt=self.inputs.lh_ml,
                output_tt=output_lh_ml_path,
                meyersloop_roi=self.inputs.lh_meyersloop_roi,
                v1_roi=self.inputs.lh_v1_roi,
                max_step=float(self.inputs.max_step),
                n_points_new=int(self.inputs.n_points_new),
            )

        if self.inputs.rh_ml and os.path.exists(self.inputs.rh_ml):
            self._refined_rh_ml = run_ml_refinement(
                input_tt=self.inputs.rh_ml,
                output_tt=output_rh_ml_path,
                meyersloop_roi=self.inputs.rh_meyersloop_roi,
                v1_roi=self.inputs.rh_v1_roi,
                max_step=float(self.inputs.max_step),
                n_points_new=int(self.inputs.n_points_new),
            )

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