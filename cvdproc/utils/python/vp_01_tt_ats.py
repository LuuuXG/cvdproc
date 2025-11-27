import gzip
import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import splprep, splev


# -------------------------------
# 1. Parse TT file (DSI Studio format)
# -------------------------------
def parse_tt(track_bytes):
    buf1 = np.array(track_bytes, dtype=np.uint8)
    buf2 = buf1.view(np.int8)
    pos = []
    i = 0
    length = len(buf1)

    while i < length:
        pos.append(i)
        size = np.frombuffer(buf1[i:i+4].tobytes(), dtype=np.uint32)[0]
        i += size + 13

    streamlines = []
    for p in pos:
        size = np.frombuffer(buf1[p:p+4].tobytes(), dtype=np.uint32)[0] // 3
        x = np.frombuffer(buf1[p+4:p+8].tobytes(), dtype=np.int32)[0]
        y = np.frombuffer(buf1[p+8:p+12].tobytes(), dtype=np.int32)[0]
        z = np.frombuffer(buf1[p+12:p+16].tobytes(), dtype=np.int32)[0]

        coords = np.zeros((size, 3), dtype=np.float32)
        coords[0] = [x, y, z]

        p_offset = p + 16
        for j in range(1, size):
            x += int(buf2[p_offset])
            y += int(buf2[p_offset + 1])
            z += int(buf2[p_offset + 2])
            coords[j] = [x, y, z]
            p_offset += 3

        streamlines.append(coords / 32.0)
    return streamlines


# -------------------------------
# 2. Interpolate streamlines
# -------------------------------
def trk_interp(streamlines, n_points_new=100, spacing=None, tie_at_center=False):
    """Interpolate all streamlines to have exactly n_points_new points."""
    tracks_interp = [None] * len(streamlines)  # 固定长度列表
    splines = []

    # 1. 为每条 fiber 拟合 spline
    for idx, coords in enumerate(streamlines):
        coords = np.asarray(coords)
        if coords.shape[0] < 4:
            # 太短则复制第一个点填充
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

    # 2. 固定点数重采样
    for idx, tck, total_len in splines:
        u_new = np.linspace(0, total_len, n_points_new)
        interp_coords = np.array(splev(u_new, tck)).T
        tracks_interp[idx] = interp_coords

    # 3. tie-at-center 模式（几何中心对齐）
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

            # 强制补足或裁剪
            if merged.shape[0] < n_points_new:
                pad = np.repeat(merged[-1][None, :], n_points_new - merged.shape[0], axis=0)
                merged = np.vstack([merged, pad])
            elif merged.shape[0] > n_points_new:
                merged = merged[:n_points_new]

            tracks_interp[idx] = merged

    # 4. 最后检查并强制补足
    for i, t in enumerate(tracks_interp):
        if t is None or t.shape[0] < n_points_new:
            if t is None or t.shape[0] == 0:
                t = np.zeros((1, 3))
            pad = np.repeat(t[-1][None, :], n_points_new - t.shape[0], axis=0)
            tracks_interp[i] = np.vstack([t, pad])
        elif t.shape[0] > n_points_new:
            tracks_interp[i] = t[:n_points_new]

    return tracks_interp

# -------------------------------
# 3. Encode back to TT binary
# -------------------------------
def encode_tt(streamlines):
    """
    Encode streamlines into DSI Studio TT binary:
    [uint32: 3*N] + [int32*3: first xyz] + [int8*3*(N-1): deltas]
    Coordinates are expected in voxel units (same as parsed), scaled by 32.
    """
    out = bytearray()
    for coords in streamlines:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.shape[0] < 2:
            continue

        # scale and cast (little-endian)
        coords_i = np.round(coords * 32).astype('<i4', copy=False)     # (N,3) int32 LE
        diffs   = np.diff(coords_i, axis=0).astype(np.int8, copy=False)  # (N-1,3) int8

        # header must be 3*N (NOT bytes)
        N = coords_i.shape[0]
        header = np.array([3 * N], dtype='<u4').tobytes()              # uint32 LE

        # append block: header + first xyz + deltas
        out += header
        out += coords_i[0].tobytes()
        out += diffs.tobytes()

    # return as uint8 array for savemat
    return np.frombuffer(bytes(out), dtype=np.uint8)


# -------------------------------
# 4. Main interface
# -------------------------------
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


# -------------------------------
# 5. Example usage (run directly)
# -------------------------------
if __name__ == "__main__":
    input_tt = "/mnt/f/BIDS/demo_BIDS/derivatives/tmp/rh_OR.tt.gz"
    output_tt = "/mnt/f/BIDS/demo_BIDS/derivatives/tmp/rh_OR_interp.tt.gz"

    interpolate_tt(input_tt, output_tt, n_points_new=100, tie_at_center=True)
