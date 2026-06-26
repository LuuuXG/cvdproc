import os
import gzip
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy.io import loadmat, savemat


@dataclass
class TinyTrackData:
    streamlines: List[np.ndarray]
    dimension: np.ndarray
    voxel_size: np.ndarray
    metadata: Dict[str, Any]


def _as_dim3(dimension: np.ndarray) -> np.ndarray:
    dim = np.asarray(dimension).astype(np.int32).reshape(-1)
    if dim.size < 3:
        raise ValueError(f"Invalid TT dimension: {dimension}")
    return dim[:3]


def parse_tt(track_bytes: np.ndarray) -> List[np.ndarray]:
    """Parse a DSI Studio TinyTrack byte stream into streamline coordinates."""
    buf1 = np.array(track_bytes, dtype=np.uint8)
    buf2 = buf1.view(np.int8)

    positions = []
    i = 0
    total_len = len(buf1)

    while i < total_len:
        positions.append(i)
        size = np.frombuffer(buf1[i:i + 4].tobytes(), dtype=np.uint32)[0]
        i += int(size) + 13

    streamlines = []

    for p in positions:
        n_points = np.frombuffer(buf1[p:p + 4].tobytes(), dtype=np.uint32)[0] // 3

        x = np.frombuffer(buf1[p + 4:p + 8].tobytes(), dtype=np.int32)[0]
        y = np.frombuffer(buf1[p + 8:p + 12].tobytes(), dtype=np.int32)[0]
        z = np.frombuffer(buf1[p + 12:p + 16].tobytes(), dtype=np.int32)[0]

        coords = np.zeros((n_points, 3), dtype=np.float32)
        coords[0] = [x, y, z]

        q = p + 16
        for j in range(1, n_points):
            x += int(buf2[q])
            y += int(buf2[q + 1])
            z += int(buf2[q + 2])
            q += 3
            coords[j] = [x, y, z]

        streamlines.append(coords / 32.0)

    return streamlines


def encode_tt(streamlines: List[np.ndarray]) -> np.ndarray:
    """Encode streamline coordinates into a DSI Studio TinyTrack byte stream."""
    out = bytearray()

    for streamline in streamlines:
        s = np.asarray(streamline, dtype=np.float32)

        if s.ndim != 2 or s.shape[0] < 2 or s.shape[1] != 3:
            continue

        s_i = np.round(s * 32).astype("<i4", copy=False)
        diff = np.diff(s_i, axis=0)

        if np.any(diff < -128) or np.any(diff > 127):
            raise ValueError(
                "A streamline contains a step outside int8 range after scaling. "
                "TinyTrack cannot encode this streamline without resampling."
            )

        diff = diff.astype(np.int8, copy=False)
        n_points = s_i.shape[0]

        out += np.array([3 * n_points], dtype="<u4").tobytes()
        out += s_i[0].tobytes()
        out += diff.tobytes()

    return np.frombuffer(bytes(out), dtype=np.uint8)


def flip_streamlines_axes(
    streamlines: List[np.ndarray],
    dimension: np.ndarray,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    flip_z_axis: bool = False,
) -> List[np.ndarray]:
    """
    Flip streamline voxel coordinates along selected axes.

    This is for voxel-index convention conversion, for example:
    coord_new = dim - 1 - coord_old

    Do not use this function for RAS/LPS world-coordinate conversion.
    """
    dim = _as_dim3(dimension)
    flipped = []

    for streamline in streamlines:
        s = np.asarray(streamline, dtype=np.float32).copy()

        if s.ndim != 2 or s.shape[1] != 3:
            raise ValueError(f"Invalid streamline shape: {s.shape}")

        if flip_x_axis:
            s[:, 0] = (dim[0] - 1) - s[:, 0]

        if flip_y_axis:
            s[:, 1] = (dim[1] - 1) - s[:, 1]

        if flip_z_axis:
            s[:, 2] = (dim[2] - 1) - s[:, 2]

        flipped.append(s)

    return flipped


def ras_to_lps_world_streamlines(streamlines: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert RAS world coordinates to LPS world coordinates.

    This operation is symmetric, so the same function can be used for LPS to RAS.
    """
    converted = []

    for streamline in streamlines:
        s = np.asarray(streamline, dtype=np.float32).copy()

        if s.ndim != 2 or s.shape[1] != 3:
            raise ValueError(f"Invalid streamline shape: {s.shape}")

        s[:, 0] *= -1
        s[:, 1] *= -1

        converted.append(s)

    return converted


def lps_to_ras_world_streamlines(streamlines: List[np.ndarray]) -> List[np.ndarray]:
    """Convert LPS world coordinates to RAS world coordinates."""
    return ras_to_lps_world_streamlines(streamlines)


def load_tt(
    tt_file: str,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    flip_z_axis: bool = False,
    preserve_metadata: bool = True,
) -> TinyTrackData:
    """
    Load a DSI Studio TinyTrack file.

    Parameters
    ----------
    tt_file : str
        Path to a .tt.gz file.
    flip_x_axis, flip_y_axis, flip_z_axis : bool
        Whether to flip voxel coordinates after loading.
        Use these options to convert TT native voxel coordinates into your internal
        NIfTI-like voxel coordinates.
    preserve_metadata : bool
        If True, preserve non-track fields from the original MATLAB file.

    Returns
    -------
    TinyTrackData
        Loaded streamline data.
    """
    with gzip.open(tt_file, "rb") as f:
        mat = loadmat(f, squeeze_me=True, struct_as_record=False)

    if "track" not in mat:
        raise KeyError(f"No 'track' field found in TT file: {tt_file}")

    dimension = mat.get("dimension", np.array([1, 1, 1], dtype=np.int32))
    voxel_size = mat.get("voxel_size", np.array([1, 1, 1], dtype=np.float32))

    streamlines = parse_tt(mat["track"])

    if flip_x_axis or flip_y_axis or flip_z_axis:
        streamlines = flip_streamlines_axes(
            streamlines,
            dimension,
            flip_x_axis=flip_x_axis,
            flip_y_axis=flip_y_axis,
            flip_z_axis=flip_z_axis,
        )

    metadata = {}
    if preserve_metadata:
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            if key == "track":
                continue
            metadata[key] = value

    metadata["dimension"] = dimension
    metadata["voxel_size"] = voxel_size

    return TinyTrackData(
        streamlines=streamlines,
        dimension=np.asarray(dimension),
        voxel_size=np.asarray(voxel_size),
        metadata=metadata,
    )


def save_tt(
    streamlines: List[np.ndarray],
    output_file: str,
    dimension: np.ndarray,
    voxel_size: np.ndarray,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    flip_z_axis: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save streamlines as a DSI Studio TinyTrack file.

    Parameters
    ----------
    streamlines : list of ndarray
        Streamlines in your internal coordinate convention.
    output_file : str
        Output .tt.gz path.
    dimension : ndarray
        Image dimension used by the TT file.
    voxel_size : ndarray
        Voxel size used by the TT file.
    flip_x_axis, flip_y_axis, flip_z_axis : bool
        Whether to flip voxel coordinates before saving.
        Use the same flags that were used during loading to convert internal
        coordinates back to TT native voxel coordinates.
    metadata : dict, optional
        Extra metadata fields to write into the MATLAB file.

    Returns
    -------
    str
        Output file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    streamlines_to_save = streamlines

    if flip_x_axis or flip_y_axis or flip_z_axis:
        streamlines_to_save = flip_streamlines_axes(
            streamlines_to_save,
            dimension,
            flip_x_axis=flip_x_axis,
            flip_y_axis=flip_y_axis,
            flip_z_axis=flip_z_axis,
        )

    encoded = encode_tt(streamlines_to_save)

    out = {}
    if metadata:
        for key, value in metadata.items():
            if key.startswith("__"):
                continue
            if key == "track":
                continue
            out[key] = value

    out["dimension"] = np.asarray(dimension)
    out["voxel_size"] = np.asarray(voxel_size)
    out["track"] = encoded

    with gzip.open(output_file, "wb") as f:
        savemat(f, out, format="4")

    return output_file

def plot_streamlines_pyvista_colored(
    streamlines,
    max_streamlines=300,
    every_n_point=5,
    max_total_points=50000,
    line_width=3,
    show_axes=True,
    show_bounds=True,
    title="Streamline Visualization",
    background="black",
    screenshot=None,
):
    """
    Fast colored streamline visualization using PyVista.

    Color is assigned by local streamline direction:
    - R-L / X direction: red
    - A-P / Y direction: green
    - S-I / Z direction: blue
    """
    import os
    import numpy as np
    import pyvista as pv

    total_streamlines = len(streamlines)

    valid = []
    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        if s.ndim == 2 and s.shape[1] == 3 and s.shape[0] >= 2:
            valid.append(s)

    if len(valid) == 0:
        raise ValueError("No valid streamlines found.")

    if len(valid) > max_streamlines:
        idx = np.linspace(0, len(valid) - 1, max_streamlines).astype(int)
        valid = [valid[i] for i in idx]

    selected = []
    total_points = 0

    for s in valid:
        s2 = s[::every_n_point, :]
        if s2.shape[0] < 2:
            continue

        if total_points + s2.shape[0] > max_total_points:
            break

        selected.append(s2)
        total_points += s2.shape[0]

    if len(selected) == 0:
        raise ValueError("No streamlines left after downsampling.")

    points_list = []
    lines_list = []
    color_list = []
    point_offset = 0

    for s in selected:
        n = s.shape[0]

        direction = np.zeros_like(s, dtype=np.float32)
        direction[1:-1] = s[2:] - s[:-2]
        direction[0] = s[1] - s[0]
        direction[-1] = s[-1] - s[-2]

        color = np.abs(direction)
        norm = np.linalg.norm(color, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        color = color / norm

        points_list.append(s)

        line = np.empty(n + 1, dtype=np.int64)
        line[0] = n
        line[1:] = np.arange(point_offset, point_offset + n)
        lines_list.append(line)

        color_list.append(color)

        point_offset += n

    points = np.vstack(points_list)
    lines = np.concatenate(lines_list)
    colors = np.vstack(color_list)

    poly = pv.PolyData()
    poly.points = points
    poly.lines = lines
    poly["direction_rgb"] = colors

    plotter = pv.Plotter()
    plotter.set_background(background)

    plotter.add_mesh(
        poly,
        scalars="direction_rgb",
        rgb=True,
        line_width=line_width,
        render_lines_as_tubes=False,
    )

    text = (
        f"{title}\n"
        f"Total streamlines: {total_streamlines}\n"
        f"Displayed streamlines: {len(selected)}\n"
        f"Displayed points: {points.shape[0]}"
    )

    plotter.add_text(
        text,
        position="upper_left",
        font_size=10,
        color="white" if background == "black" else "black",
    )

    if show_axes:
        plotter.add_axes(
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            line_width=3,
            labels_off=False,
        )

    if show_bounds:
        plotter.show_bounds(
            grid="front",
            location="outer",
            all_edges=True,
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            color="white" if background == "black" else "black",
        )

    plotter.view_isometric()

    if screenshot is not None:
        os.makedirs(os.path.dirname(os.path.abspath(screenshot)), exist_ok=True)
        plotter.show(screenshot=screenshot)
    else:
        plotter.show()

    return poly

class TinyTrackIO:
    """
    Reusable TinyTrack reader/writer.

    Recommended use:
    - Set flip_x_axis, flip_y_axis, and flip_z_axis once.
    - Load TT into an internal consistent voxel-coordinate convention.
    - Process streamlines in other functions.
    - Save TT using the same object to convert back to TT native convention.
    """

    def __init__(
        self,
        flip_x_axis: bool = False,
        flip_y_axis: bool = False,
        flip_z_axis: bool = False,
    ):
        self.flip_x_axis = bool(flip_x_axis)
        self.flip_y_axis = bool(flip_y_axis)
        self.flip_z_axis = bool(flip_z_axis)

    def load(self, tt_file: str, preserve_metadata: bool = True) -> TinyTrackData:
        return load_tt(
            tt_file=tt_file,
            flip_x_axis=self.flip_x_axis,
            flip_y_axis=self.flip_y_axis,
            flip_z_axis=self.flip_z_axis,
            preserve_metadata=preserve_metadata,
        )

    def save(
        self,
        streamlines: List[np.ndarray],
        output_file: str,
        dimension: np.ndarray,
        voxel_size: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return save_tt(
            streamlines=streamlines,
            output_file=output_file,
            dimension=dimension,
            voxel_size=voxel_size,
            flip_x_axis=self.flip_x_axis,
            flip_y_axis=self.flip_y_axis,
            flip_z_axis=self.flip_z_axis,
            metadata=metadata,
        )


if __name__ == "__main__":
    input_tt = r"/path/to/input.tt.gz"
    output_tt = r"/path/to/output.tt.gz"

    ttio = TinyTrackIO(
        flip_x_axis=False,
        flip_y_axis=True,
        flip_z_axis=False,
    )

    data = ttio.load(input_tt)

    ttio.save(
        streamlines=data.streamlines,
        output_file=output_tt,
        dimension=data.dimension,
        voxel_size=data.voxel_size,
        metadata=data.metadata,
    )