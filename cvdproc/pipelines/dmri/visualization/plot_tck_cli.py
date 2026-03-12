#!/usr/bin/env python3
import os
import argparse
from typing import Tuple, List, Optional

import numpy as np
import nibabel as nib
from PIL import Image

from fury import window, actor


def robust_norm_nonzero(vol: np.ndarray, p_low: float, p_high: float, eps: float = 1e-6) -> Tuple[float, float]:
    v = vol[np.isfinite(vol) & (vol > 0)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    if hi <= lo + eps:
        hi = lo + eps
    return lo, hi


def load_volume(nii_path: str):
    img = nib.load(nii_path)
    vol = img.get_fdata(dtype=np.float32)
    return img, vol


def load_tck_streamlines(tck_path: str):
    tck = nib.streamlines.load(tck_path)
    return tck.streamlines


def filter_streamlines(streamlines):
    valid = []
    for s in streamlines:
        if s is None:
            continue
        if len(s) < 2:
            continue
        if not np.all(np.isfinite(s)):
            continue
        valid.append(s)
    return valid


def sample_streamlines(streamlines, max_streamlines: int, seed: int = 0):
    n = len(streamlines)
    if max_streamlines <= 0 or n <= max_streamlines:
        return streamlines
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_streamlines, replace=False)
    return [streamlines[i] for i in idx]


def ensure_out_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def get_slice_index(shape, plane: str, index: Optional[int]):
    nx, ny, nz = shape
    plane = plane.lower()
    if index is None:
        if plane == "sagittal":
            return nx // 2
        if plane == "coronal":
            return ny // 2
        if plane == "axial":
            return nz // 2
        raise ValueError("slice_plane must be axial/sagittal/coronal")
    return int(index)


def get_camera_params(scene):
    cam = scene.camera()
    return {
        "pos": cam.GetPosition(),
        "focal": cam.GetFocalPoint(),
        "viewup": cam.GetViewUp(),
        "clip": cam.GetClippingRange(),
    }


def set_camera(scene, cam_params):
    cam = scene.camera()
    cam.SetPosition(*cam_params["pos"])
    cam.SetFocalPoint(*cam_params["focal"])
    cam.SetViewUp(*cam_params["viewup"])
    cam.SetClippingRange(*cam_params["clip"])
    scene.reset_clipping_range()


def chroma_to_alpha(img_rgba: Image.Image, chroma_rgb_255=(255, 0, 255), tol: int = 10) -> Image.Image:
    arr = np.array(img_rgba).astype(np.uint8)
    r, g, b = chroma_rgb_255
    dr = np.abs(arr[..., 0].astype(np.int16) - r)
    dg = np.abs(arr[..., 1].astype(np.int16) - g)
    db = np.abs(arr[..., 2].astype(np.int16) - b)
    m = (dr <= tol) & (dg <= tol) & (db <= tol)
    arr[m, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


def parse_args():
    p = argparse.ArgumentParser(description="Render a single slice background and overlay full 3D TCK streamlines.")

    p.add_argument("--ref-nii", required=True, help="Reference NIfTI for slice background and affine.")
    p.add_argument("--tck", required=True, nargs="+", help="One or more .tck files.")
    p.add_argument("--out-png", required=True, help="Output PNG path.")

    p.add_argument("--slice-plane", default="axial", choices=["axial", "sagittal", "coronal"])
    p.add_argument("--slice-index", type=int, default=None)
    p.add_argument("--slice-opacity", type=float, default=0.60)

    p.add_argument("--use-fixed-window", action="store_true")
    p.add_argument("--fixed-vmin", type=float, default=0.0)
    p.add_argument("--fixed-vmax", type=float, default=0.6)
    p.add_argument("--p-low", type=float, default=2.0)
    p.add_argument("--p-high", type=float, default=99.5)

    p.add_argument("--max-streamlines-per-tck", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--streamline-color", type=float, nargs=3, default=[0.10, 0.35, 0.85])
    p.add_argument("--streamline-opacity", type=float, default=0.98)
    p.add_argument("--streamline-linewidth", type=float, default=2.5)

    p.add_argument("--snapshot-width", type=int, default=1600)
    p.add_argument("--snapshot-height", type=int, default=1200)
    p.add_argument("--zoom", type=float, default=1.35)

    p.add_argument("--force-chroma-key", action="store_true")
    p.add_argument("--chroma-bg", type=float, nargs=3, default=[1.0, 0.0, 1.0])
    p.add_argument("--chroma-tol", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    ref_nii = os.path.abspath(args.ref_nii)
    out_png = os.path.abspath(args.out_png)
    tck_paths = [os.path.abspath(p) for p in args.tck]

    if not os.path.exists(ref_nii):
        raise FileNotFoundError(f"Missing ref_nii: {ref_nii}")
    missing = [p for p in tck_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing tck files:\n" + "\n".join(missing))

    ensure_out_dir(out_png)

    ref_img, vol = load_volume(ref_nii)

    if args.use_fixed_window:
        vmin = float(args.fixed_vmin)
        vmax = float(args.fixed_vmax)
        if vmax <= vmin:
            raise ValueError("fixed_vmax must be greater than fixed_vmin")
    else:
        vmin, vmax = robust_norm_nonzero(vol, float(args.p_low), float(args.p_high))

    vol_norm = np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)

    all_streamlines = []
    for i, pth in enumerate(tck_paths):
        sl = load_tck_streamlines(pth)
        sl = filter_streamlines(sl)
        sl = sample_streamlines(sl, max_streamlines=int(args.max_streamlines_per_tck), seed=int(args.seed) + i)
        if len(sl) > 0:
            all_streamlines.extend(sl)

    if len(all_streamlines) == 0:
        raise RuntimeError("No valid streamlines found after filtering/sampling.")

    plane = str(args.slice_plane).lower()
    idx = get_slice_index(vol_norm.shape, plane, args.slice_index)

    size = (int(args.snapshot_width), int(args.snapshot_height))
    tract_color = tuple(float(x) for x in args.streamline_color)
    chroma_bg = tuple(float(x) for x in args.chroma_bg)

    # 1) Reference scene for camera
    ref_scene = window.Scene()
    ref_scene.background((1.0, 1.0, 1.0))

    slicer_ref = actor.slicer(vol_norm, affine=ref_img.affine)
    slicer_ref.opacity(float(args.slice_opacity))
    if plane == "sagittal":
        slicer_ref.display(x=idx)
    elif plane == "coronal":
        slicer_ref.display(y=idx)
    elif plane == "axial":
        slicer_ref.display(z=idx)
    else:
        raise ValueError("slice_plane must be axial/sagittal/coronal")
    ref_scene.add(slicer_ref)

    tract_ref = actor.line(
        all_streamlines,
        colors=tract_color,
        linewidth=float(args.streamline_linewidth),
        opacity=float(args.streamline_opacity),
    )
    ref_scene.add(tract_ref)

    ref_scene.reset_camera()
    ref_scene.zoom(float(args.zoom))
    cam_params = get_camera_params(ref_scene)

    # 2) Slice-only render
    slice_scene = window.Scene()
    slice_scene.background((1.0, 1.0, 1.0))

    slicer = actor.slicer(vol_norm, affine=ref_img.affine)
    slicer.opacity(float(args.slice_opacity))
    if plane == "sagittal":
        slicer.display(x=idx)
    elif plane == "coronal":
        slicer.display(y=idx)
    elif plane == "axial":
        slicer.display(z=idx)
    slice_scene.add(slicer)

    set_camera(slice_scene, cam_params)

    tmp_slice = out_png + ".slice_tmp.png"
    window.snapshot(slice_scene, fname=tmp_slice, size=size, offscreen=True)
    bg_img = Image.open(tmp_slice).convert("RGBA")

    # 3) Tract-only render (transparent -> chroma fallback)
    tract_scene = window.Scene()
    tract_scene.background(chroma_bg)

    tract = actor.line(
        all_streamlines,
        colors=tract_color,
        linewidth=float(args.streamline_linewidth),
        opacity=float(args.streamline_opacity),
    )
    tract_scene.add(tract)

    set_camera(tract_scene, cam_params)

    tmp_tract = out_png + ".tract_tmp.png"

    tract_img = None
    if not bool(args.force_chroma_key):
        try:
            window.snapshot(tract_scene, fname=tmp_tract, size=size, offscreen=True, transparent=True)
            tract_img = Image.open(tmp_tract).convert("RGBA")
        except Exception:
            tract_img = None

    if tract_img is None:
        window.snapshot(tract_scene, fname=tmp_tract, size=size, offscreen=True)
        tract_img = Image.open(tmp_tract).convert("RGBA")
        tract_img = chroma_to_alpha(tract_img, chroma_rgb_255=(255, 0, 255), tol=int(args.chroma_tol))

    out_img = Image.alpha_composite(bg_img, tract_img)
    out_img.save(out_png)

    for pth in [tmp_slice, tmp_tract]:
        try:
            os.remove(pth)
        except Exception:
            pass


if __name__ == "__main__":
    main()