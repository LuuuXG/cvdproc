import os

# ---------------------------------------------------------------------
# IMPORTANT: Set env vars BEFORE importing fury/vtk.
# This increases the chance that offscreen rendering works in subprocs.
# ---------------------------------------------------------------------
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")

from typing import Tuple

import numpy as np
import nibabel as nib
from PIL import Image

from fury import window, actor

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    isdefined,
)


def _robust_norm_nonzero(vol: np.ndarray, p_low: float, p_high: float, eps: float = 1e-6) -> Tuple[float, float]:
    v = vol[np.isfinite(vol) & (vol > 0)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    if hi <= lo + eps:
        hi = lo + eps
    return lo, hi


def _load_volume(nii_path: str):
    img = nib.load(nii_path)
    vol = img.get_fdata(dtype=np.float32)
    return img, vol


def _load_tck_streamlines(tck_path: str):
    tck = nib.streamlines.load(tck_path)
    return tck.streamlines


def _filter_streamlines(streamlines):
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


def _sample_streamlines(streamlines, max_streamlines: int, seed: int = 0):
    n = len(streamlines)
    if max_streamlines <= 0 or n <= max_streamlines:
        return streamlines
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_streamlines, replace=False)
    return [streamlines[i] for i in idx]


def _ensure_out_dir(path: str):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def _get_slice_index(shape, plane: str, index):
    nx, ny, nz = shape
    plane = plane.lower()
    if index is None:
        if plane == "sagittal":
            return nx // 2
        if plane == "coronal":
            return ny // 2
        if plane == "axial":
            return nz // 2
        raise ValueError("slice_plane must be 'axial', 'sagittal', or 'coronal'")
    return int(index)


def _get_camera_params(scene):
    cam = scene.camera()
    return {
        "pos": cam.GetPosition(),
        "focal": cam.GetFocalPoint(),
        "viewup": cam.GetViewUp(),
        "clip": cam.GetClippingRange(),
    }


def _set_camera(scene, cam_params):
    cam = scene.camera()
    cam.SetPosition(*cam_params["pos"])
    cam.SetFocalPoint(*cam_params["focal"])
    cam.SetViewUp(*cam_params["viewup"])
    cam.SetClippingRange(*cam_params["clip"])
    scene.reset_clipping_range()


def _chroma_to_alpha(img_rgba: Image.Image, chroma_rgb_255=(255, 0, 255), tol: int = 10) -> Image.Image:
    arr = np.array(img_rgba).astype(np.uint8)
    r, g, b = chroma_rgb_255
    dr = np.abs(arr[..., 0].astype(np.int16) - r)
    dg = np.abs(arr[..., 1].astype(np.int16) - g)
    db = np.abs(arr[..., 2].astype(np.int16) - b)
    m = (dr <= tol) & (dg <= tol) & (db <= tol)
    arr[m, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


def _is_black_png(png_path: str, threshold: int = 2) -> bool:
    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img)
    return int(arr.max()) <= threshold


def _snapshot_with_fallback(scene, fname: str, size, prefer_offscreen: bool = True):
    """
    1) Try prefer_offscreen (usually True).
    2) If output is black, and DISPLAY exists, retry with offscreen=False.
    3) If still black, raise RuntimeError with actionable message.
    """
    display = os.environ.get("DISPLAY", "")
    tried = []

    def _try(offscreen_flag: bool):
        window.snapshot(scene, fname=fname, size=size, offscreen=offscreen_flag)
        tried.append(offscreen_flag)
        return not _is_black_png(fname)

    if prefer_offscreen:
        ok = _try(True)
        if ok:
            return

        if display:
            ok2 = _try(False)
            if ok2:
                return
    else:
        ok = _try(False)
        if ok:
            return
        ok2 = _try(True)
        if ok2:
            return

    raise RuntimeError(
        "FURY/VTK snapshot produced a black image. "
        f"Tried offscreen={tried}. "
        "This usually means no usable OpenGL context in the Nipype subprocess. "
        "Recommended fix: run the workflow with a virtual display, e.g. `xvfb-run -a ...`."
    )


def _snapshot_transparent_with_fallback(scene, fname: str, size, force_chroma: bool, chroma_tol: int):
    """
    Try transparent snapshot first (offscreen=True, transparent=True). If it fails or produces black,
    fall back to chroma-key mode (magenta background) and remove chroma in PIL.
    """
    if not force_chroma:
        try:
            window.snapshot(scene, fname=fname, size=size, offscreen=True, transparent=True)
            if not _is_black_png(fname):
                return Image.open(fname).convert("RGBA")
        except Exception:
            pass

    # Chroma-key fallback (use snapshot fallback to avoid black in subproc)
    _snapshot_with_fallback(scene, fname, size=size, prefer_offscreen=True)
    img = Image.open(fname).convert("RGBA")
    img = _chroma_to_alpha(img, chroma_rgb_255=(255, 0, 255), tol=int(chroma_tol))
    return img


class PlotTckOnSliceInputSpec(BaseInterfaceInputSpec):
    ref_nii = File(exists=True, mandatory=True, desc="Reference NIfTI for slice background and affine.")
    tck_files = traits.List(File(exists=True), mandatory=True, desc="List of input .tck files.")
    out_png = File(mandatory=True, desc="Output PNG path.")

    slice_plane = traits.Enum("axial", "sagittal", "coronal", usedefault=True)
    slice_index = traits.Int(desc="Slice index; if undefined, center slice.")
    slice_opacity = traits.Float(0.60, usedefault=True)

    use_fixed_window = traits.Bool(False, usedefault=True)
    fixed_vmin = traits.Float(0.0, usedefault=True)
    fixed_vmax = traits.Float(0.6, usedefault=True)

    p_low = traits.Float(2.0, usedefault=True)
    p_high = traits.Float(99.5, usedefault=True)

    max_streamlines_per_tck = traits.Int(4000, usedefault=True)
    seed = traits.Int(0, usedefault=True)

    streamline_color = traits.Tuple(
        traits.Float, traits.Float, traits.Float,
        value=(0.10, 0.35, 0.85),
        usedefault=True,
    )
    streamline_opacity = traits.Float(0.98, usedefault=True)
    streamline_linewidth = traits.Float(2.5, usedefault=True)

    snapshot_width = traits.Int(1600, usedefault=True)
    snapshot_height = traits.Int(1200, usedefault=True)
    zoom = traits.Float(1.35, usedefault=True)

    chroma_bg = traits.Tuple(
        traits.Float, traits.Float, traits.Float,
        value=(1.0, 0.0, 1.0),
        usedefault=True,
    )
    chroma_tol = traits.Int(10, usedefault=True)

    force_chroma_key = traits.Bool(False, usedefault=True)
    prefer_offscreen = traits.Bool(
        True,
        usedefault=True,
        desc="Prefer offscreen snapshot. If black and DISPLAY exists, fallback to offscreen=False.",
    )


class PlotTckOnSliceOutputSpec(TraitedSpec):
    out_png = File(exists=True, desc="Rendered PNG.")


class PlotTckOnSlice(BaseInterface):
    input_spec = PlotTckOnSliceInputSpec
    output_spec = PlotTckOnSliceOutputSpec

    def _run_interface(self, runtime):
        ref_nii = os.path.abspath(self.inputs.ref_nii)
        out_png = os.path.abspath(self.inputs.out_png)

        tck_paths = [os.path.abspath(p) for p in self.inputs.tck_files]
        for p in tck_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing TCK file: {p}")

        _ensure_out_dir(out_png)

        ref_img, vol = _load_volume(ref_nii)

        if bool(self.inputs.use_fixed_window):
            vmin = float(self.inputs.fixed_vmin)
            vmax = float(self.inputs.fixed_vmax)
            if vmax <= vmin:
                raise ValueError("fixed_vmax must be greater than fixed_vmin")
        else:
            vmin, vmax = _robust_norm_nonzero(vol, float(self.inputs.p_low), float(self.inputs.p_high))

        vol_norm = np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)

        all_streamlines = []
        for i, p in enumerate(tck_paths):
            sl = _load_tck_streamlines(p)
            sl = _filter_streamlines(sl)
            sl = _sample_streamlines(sl, int(self.inputs.max_streamlines_per_tck), int(self.inputs.seed) + i)
            if len(sl) > 0:
                all_streamlines.extend(sl)

        if len(all_streamlines) == 0:
            raise RuntimeError("No valid streamlines found across all TCK files after filtering/sampling.")

        plane = str(self.inputs.slice_plane).lower()
        idx = _get_slice_index(
            vol_norm.shape,
            plane,
            self.inputs.slice_index if isdefined(self.inputs.slice_index) else None,
        )

        size = (int(self.inputs.snapshot_width), int(self.inputs.snapshot_height))
        tract_color = tuple(float(x) for x in self.inputs.streamline_color)
        chroma_bg = tuple(float(x) for x in self.inputs.chroma_bg)

        # 1) Reference scene -> camera
        ref_scene = window.Scene()
        ref_scene.background((1.0, 1.0, 1.0))

        slicer_ref = actor.slicer(vol_norm, affine=ref_img.affine)
        slicer_ref.opacity(float(self.inputs.slice_opacity))
        if plane == "sagittal":
            slicer_ref.display(x=idx)
        elif plane == "coronal":
            slicer_ref.display(y=idx)
        elif plane == "axial":
            slicer_ref.display(z=idx)
        else:
            raise ValueError("slice_plane must be 'axial', 'sagittal', or 'coronal'")
        ref_scene.add(slicer_ref)

        tract_ref = actor.line(
            all_streamlines,
            colors=tract_color,
            linewidth=float(self.inputs.streamline_linewidth),
            opacity=float(self.inputs.streamline_opacity),
        )
        ref_scene.add(tract_ref)

        ref_scene.reset_camera()
        ref_scene.zoom(float(self.inputs.zoom))
        cam_params = _get_camera_params(ref_scene)

        # 2) Slice-only render
        slice_scene = window.Scene()
        slice_scene.background((1.0, 1.0, 1.0))

        slicer = actor.slicer(vol_norm, affine=ref_img.affine)
        slicer.opacity(float(self.inputs.slice_opacity))
        if plane == "sagittal":
            slicer.display(x=idx)
        elif plane == "coronal":
            slicer.display(y=idx)
        elif plane == "axial":
            slicer.display(z=idx)
        slice_scene.add(slicer)

        _set_camera(slice_scene, cam_params)

        tmp_slice = out_png + ".slice_tmp.png"
        _snapshot_with_fallback(
            slice_scene,
            fname=tmp_slice,
            size=size,
            prefer_offscreen=bool(self.inputs.prefer_offscreen),
        )
        bg_img = Image.open(tmp_slice).convert("RGBA")

        # 3) Tract-only render (transparent -> chroma fallback)
        tract_scene = window.Scene()
        tract_scene.background(chroma_bg)

        tract = actor.line(
            all_streamlines,
            colors=tract_color,
            linewidth=float(self.inputs.streamline_linewidth),
            opacity=float(self.inputs.streamline_opacity),
        )
        tract_scene.add(tract)

        _set_camera(tract_scene, cam_params)

        tmp_tract = out_png + ".tract_tmp.png"
        tract_img = _snapshot_transparent_with_fallback(
            tract_scene,
            fname=tmp_tract,
            size=size,
            force_chroma=bool(self.inputs.force_chroma_key),
            chroma_tol=int(self.inputs.chroma_tol),
        )

        # 4) Composite
        out_img = Image.alpha_composite(bg_img, tract_img)
        out_img.save(out_png)

        # Cleanup
        for p in [tmp_slice, tmp_tract]:
            try:
                os.remove(p)
            except Exception:
                pass

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_png"] = os.path.abspath(self.inputs.out_png)
        return outputs


if __name__ == "__main__":
    qc = PlotTckOnSlice()
    qc.inputs.ref_nii = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/dtifit/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_space-ACPC_model-tensor_param-fa_dwimap.nii.gz"
    qc.inputs.tck_files = [
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OR_streamlines.tck",
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_hemi-R_space-ACPC_bundle-OR_streamlines.tck",
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_hemi-L_space-ACPC_bundle-OT_streamlines.tck",
        "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/sub-HC0068_ses-baseline_acq-DSIb4000_dir-AP_hemi-R_space-ACPC_bundle-OT_streamlines.tck",
    ]
    qc.inputs.out_png = "/mnt/f/BIDS/WCH_AF_Project/derivatives/dwi_pipeline/sub-HC0068/ses-baseline/visual_pathway_analysis/QC_interface.png"

    # If you have DISPLAY in your environment, this often fixes black frames in Nipype subprocs:
    # qc.inputs.prefer_offscreen = False

    qc.run()