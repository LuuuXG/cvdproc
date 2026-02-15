#!/usr/bin/env python3

import os
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def find_best_slice(mask_3d: np.ndarray, axis: int = 2) -> int:
    counts = []
    for i in range(mask_3d.shape[axis]):
        if axis == 0:
            sl = mask_3d[i, :, :]
        elif axis == 1:
            sl = mask_3d[:, i, :]
        else:
            sl = mask_3d[:, :, i]
        counts.append(int(sl.sum()))
    return int(np.argmax(counts))


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def upsample_and_smooth(mask_2d: np.ndarray, factor: int = 4, sigma: float = 1.0) -> np.ndarray:
    """
    Upsample + smooth a binary mask to obtain a soft field for smooth contours.
    Returns float in [0, 1].
    """
    m = mask_2d.astype(np.float32)

    try:
        from scipy.ndimage import zoom, gaussian_filter  # type: ignore
        m_up = zoom(m, zoom=factor, order=1)
        if sigma > 0:
            m_up = gaussian_filter(m_up, sigma=sigma)
        return np.clip(m_up, 0.0, 1.0)
    except Exception:
        if factor > 1:
            m_up = np.repeat(np.repeat(m, factor, axis=0), factor, axis=1)
        else:
            m_up = m
        return np.clip(m_up, 0.0, 1.0)


def fill_holes_if_possible(mask_2d: np.ndarray) -> np.ndarray:
    """
    Fill holes to avoid drawing inner contours. If scipy is unavailable, return input unchanged.
    """
    try:
        from scipy.ndimage import binary_fill_holes  # type: ignore
        return binary_fill_holes(mask_2d.astype(bool)).astype(bool)
    except Exception:
        return mask_2d.astype(bool)


def upsample_mask_nn(mask_2d: np.ndarray, factor: int) -> np.ndarray:
    if factor > 1:
        return np.repeat(np.repeat(mask_2d, factor, axis=0), factor, axis=1)
    return mask_2d


def plot_slice_multicolor(
    seg_path: str,
    out_png: str,
    axis: int = 2,
    slice_index: Optional[int] = None,
    upsample_factor: int = 4,
    smooth_sigma: float = 1.0,
    radiological_flip: bool = False,
    add_title: bool = False,
    edge_lw: float = 1.6,
    edge_color: Tuple[float, float, float] = (0.10, 0.10, 0.10),
) -> int:
    """
    Label rules:
      - label == 24: CSF -> transparent, no edge
      - label in {2, 41}: white matter -> white
      - label in {4, 43}: lateral ventricles -> one color
      - all other labels > 0 (excluding above): gray
      - label == 0: background -> transparent
    """

    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    img = nib.load(seg_path)
    data = img.get_fdata().astype(np.int32)

    # Auto-pick slice based on non-CSF tissue
    tissue_3d = (data > 0) & (data != 24)
    if slice_index is None:
        slice_index = find_best_slice(tissue_3d, axis=axis)

    # Extract slice
    if axis == 0:
        sl = data[slice_index, :, :]
    elif axis == 1:
        sl = data[:, slice_index, :]
    elif axis == 2:
        sl = data[:, :, slice_index]
    else:
        raise ValueError("axis must be 0, 1, or 2")

    if radiological_flip:
        sl = np.fliplr(sl)

    # Category masks (2D, original resolution)
    mask_csf = (sl == 24)
    mask_wm = (sl == 2) | (sl == 41)
    mask_vent = (sl == 4) | (sl == 43)
    mask_gm = (sl > 0) & (~mask_csf) & (~mask_wm) & (~mask_vent)

    # Tissue mask (non-CSF, non-background)
    mask_tissue = (sl > 0) & (~mask_csf)

    # Edge mask: fill holes so that CSF does not create inner contours
    mask_edge = fill_holes_if_possible(mask_tissue)

    # Soft field for edge contour
    soft_edge = upsample_and_smooth(mask_edge, factor=upsample_factor, sigma=smooth_sigma)

    # Upsample masks for fill (nearest neighbor)
    tissue_up = upsample_mask_nn(mask_tissue, upsample_factor)
    wm_up = upsample_mask_nn(mask_wm, upsample_factor)
    vent_up = upsample_mask_nn(mask_vent, upsample_factor)
    gm_up = upsample_mask_nn(mask_gm, upsample_factor)

    # Colors (tweak as needed)
    COLOR_WM = (1.0, 1.0, 1.0)        # white
    COLOR_VENT = (0.92, 1.0, 1.0)   # light blue
    COLOR_GM = (0.70, 0.70, 0.70)     # gray

    # Initialize canvas: white RGB, fully transparent alpha
    h, w = soft_edge.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = 1.0
    rgba[..., 3] = 0.0

    # Only allow drawing where tissue exists (prevents "black holes")
    inside = (soft_edge >= 0.5) & tissue_up

    rgba[wm_up & inside, :3] = COLOR_WM
    rgba[vent_up & inside, :3] = COLOR_VENT
    rgba[gm_up & inside, :3] = COLOR_GM
    rgba[inside, 3] = 0.95

    # Plot
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = plt.gca()
    ax.imshow(rgba, origin="lower", interpolation="bilinear")

    # Smooth outer contour (CSF holes suppressed via fill_holes)
    ax.contour(
        soft_edge,
        levels=[0.5],
        colors=[edge_color],
        linewidths=edge_lw,
        origin="lower",
        antialiased=True,
    )

    ax.set_axis_off()
    if add_title:
        ax.set_title(f"Slice axis={axis}, index={slice_index}", fontsize=10)

    ensure_parent_dir(out_png)
    plt.tight_layout(pad=0.0)
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    return int(slice_index)


def main() -> None:
    seg_path = "/mnt/e/Neuroimage/workdir/seg.nii.gz"
    out_png = "/mnt/e/Neuroimage/workdir/seg_slice_multicolor_clean.png"

    axis = 2  # 0=sagittal, 1=coronal, 2=axial
    slice_index = 184  # auto-pick

    used = plot_slice_multicolor(
        seg_path=seg_path,
        out_png=out_png,
        axis=axis,
        slice_index=slice_index,
        upsample_factor=4,
        smooth_sigma=1.0,
        radiological_flip=False,
        add_title=False,
        edge_lw=1.6,
        edge_color=(0.10, 0.10, 0.10),
    )

    print(f"Saved: {out_png}")
    print(f"Slice index used: {used} (axis={axis})")


if __name__ == "__main__":
    main()
