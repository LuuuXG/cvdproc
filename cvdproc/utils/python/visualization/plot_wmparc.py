import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    from nilearn.image import resample_to_img
except Exception:
    resample_to_img = None


# =========================
# Parameters
# =========================
bg_file = "/mnt/e/Codes/cvdproc/cvdproc/data/standard/MNI152/FSL_HCP1065_MD_1mm.nii.gz"
parc_file = "/mnt/e/Codes/cvdproc/cvdproc/data/standard/MNI152/wmparc.nii.gz"
xlsx_file = "/mnt/e/Codes/cvdproc/cvdproc/utils/python/visualization/wmparc.xlsx"
out_dir = "/mnt/e/Neuroimage/workdir"

roi_col = 0
name_col = 1
value_col = 2
stat_col = 3
has_header = False

sheet_names = None
save_nii = False

stat_thr = 2.0
alpha_sig = 1.0
alpha_nonsig = 0.25
cmap_name = "coolwarm"
dpi = 300

colorbar_vmin = -0.35
colorbar_vmax = 0.35

bg_mode = "manual"
bg_vmin = 0.0005
bg_vmax = 0.0015
bg_pmin = 2
bg_pmax = 99.5
bg_gamma = 0.7

display_left_is_left = True
transparent_output = True
outside_brain_color = "white"

axial_slices = [65, 75, 85, 95, 105, 115, 125]
slice_overlap = 0.275
slice_zoom = 1.00
slice_pad = 6

show_slice_labels = False
show_colorbar = True
draw_slice_edge = False
slice_edge_width = 0.8

output_prefix = "wmparc_overlay_axial_mosaic"


def sanitize_name(name):
    name = str(name)
    name = re.sub(r"[\\/:*?\"<>| ]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "sheet"


def get_bg_limits(bg_data):
    vals = bg_data[np.isfinite(bg_data) & (bg_data > 0)]
    if vals.size == 0:
        return 0, 1

    if bg_mode == "manual":
        if bg_vmin is None or bg_vmax is None:
            raise ValueError("bg_vmin and bg_vmax must be defined when bg_mode is manual.")
        return bg_vmin, bg_vmax

    if bg_mode == "percentile":
        return np.percentile(vals, bg_pmin), np.percentile(vals, bg_pmax)

    raise ValueError("bg_mode must be manual or percentile.")


def crop_to_brain(bg_sl, val_sl, sig_sl, pad=5):
    mask = np.isfinite(bg_sl) & (bg_sl > 0)
    if not np.any(mask):
        return bg_sl, val_sl, sig_sl

    rows, cols = np.where(mask)
    r0 = max(rows.min() - pad, 0)
    r1 = min(rows.max() + pad + 1, bg_sl.shape[0])
    c0 = max(cols.min() - pad, 0)
    c1 = min(cols.max() + pad + 1, bg_sl.shape[1])

    return bg_sl[r0:r1, c0:c1], val_sl[r0:r1, c0:c1], sig_sl[r0:r1, c0:c1]


def resize_float_to_canvas(img, canvas_shape, fill_value=np.nan):
    out = np.full(canvas_shape, fill_value, dtype=np.float32)
    h, w = img.shape
    ch, cw = canvas_shape
    h2 = min(h, ch)
    w2 = min(w, cw)

    dst_r0 = (ch - h2) // 2
    dst_c0 = (cw - w2) // 2
    src_r0 = max((h - h2) // 2, 0)
    src_c0 = max((w - w2) // 2, 0)

    out[dst_r0:dst_r0 + h2, dst_c0:dst_c0 + w2] = img[src_r0:src_r0 + h2, src_c0:src_c0 + w2]
    return out


def resize_uint_to_canvas(img, canvas_shape, fill_value=0):
    out = np.full(canvas_shape, fill_value, dtype=img.dtype)
    h, w = img.shape
    ch, cw = canvas_shape
    h2 = min(h, ch)
    w2 = min(w, cw)

    dst_r0 = (ch - h2) // 2
    dst_c0 = (cw - w2) // 2
    src_r0 = max((h - h2) // 2, 0)
    src_c0 = max((w - w2) // 2, 0)

    out[dst_r0:dst_r0 + h2, dst_c0:dst_c0 + w2] = img[src_r0:src_r0 + h2, src_c0:src_c0 + w2]
    return out


def hex_to_rgb01(color):
    if color == "white":
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if color == "black":
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    raise ValueError("Only white and black are supported for outside_brain_color.")


def make_slice_rgba(bg_sl, val_sl, sig_sl, bg_vmin_current, bg_vmax_current, cmap, norm):
    brain_mask = np.isfinite(bg_sl) & (bg_sl > 0)

    denom = bg_vmax_current - bg_vmin_current
    if denom == 0:
        denom = 1

    bg_norm = np.clip((bg_sl - bg_vmin_current) / denom, 0, 1)

    if bg_gamma is not None:
        bg_norm = np.power(bg_norm, bg_gamma)

    bg_rgb = np.stack([bg_norm, bg_norm, bg_norm], axis=-1)

    out = np.zeros((bg_sl.shape[0], bg_sl.shape[1], 4), dtype=np.float32)
    out[..., :3] = bg_rgb
    out[..., 3] = brain_mask.astype(np.float32)

    val_mask = (val_sl != 0) & brain_mask
    if np.any(val_mask):
        overlay = cmap(norm(val_sl))[..., :3]
        alpha = np.zeros(val_sl.shape, dtype=np.float32)
        alpha[val_mask & (sig_sl == 1)] = alpha_sig
        alpha[val_mask & (sig_sl == 0)] = alpha_nonsig

        a = alpha[..., None]
        out[..., :3] = overlay * a + out[..., :3] * (1.0 - a)

    return out


def paste_rgba(canvas, patch, x0):
    h, w, _ = patch.shape
    target = canvas[:, x0:x0 + w, :]
    mask = patch[..., 3] > 0

    target[mask, :3] = patch[mask, :3]
    target[mask, 3] = patch[mask, 3]
    canvas[:, x0:x0 + w, :] = target
    return canvas


def plot_axial_overlap_mosaic(bg_data, value_data, sig_data, output_file):
    bg_vmin_current, bg_vmax_current = get_bg_limits(bg_data)

    finite_vals = value_data[np.isfinite(value_data) & (value_data != 0)]
    if finite_vals.size == 0:
        raise ValueError("No ROI values were mapped to the parcellation image.")

    if colorbar_vmin is not None and colorbar_vmax is not None:
        norm = Normalize(vmin=colorbar_vmin, vmax=colorbar_vmax)
    else:
        vmax = np.max(np.abs(finite_vals))
        norm = Normalize(vmin=-vmax, vmax=vmax)

    cmap = plt.get_cmap(cmap_name)

    slices = []
    max_h = 0
    max_w = 0

    for idx in axial_slices:
        if idx < 0 or idx >= bg_data.shape[2]:
            continue

        bg_sl = np.rot90(bg_data[:, :, idx])
        val_sl = np.rot90(value_data[:, :, idx])
        sig_sl = np.rot90(sig_data[:, :, idx])

        if display_left_is_left:
            bg_sl = np.fliplr(bg_sl)
            val_sl = np.fliplr(val_sl)
            sig_sl = np.fliplr(sig_sl)

        bg_sl = bg_sl.copy()
        bg_sl[bg_sl <= 0] = np.nan
        bg_sl, val_sl, sig_sl = crop_to_brain(bg_sl, val_sl, sig_sl, pad=slice_pad)

        if slice_zoom != 1.0:
            try:
                from scipy.ndimage import zoom
                bg_sl = zoom(bg_sl, slice_zoom, order=1)
                val_sl = zoom(val_sl, slice_zoom, order=0)
                sig_sl = zoom(sig_sl, slice_zoom, order=0)
            except Exception:
                pass

        max_h = max(max_h, bg_sl.shape[0])
        max_w = max(max_w, bg_sl.shape[1])
        slices.append((idx, bg_sl, val_sl, sig_sl))

    if len(slices) == 0:
        raise ValueError("No valid axial slices were selected.")

    step = max(int(max_w * (1.0 - slice_overlap)), 1)
    canvas_h = max_h
    canvas_w = max_w + step * (len(slices) - 1)

    base_rgb = hex_to_rgb01(outside_brain_color)
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
    canvas[..., :3] = base_rgb
    canvas[..., 3] = 0.0 if transparent_output else 1.0

    for i, (_, bg_sl, val_sl, sig_sl) in enumerate(slices):
        x0 = i * step
        bg_patch = resize_float_to_canvas(bg_sl.astype(np.float32), (max_h, max_w), fill_value=np.nan)
        val_patch = resize_float_to_canvas(val_sl.astype(np.float32), (max_h, max_w), fill_value=0)
        sig_patch = resize_uint_to_canvas(sig_sl.astype(np.uint8), (max_h, max_w), fill_value=0)
        patch_rgba = make_slice_rgba(bg_patch, val_patch, sig_patch, bg_vmin_current, bg_vmax_current, cmap, norm)
        canvas = paste_rgba(canvas, patch_rgba, x0)

    fig_w = max(6, canvas_w / 70)
    fig_h = max(2.5, canvas_h / 70)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(outside_brain_color)
    ax.set_facecolor(outside_brain_color)

    ax.imshow(canvas)
    ax.axis("off")

    if show_slice_labels:
        for i, (idx, _, _, _) in enumerate(slices):
            x = i * step + max_w / 2
            ax.text(x, canvas_h - 4, str(idx), ha="center", va="bottom", fontsize=7)

    if draw_slice_edge:
        for i in range(len(slices)):
            x0 = i * step
            rect = plt.Rectangle((x0, 0), max_w, max_h, fill=False, linewidth=slice_edge_width)
            ax.add_patch(rect)

    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01)
        cbar.set_label("ROI value", fontsize=9)

    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", transparent=transparent_output, pad_inches=0.02)
    plt.close(fig)


def build_maps_for_sheet(df, parc_data):
    df = df.iloc[:, [roi_col, name_col, value_col, stat_col]].copy()
    df.columns = ["roi", "name", "value", "stat"]
    df = df.dropna(subset=["roi", "value", "stat"])
    df["roi"] = df["roi"].astype(int)
    df["value"] = df["value"].astype(float)
    df["stat"] = df["stat"].astype(float)

    value_map = np.zeros(parc_data.shape, dtype=np.float32)
    sig_map = np.zeros(parc_data.shape, dtype=np.uint8)

    for _, row in df.iterrows():
        mask = parc_data == int(row["roi"])
        if np.any(mask):
            value_map[mask] = float(row["value"])
            sig_map[mask] = int(abs(float(row["stat"])) >= stat_thr)

    mapped_rois = sorted(set(np.unique(parc_data)).intersection(set(df["roi"])))
    return value_map, sig_map, mapped_rois


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)

    bg_img = nib.load(bg_file)
    parc_img = nib.load(parc_file)

    if bg_img.shape[:3] != parc_img.shape[:3] or not np.allclose(bg_img.affine, parc_img.affine):
        if resample_to_img is None:
            raise RuntimeError("Install nilearn or resample wmparc to the background image first.")
        parc_img = resample_to_img(parc_img, bg_img, interpolation="nearest", force_resample=True, copy_header=True)

    parc_data = np.rint(parc_img.get_fdata()).astype(np.int32)
    bg_data = nib.as_closest_canonical(bg_img).get_fdata()

    xls = pd.ExcelFile(xlsx_file)
    selected_sheets = xls.sheet_names if sheet_names is None else sheet_names
    header = 0 if has_header else None

    for sheet in selected_sheets:
        df_sheet = pd.read_excel(xlsx_file, sheet_name=sheet, header=header)
        value_map, sig_map, mapped_rois = build_maps_for_sheet(df_sheet, parc_data)

        value_img = nib.Nifti1Image(value_map, bg_img.affine, bg_img.header)
        sig_img = nib.Nifti1Image(sig_map.astype(np.uint8), bg_img.affine, bg_img.header)

        if save_nii:
            sheet_tag = sanitize_name(sheet)
            nib.save(value_img, os.path.join(out_dir, f"{sheet_tag}_wmparc_roi_value_map.nii.gz"))
            nib.save(sig_img, os.path.join(out_dir, f"{sheet_tag}_wmparc_roi_significant_map.nii.gz"))

        value_data = nib.as_closest_canonical(value_img).get_fdata()
        sig_data = nib.as_closest_canonical(sig_img).get_fdata()

        sheet_tag = sanitize_name(sheet)
        out_png = os.path.join(out_dir, f"{output_prefix}_{sheet_tag}.png")
        plot_axial_overlap_mosaic(bg_data, value_data, sig_data, out_png)

        print(f"Sheet: {sheet}")
        print(f"Mapped ROI number: {len(mapped_rois)}")
        print(f"Output file: {out_png}")

    print(f"Selected axial slices: {axial_slices}")
    print(f"Colorbar range: {colorbar_vmin} to {colorbar_vmax}")
    print(f"Background mode: {bg_mode}")
    print(f"Background range: {bg_vmin} to {bg_vmax}")
    print(f"Background gamma: {bg_gamma}")
    print(f"Display left is left: {display_left_is_left}")
    print(f"Output directory: {out_dir}")