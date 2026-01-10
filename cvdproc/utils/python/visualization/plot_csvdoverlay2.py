import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

# ========= User settings =========
flair_path   = r"F:\BIDS\demo_BIDS\derivatives\xfm\sub-AFib0241\ses-baseline\sub-AFib0241_ses-baseline_acq-highres_space-T1w_desc-brain_FLAIR.nii.gz"
pwmh_path    = r"F:\BIDS\demo_BIDS\derivatives\wmh_quantification\sub-AFib0241\ses-baseline\sub-AFib0241_ses-baseline_space-T1w_label-PWMH_desc-truenetThr0p30_mask.nii.gz"
dwmh_path    = r"F:\BIDS\demo_BIDS\derivatives\wmh_quantification\sub-AFib0241\ses-baseline\sub-AFib0241_ses-baseline_space-T1w_label-DWMH_desc-truenetThr0p30_mask.nii.gz"
output_gif   = r"F:\BIDS\demo_BIDS\derivatives\wmh_quantification\sub-AFib0241\ses-baseline\wmh_overlay_pwmh_dwmh.gif"

slice_idx = 94          # choose axial slice
min_alpha = 0.05         # minimum opacity
fade_down_frames = 15
fade_up_frames   = 15
frame_duration   = 0.08  # seconds per frame

# colors (RGB in [0,1])
pwmh_color = (1.00, 0.84, 0.10)  # yellow-ish
dwmh_color = (0.10, 0.70, 1.00)  # cyan-ish

# animation mode: "together" (both fade same) or "counterphase" (one fades in while the other fades out)
mode = "together"

# output size
dpi = 100
target_height_px = 600

# ========= Load & prepare =========
flair_nii = nib.as_closest_canonical(nib.load(flair_path))
pwmh_nii  = nib.as_closest_canonical(nib.load(pwmh_path))
dwmh_nii  = nib.as_closest_canonical(nib.load(dwmh_path))

flair = flair_nii.get_fdata(dtype=np.float32)
pwmh  = pwmh_nii.get_fdata(dtype=np.float32)
dwmh  = dwmh_nii.get_fdata(dtype=np.float32)

# basic checks
if flair.shape != pwmh.shape or flair.shape != dwmh.shape:
    raise ValueError(f"Shape mismatch: FLAIR {flair.shape}, PWMH {pwmh.shape}, DWMH {dwmh.shape}")
if not (0 <= slice_idx < flair.shape[2]):
    raise ValueError(f"slice_idx out of range: {slice_idx} (valid 0..{flair.shape[2]-1})")

# pick slice
flair_slice = flair[:, :, slice_idx]
pwmh_slice  = pwmh[:,  :, slice_idx]
dwmh_slice  = dwmh[:,  :, slice_idx]

# normalize flair for display
fmin, fmax = np.nanmin(flair_slice), np.nanmax(flair_slice)
flair_disp = (flair_slice - fmin) / (fmax - fmin + 1e-8)

# binarize masks
pwmh_bin = (pwmh_slice > 0).astype(np.uint8)
dwmh_bin = (dwmh_slice > 0).astype(np.uint8)

# prebuild RGB overlays
h, w = flair_disp.shape
pwmh_rgb = np.zeros((h, w, 3), dtype=np.float32); pwmh_rgb[:] = pwmh_color
dwmh_rgb = np.zeros((h, w, 3), dtype=np.float32); dwmh_rgb[:] = dwmh_color

# alpha schedule
alpha_seq = np.concatenate([
    np.linspace(1.0, min_alpha, fade_down_frames, endpoint=True),
    np.linspace(min_alpha, 1.0, fade_up_frames,   endpoint=True)
])

# set output canvas size (keep aspect ratio, no borders)
target_width_px = int(round(target_height_px * (w / h)))

frames = []
for a in alpha_seq:
    if mode == "together":
        a_pwmh = a
        a_dwmh = a
    elif mode == "counterphase":
        a_pwmh = a
        a_dwmh = 1.0 - (a - min_alpha) / (1.0 - min_alpha + 1e-8)  # roughly opposite phase
        a_dwmh = float(np.clip(a_dwmh, min_alpha, 1.0))
    else:
        raise ValueError("mode must be 'together' or 'counterphase'")

    # alpha maps (mask==0 -> 0, mask>0 -> a_xxx)
    pwmh_alpha = pwmh_bin.astype(np.float32) * a_pwmh
    dwmh_alpha = dwmh_bin.astype(np.float32) * a_dwmh

    # RGBA
    pwmh_rgba = np.dstack([pwmh_rgb, pwmh_alpha])
    dwmh_rgba = np.dstack([dwmh_rgb, dwmh_alpha])

    # figure without margins
    fig = plt.figure(figsize=(target_width_px / dpi, target_height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_alpha(0)

    # draw base and overlays; the later one is on top when overlapping
    ax.imshow(flair_disp, cmap="gray", origin="lower", interpolation="nearest")
    ax.imshow(pwmh_rgba, origin="lower", interpolation="nearest")
    ax.imshow(dwmh_rgba, origin="lower", interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(0, h)

    # grab frame
    fig.canvas.draw()
    cw, ch = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((ch, cw, 3))
    frames.append(frame)
    plt.close(fig)

# ========= Save GIF =========
os.makedirs(os.path.dirname(output_gif), exist_ok=True)
imageio.mimsave(output_gif, frames, duration=frame_duration, loop=0)

print(f"Saved GIF: {output_gif}")
print(f"Slice: {slice_idx}/{flair.shape[2]-1} | Size: {target_width_px}x{target_height_px}px | Mode: {mode}")
