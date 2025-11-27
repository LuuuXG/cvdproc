import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # 防止GUI后台冲突
import matplotlib.pyplot as plt
import imageio

# ======== User settings ========
flair_path = r"F:\BIDS\demo_BIDS\derivatives\xfm\sub-AFib0241\ses-baseline\sub-AFib0241_ses-baseline_acq-highres_space-T1w_desc-brain_FLAIR.nii.gz"
wmh_mask_path = r"F:\BIDS\demo_BIDS\derivatives\wmh_quantification\sub-AFib0241\ses-baseline\sub-AFib0241_ses-baseline_space-T1w_label-WMH_desc-truenetThr0p30_mask.nii.gz"
output_gif = r"F:\BIDS\demo_BIDS\derivatives\wmh_quantification\sub-AFib0241\ses-baseline\wmh_overlay.gif"

slice_idx = 94     # 指定层面
min_alpha = 0.05   # 最低透明度
fade_down_frames = 15
fade_up_frames = 15
frame_duration = 0.08  # 每帧间隔秒数
dpi = 100              # 输出分辨率
target_height_px = 600 # 高度像素

# ======== Load & preprocess ========
flair_nii = nib.as_closest_canonical(nib.load(flair_path))
mask_nii  = nib.as_closest_canonical(nib.load(wmh_mask_path))
flair = flair_nii.get_fdata(dtype=np.float32)
mask  = mask_nii.get_fdata(dtype=np.float32)

if flair.shape != mask.shape:
    raise ValueError(f"Shape mismatch: FLAIR {flair.shape} vs WMH mask {mask.shape}")
if not (0 <= slice_idx < flair.shape[2]):
    raise ValueError(f"slice_idx out of range: {slice_idx}")

# 提取指定层面
flair_slice = flair[:, :, slice_idx]
mask_slice  = mask[:,  :, slice_idx]

# 归一化 FLAIR
flair_min, flair_max = np.nanmin(flair_slice), np.nanmax(flair_slice)
flair_disp = (flair_slice - flair_min) / (flair_max - flair_min + 1e-8)

# 二值化 mask
mask_bin = (mask_slice > 0).astype(np.uint8)
overlay_rgb = np.zeros((*mask_bin.shape, 3), dtype=np.float32)
overlay_rgb[..., 0] = 1.0  # 红色
# use yellow'
# overlay_rgb[..., 0] = 1.0
# overlay_rgb[..., 1] = 0.84
# overlay_rgb[..., 2] = 0.10

# 透明度曲线
alphas = np.concatenate([
    np.linspace(1.0, min_alpha, fade_down_frames, endpoint=True),
    np.linspace(min_alpha, 1.0, fade_up_frames,   endpoint=True)
])

# ======== Generate frames ========
h, w = flair_disp.shape
target_width_px = int(round(target_height_px * (w / h)))  # 保持比例
frames = []

for alpha in alphas:
    alpha_map = mask_bin.astype(np.float32) * alpha
    overlay_rgba = np.dstack([overlay_rgb, alpha_map])

    # 无边距画布
    fig = plt.figure(figsize=(target_width_px / dpi, target_height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # 填满整个画布
    ax.set_axis_off()
    fig.patch.set_alpha(0)

    ax.imshow(flair_disp, cmap="gray", origin="lower", interpolation="nearest")
    ax.imshow(overlay_rgba, origin="lower", interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)

    fig.canvas.draw()
    cw, ch = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((ch, cw, 3))
    frames.append(frame)
    plt.close(fig)

# ======== Save GIF ========
os.makedirs(os.path.dirname(output_gif), exist_ok=True)
imageio.mimsave(output_gif, frames, duration=frame_duration, loop=0)
print(f"✅ Saved overlay GIF: {output_gif}")
print(f"Slice index used: {slice_idx} / {flair.shape[2]-1}")
print(f"Image size: {target_width_px}x{target_height_px}px")
