#!/usr/bin/env python3

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress

from neuromaps.stats import compare_images
from neuromaps import images, nulls
from neuromaps.transforms import fsaverage_to_fsaverage


def ensure_single_darray(in_gii: str, out_gii: str) -> str:
    img = nib.load(in_gii)
    if not isinstance(img, nib.gifti.gifti.GiftiImage):
        raise TypeError(f"Not a GIFTI file: {in_gii}")

    data = np.asarray(img.agg_data())
    if data.ndim == 1:
        if os.path.abspath(in_gii) != os.path.abspath(out_gii):
            nib.save(img, out_gii)
        return out_gii

    if data.ndim == 2:
        single = data[:, 0].astype(np.float32)
        out_img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(single)])
        nib.save(out_img, out_gii)
        return out_gii

    raise ValueError(f"Unsupported data shape in {in_gii}: {data.shape}")


def save_gifti(img: nib.gifti.GiftiImage, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(img, out_path)


def zero_fraction(arr: np.ndarray, tol: float = 0.0) -> float:
    arr = np.asarray(arr)
    if tol <= 0.0:
        return float(np.mean(arr == 0))
    return float(np.mean(np.isclose(arr, 0.0, atol=tol)))


def report_zero_fraction(name: str, lh: np.ndarray, rh: np.ndarray, tol: float = 0.0) -> None:
    lh0 = zero_fraction(lh, tol=tol)
    rh0 = zero_fraction(rh, tol=tol)
    both0 = zero_fraction(np.concatenate([lh, rh]), tol=tol)
    print(f"{name} zero fraction (LH): {lh0 * 100:.2f}%")
    print(f"{name} zero fraction (RH): {rh0 * 100:.2f}%")
    print(f"{name} zero fraction (Both): {both0 * 100:.2f}%")


def log_eps(arr: np.ndarray, eps: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return np.log(arr + eps)


def set_pub_style() -> None:
    plt.rcParams.update({
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })


def plot_scatter_spearman(
    x: np.ndarray,
    y: np.ndarray,
    out_file: str,
    title: str,
    xlabel: str,
    ylabel: str,
    mask_zero_union: bool = True,
    point_size: float = 5.0,
    alpha: float = 0.25,
    line_width: float = 2.5,
    left: float = 0.18,
    bottom: float = 0.18,
    width: float = 0.72,
    height: float = 0.72,
) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")

    if mask_zero_union:
        m = (x != 0) | (y != 0)
        x = x[m]
        y = y[m]

    if x.size < 3:
        raise RuntimeError("Not enough data points after masking.")

    r_s, p_s = spearmanr(x, y)

    # OLS line for visualization (consistent with your ROI plots)
    lr = linregress(x, y)

    set_pub_style()
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_axes([left, bottom, width, height])

    ax.scatter(x, y, s=point_size, alpha=alpha)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    xx = np.array([x_min, x_max], dtype=np.float64)
    yy = lr.slope * xx + lr.intercept
    ax.plot(xx, yy, linewidth=line_width)

    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)

    ax.set_title(
        f"{title}\nSpearman r = {r_s:.3f}, p = {p_s:.3e}, N = {x.size}",
        fontsize=17,
        fontweight="bold",
        pad=15,
    )

    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Scatter saved: {out_file}")
    print(f"Spearman r = {r_s:.6f}, p = {p_s:.3e}, N = {x.size}")
    print(f"OLS line: slope = {lr.slope:.6g}, intercept = {lr.intercept:.6g}")

    return float(r_s), float(p_s), int(x.size)


def main():
    out_dir = "/mnt/e/Neuroimage/takehome/compare"
    target_density = "10k"

    n_perm = 5000
    seed = 1234

    zero_tol = 0.0
    mask_zero_union_for_scatter = True

    eps = 1e-6

    direct_res = [
        os.path.join(out_dir, "direct_lh.func.gii"),
        os.path.join(out_dir, "direct_rh.func.gii"),
    ]
    indirect_res = [
        os.path.join(out_dir, "lh_smooth6mm_mean.func.gii"),
        os.path.join(out_dir, "rh_smooth6mm_mean.func.gii"),
    ]

    for p in direct_res + indirect_res:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    # 1) Ensure single darray
    direct_single = []
    for p in direct_res:
        base = os.path.basename(p).replace(".func.gii", "_single.func.gii")
        out_p = os.path.join(out_dir, base)
        direct_single.append(ensure_single_darray(p, out_p))

    indirect_single = []
    for p in indirect_res:
        base = os.path.basename(p).replace(".func.gii", "_single.func.gii")
        out_p = os.path.join(out_dir, base)
        indirect_single.append(ensure_single_darray(p, out_p))

    # 2) Load and resample to target density
    direct_lh = nib.load(direct_single[0])
    direct_rh = nib.load(direct_single[1])
    indirect_lh = nib.load(indirect_single[0])
    indirect_rh = nib.load(indirect_single[1])

    direct_ds = fsaverage_to_fsaverage((direct_lh, direct_rh), target_density=target_density, method="linear")
    indirect_ds = fsaverage_to_fsaverage((indirect_lh, indirect_rh), target_density=target_density, method="linear")

    direct_lh_data = np.asarray(direct_ds[0].agg_data(), dtype=np.float32)
    direct_rh_data = np.asarray(direct_ds[1].agg_data(), dtype=np.float32)
    indirect_lh_data = np.asarray(indirect_ds[0].agg_data(), dtype=np.float32)
    indirect_rh_data = np.asarray(indirect_ds[1].agg_data(), dtype=np.float32)

    report_zero_fraction("Direct", direct_lh_data, direct_rh_data, tol=zero_tol)
    report_zero_fraction("Indirect", indirect_lh_data, indirect_rh_data, tol=zero_tol)

    # 3) Save downsampled files for compare_images()
    direct_ds_paths = (
        os.path.join(out_dir, f"direct_lh_{target_density}.func.gii"),
        os.path.join(out_dir, f"direct_rh_{target_density}.func.gii"),
    )
    indirect_ds_paths = (
        os.path.join(out_dir, f"indirect_lh_{target_density}.func.gii"),
        os.path.join(out_dir, f"indirect_rh_{target_density}.func.gii"),
    )
    save_gifti(direct_ds[0], direct_ds_paths[0])
    save_gifti(direct_ds[1], direct_ds_paths[1])
    save_gifti(indirect_ds[0], indirect_ds_paths[0])
    save_gifti(indirect_ds[1], indirect_ds_paths[1])

    # 4) Prepare concatenated arrays for scatter
    direct_all = np.concatenate([direct_lh_data, direct_rh_data]).astype(np.float64)
    indirect_all = np.concatenate([indirect_lh_data, indirect_rh_data]).astype(np.float64)

    # 5) Vertex-level spin nulls computed from ORIGINAL direct data
    direct_data_ds = images.load_data(direct_ds).astype(np.float32)
    rotated = nulls.alexander_bloch(
        direct_data_ds,
        atlas="fsaverage",
        density=target_density,
        n_perm=n_perm,
        seed=seed,
    )

    # ---------- Analysis A: ORIGINAL (Spearman + spin p) ----------
    scatter_path_a = os.path.join(out_dir, f"direct_vs_indirect_scatter_{target_density}_spearman.png")
    plot_scatter_spearman(
        direct_all,
        indirect_all,
        out_file=scatter_path_a,
        title=f"Direct vs Indirect ({target_density})",
        xlabel="Direct map (vertex level)",
        ylabel="Indirect map (vertex level)",
        mask_zero_union=mask_zero_union_for_scatter,
        point_size=5.0,
        alpha=0.25,
        line_width=2.5,
    )

    corr_a, p_spin_a = compare_images(
        list(direct_ds_paths),
        list(indirect_ds_paths),
        metric="spearmanr",
        nulls=rotated,
    )
    print(f"Original scale ({target_density}) spin test: r = {corr_a:.6f}, p = {p_spin_a:.6g}")

    # ---------- Analysis B: LOG (Spearman + spin p) ----------
    direct_log = log_eps(direct_all, eps=eps)
    indirect_log = log_eps(indirect_all, eps=eps)

    scatter_path_b = os.path.join(out_dir, f"direct_vs_indirect_scatter_{target_density}_log_spearman.png")
    plot_scatter_spearman(
        direct_log,
        indirect_log,
        out_file=scatter_path_b,
        title=f"Direct vs Indirect (log, eps={eps:g}, {target_density})",
        xlabel="Direct map (vertex level, log)",
        ylabel="Indirect map (vertex level, log)",
        mask_zero_union=mask_zero_union_for_scatter,
        point_size=5.0,
        alpha=0.25,
        line_width=2.5,
    )

    # Create log-transformed GIFTIs (10k) for compare_images, reuse SAME rotated spins
    direct_lh_log = log_eps(direct_lh_data, eps=eps).astype(np.float32)
    direct_rh_log = log_eps(direct_rh_data, eps=eps).astype(np.float32)
    indirect_lh_log = log_eps(indirect_lh_data, eps=eps).astype(np.float32)
    indirect_rh_log = log_eps(indirect_rh_data, eps=eps).astype(np.float32)

    direct_log_paths = (
        os.path.join(out_dir, f"direct_lh_{target_density}_log.func.gii"),
        os.path.join(out_dir, f"direct_rh_{target_density}_log.func.gii"),
    )
    indirect_log_paths = (
        os.path.join(out_dir, f"indirect_lh_{target_density}_log.func.gii"),
        os.path.join(out_dir, f"indirect_rh_{target_density}_log.func.gii"),
    )

    nib.save(
        nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(direct_lh_log)]),
        direct_log_paths[0],
    )
    nib.save(
        nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(direct_rh_log)]),
        direct_log_paths[1],
    )
    nib.save(
        nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(indirect_lh_log)]),
        indirect_log_paths[0],
    )
    nib.save(
        nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(indirect_rh_log)]),
        indirect_log_paths[1],
    )

    corr_b, p_spin_b = compare_images(
        list(direct_log_paths),
        list(indirect_log_paths),
        metric="spearmanr",
        nulls=rotated,
    )
    print(f"Log scale ({target_density}, eps={eps:g}) spin test: r = {corr_b:.6f}, p = {p_spin_b:.6g}")


if __name__ == "__main__":
    main()
