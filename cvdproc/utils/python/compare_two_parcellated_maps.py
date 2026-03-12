import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr

from neuromaps import images, nulls, parcellate
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


def load_gifti_as_int(path: str) -> np.ndarray:
    img = nib.load(path)
    if not isinstance(img, nib.gifti.gifti.GiftiImage):
        raise TypeError(f"Not a GIFTI file: {path}")
    return np.asarray(img.agg_data()).ravel().astype(np.int32)


def labels_to_gifti_label(arr: np.ndarray) -> nib.gifti.gifti.GiftiImage:
    arr = np.asarray(arr, dtype=np.int32).ravel()
    return images.construct_shape_gii(arr, intent="NIFTI_INTENT_LABEL")


def metric_to_gifti_shape(arr: np.ndarray) -> nib.gifti.gifti.GiftiImage:
    arr = np.asarray(arr, dtype=np.float32).ravel()
    return images.construct_shape_gii(arr, intent="NIFTI_INTENT_SHAPE")


def summarize_labels(name: str, lh: np.ndarray, rh: np.ndarray) -> None:
    lh_u = np.unique(lh)
    rh_u = np.unique(rh)
    all_u = np.unique(np.concatenate([lh, rh]))
    all_u_nz = all_u[all_u != 0]

    print(f"--- {name} ---")
    print(f"LH unique: {lh_u.size}, min={int(lh_u.min())}, max={int(lh_u.max())}")
    print(f"RH unique: {rh_u.size}, min={int(rh_u.min())}, max={int(rh_u.max())}")
    print(f"Both unique (non-zero): {all_u_nz.size}")
    if all_u_nz.size > 0:
        print(f"Non-zero labels: min={int(all_u_nz.min())}, max={int(all_u_nz.max())}")
    print(f"Zero fraction LH: {np.mean(lh == 0) * 100:.3f}%")
    print(f"Zero fraction RH: {np.mean(rh == 0) * 100:.3f}%")
    print("")


def relabel_to_consecutive_across_hemi(lh: np.ndarray, rh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforce neuromaps parcellation convention:
      - background is 0
      - LH labels: 1..K_L
      - RH labels: K_L+1 .. K_L+K_R
      - no overlap across hemispheres
    """
    lh = np.asarray(lh, dtype=np.int32).copy()
    rh = np.asarray(rh, dtype=np.int32).copy()

    lh[lh < 0] = 0
    rh[rh < 0] = 0

    lh_ids = np.unique(lh)
    lh_ids = lh_ids[lh_ids != 0]
    if lh_ids.size == 0:
        raise RuntimeError("LH has no non-zero labels after fixing negatives.")
    lh_ids = np.sort(lh_ids)

    rh_ids = np.unique(rh)
    rh_ids = rh_ids[rh_ids != 0]
    if rh_ids.size == 0:
        raise RuntimeError("RH has no non-zero labels after fixing negatives.")
    rh_ids = np.sort(rh_ids)

    lh_new = lh.copy()
    lh_mask = lh_new != 0
    lh_idx = np.searchsorted(lh_ids, lh_new[lh_mask])
    lh_new[lh_mask] = (lh_idx + 1).astype(np.int32)
    k_l = int(lh_ids.size)

    rh_new = rh.copy()
    rh_mask = rh_new != 0
    rh_idx = np.searchsorted(rh_ids, rh_new[rh_mask])
    rh_new[rh_mask] = (rh_idx + 1 + k_l).astype(np.int32)

    overlap = np.intersect1d(np.unique(lh_new[lh_new != 0]), np.unique(rh_new[rh_new != 0]))
    if overlap.size > 0:
        raise RuntimeError(f"Non-zero label overlap remains after relabeling: {overlap[:10]}")

    return lh_new, rh_new


def log_eps_transform(arr: np.ndarray, eps: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return np.log(arr + eps)


def set_pub_style():
    plt.rcParams.update({
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })


def spin_pvalue(null_corr: np.ndarray, obs_corr: float) -> float:
    null_corr = np.asarray(null_corr, dtype=np.float64)
    null_corr = null_corr[~np.isnan(null_corr)]
    if null_corr.size == 0 or np.isnan(obs_corr):
        return np.nan
    return float((1.0 + np.sum(np.abs(null_corr) >= abs(obs_corr))) / (null_corr.size + 1.0))


def plot_scatter_pub_with_line(
    x: np.ndarray,
    y: np.ndarray,
    out_file: str,
    title: str,
    xlabel: str,
    ylabel: str,
    point_size: float,
    alpha: float,
    left: float = 0.18,
    bottom: float = 0.18,
    width: float = 0.72,
    height: float = 0.72,
    line_width: float = 2.5,
) -> tuple[float, float, float]:
    """
    Vertex-level style: same layout / fonts / axis thickness.
    Reports Spearman r + classical Spearman p, and draws an OLS guide line.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    r_s, p_s = spearmanr(x, y)
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

    print(f"Saved scatter: {out_file}")
    print(f"Spearman r = {r_s:.6f}, p = {p_s:.3e}, N = {x.size}")
    return float(r_s), float(p_s), float(lr.rvalue)


def plot_null_hist_pub(null_corr: np.ndarray, obs_corr: float, out_file: str, title: str) -> None:
    null_corr = np.asarray(null_corr, dtype=np.float64)
    null_corr = null_corr[~np.isnan(null_corr)]

    set_pub_style()
    fig = plt.figure(figsize=(7.0, 4.5))
    ax = fig.add_axes([0.12, 0.18, 0.82, 0.72])

    ax.hist(null_corr, bins=40)
    ax.axvline(obs_corr, linewidth=2)
    ax.axvline(-obs_corr, linewidth=2)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel("Null correlations", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)

    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved null histogram: {out_file}")


def main():
    out_dir = "/mnt/e/Neuroimage/takehome/compare"
    target_density = "10k"

    n_perm = 100
    seed = 1234
    log_eps = 1e-6

    direct_res = [
        os.path.join(out_dir, "direct_lh.func.gii"),
        os.path.join(out_dir, "direct_rh.func.gii"),
    ]
    indirect_res = [
        os.path.join(out_dir, "lh_smooth6mm_mean.func.gii"),
        os.path.join(out_dir, "rh_smooth6mm_mean.func.gii"),
    ]

    parc_lh_path = os.path.join(out_dir, "lh.aparc.label.gii")
    parc_rh_path = os.path.join(out_dir, "rh.aparc.label.gii")

    for p in direct_res + indirect_res + [parc_lh_path, parc_rh_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    # Ensure single darray for metric maps
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

    # Load and resample metric maps
    direct_lh = nib.load(direct_single[0])
    direct_rh = nib.load(direct_single[1])
    indirect_lh = nib.load(indirect_single[0])
    indirect_rh = nib.load(indirect_single[1])

    direct_ds = fsaverage_to_fsaverage((direct_lh, direct_rh), target_density=target_density, method="linear")
    indirect_ds = fsaverage_to_fsaverage((indirect_lh, indirect_rh), target_density=target_density, method="linear")

    # Load and resample parcellation (nearest)
    parc_lh_164k = load_gifti_as_int(parc_lh_path)
    parc_rh_164k = load_gifti_as_int(parc_rh_path)

    parc_lh_gii_164k = labels_to_gifti_label(parc_lh_164k)
    parc_rh_gii_164k = labels_to_gifti_label(parc_rh_164k)

    parc_ds = fsaverage_to_fsaverage(
        (parc_lh_gii_164k, parc_rh_gii_164k),
        target_density=target_density,
        method="nearest",
    )

    parc_lh_ds = np.asarray(parc_ds[0].agg_data()).ravel().astype(np.int32)
    parc_rh_ds = np.asarray(parc_ds[1].agg_data()).ravel().astype(np.int32)

    summarize_labels(f"aparc labels after resample ({target_density}) [raw]", parc_lh_ds, parc_rh_ds)

    parc_lh_fix, parc_rh_fix = relabel_to_consecutive_across_hemi(parc_lh_ds, parc_rh_ds)
    summarize_labels(f"aparc labels after resample ({target_density}) [consecutive]", parc_lh_fix, parc_rh_fix)

    parcellation = (labels_to_gifti_label(parc_lh_fix), labels_to_gifti_label(parc_rh_fix))
    parc_model = parcellate.Parcellater(parcellation, "fsaverage").fit()

    # Observed parcellated vectors
    direct_parc = np.asarray(parc_model.transform(direct_ds, "fsaverage"), dtype=np.float64)
    indirect_parc = np.asarray(parc_model.transform(indirect_ds, "fsaverage"), dtype=np.float64)

    if direct_parc.shape != indirect_parc.shape:
        raise RuntimeError(f"Parcellated shapes mismatch: {direct_parc.shape} vs {indirect_parc.shape}")

    k = int(direct_parc.size)
    print(f"Parcellated ROI count (excluding background): {k}")

    # Generate vertex-level spins on DIRECT, then parcellate each rotated map
    direct_data = images.load_data(direct_ds).astype(np.float32)
    rotated = nulls.alexander_bloch(
        direct_data,
        atlas="fsaverage",
        density=target_density,
        n_perm=n_perm,
        seed=seed,
    )

    n_lh = int(np.asarray(direct_ds[0].agg_data()).ravel().shape[0])
    n_rh = int(np.asarray(direct_ds[1].agg_data()).ravel().shape[0])
    if n_lh + n_rh != int(rotated.shape[0]):
        raise RuntimeError(f"Vertex count mismatch: {n_lh}+{n_rh} vs rotated {rotated.shape[0]}")

    # -------------------------
    # A) ORIGINAL SCALE (Spearman + spin p)
    # -------------------------
    scatter_path_a = os.path.join(out_dir, f"direct_vs_indirect_aparc_scatter_{target_density}_spearman_original.png")
    obs_r_a, obs_p_a, _ = plot_scatter_pub_with_line(
        direct_parc,
        indirect_parc,
        out_file=scatter_path_a,
        title=f"Direct vs Indirect (aparc, {target_density})",
        xlabel="Direct (ROI mean)",
        ylabel="Indirect (ROI mean)",
        point_size=55.0,
        alpha=0.85,
    )

    null_corr_a = np.full(n_perm, np.nan, dtype=np.float64)
    for i in range(n_perm):
        rot_vec = rotated[:, i].astype(np.float32)
        rot_lh = rot_vec[:n_lh]
        rot_rh = rot_vec[n_lh:]
        rot_ds = (metric_to_gifti_shape(rot_lh), metric_to_gifti_shape(rot_rh))
        rot_parc = np.asarray(parc_model.transform(rot_ds, "fsaverage"), dtype=np.float64)
        if rot_parc.size != k:
            raise RuntimeError(f"Rotated parcellation size mismatch: {rot_parc.size} vs {k}")
        rr, _ = spearmanr(rot_parc, indirect_parc)
        null_corr_a[i] = rr

    p_spin_a = spin_pvalue(null_corr_a, obs_r_a)
    print(f"ROI-level Spearman (original): r = {obs_r_a:.6f}, classical p = {obs_p_a:.3e}, spin p = {p_spin_a:.6f}")

    hist_path_a = os.path.join(out_dir, f"direct_vs_indirect_aparc_spin_null_{target_density}_spearman_original.png")
    plot_null_hist_pub(
        null_corr_a,
        obs_r_a,
        out_file=hist_path_a,
        title=f"Spin nulls (ROI-level Spearman, aparc, {target_density})",
    )

    # -------------------------
    # B) LOG SCALE (Spearman + spin p)
    # -------------------------
    direct_parc_log = log_eps_transform(direct_parc, eps=log_eps)
    indirect_parc_log = log_eps_transform(indirect_parc, eps=log_eps)

    scatter_path_b = os.path.join(out_dir, f"direct_vs_indirect_aparc_scatter_{target_density}_spearman_log.png")
    obs_r_b, obs_p_b, _ = plot_scatter_pub_with_line(
        direct_parc_log,
        indirect_parc_log,
        out_file=scatter_path_b,
        title=f"Direct vs Indirect (log, eps={log_eps:g}, aparc, {target_density})",
        xlabel="Direct (ROI mean, log)",
        ylabel="Indirect (ROI mean, log)",
        point_size=55.0,
        alpha=0.85,
    )

    null_corr_b = np.full(n_perm, np.nan, dtype=np.float64)
    for i in range(n_perm):
        rot_vec = rotated[:, i].astype(np.float32)
        rot_lh = rot_vec[:n_lh]
        rot_rh = rot_vec[n_lh:]
        rot_ds = (metric_to_gifti_shape(rot_lh), metric_to_gifti_shape(rot_rh))
        rot_parc = np.asarray(parc_model.transform(rot_ds, "fsaverage"), dtype=np.float64)

        rot_parc_log = log_eps_transform(rot_parc, eps=log_eps)
        rr, _ = spearmanr(rot_parc_log, indirect_parc_log)
        null_corr_b[i] = rr

    p_spin_b = spin_pvalue(null_corr_b, obs_r_b)
    print(f"ROI-level Spearman (log): r = {obs_r_b:.6f}, classical p = {obs_p_b:.3e}, spin p = {p_spin_b:.6f}")

    hist_path_b = os.path.join(out_dir, f"direct_vs_indirect_aparc_spin_null_{target_density}_spearman_log.png")
    plot_null_hist_pub(
        null_corr_b,
        obs_r_b,
        out_file=hist_path_b,
        title=f"Spin nulls (ROI-level Spearman, log, aparc, {target_density})",
    )

    out_npz = os.path.join(out_dir, f"direct_indirect_aparc_spin_{target_density}_spearman_original_and_log.npz")
    np.savez(
        out_npz,
        direct_parc=direct_parc,
        indirect_parc=indirect_parc,
        obs_r_spearman=np.array([obs_r_a], dtype=np.float64),
        obs_p_spearman=np.array([obs_p_a], dtype=np.float64),
        null_corr_spearman=null_corr_a,
        p_spin_spearman=np.array([p_spin_a], dtype=np.float64),
        direct_parc_log=direct_parc_log,
        indirect_parc_log=indirect_parc_log,
        obs_r_spearman_log=np.array([obs_r_b], dtype=np.float64),
        obs_p_spearman_log=np.array([obs_p_b], dtype=np.float64),
        null_corr_spearman_log=null_corr_b,
        p_spin_spearman_log=np.array([p_spin_b], dtype=np.float64),
        log_eps=np.array([log_eps], dtype=np.float64),
        target_density=np.array([target_density]),
        n_perm=np.array([n_perm], dtype=np.int32),
        seed=np.array([seed], dtype=np.int32),
    )
    print(f"Saved results: {out_npz}")


if __name__ == "__main__":
    main()
