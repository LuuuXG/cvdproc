import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import linregress

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


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    out_file: str,
    title: str,
    mask_zero_union: bool = True,
    point_size: float = 5.0,
    alpha: float = 0.25,
) -> None:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")

    if mask_zero_union:
        mask = (x != 0) | (y != 0)
        x = x[mask]
        y = y[mask]

    if x.size < 3:
        raise RuntimeError("Not enough data points for scatter/regression after masking.")

    lr = linregress(x, y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=point_size, alpha=alpha)
    plt.plot(x, lr.slope * x + lr.intercept, linewidth=2)

    plt.xlabel("Direct map")
    plt.ylabel("Indirect map")
    plt.title(f"{title}\nPearson r = {lr.rvalue:.3f}, p = {lr.pvalue:.3e}")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Scatter saved: {out_file}")
    print(f"Scatter Pearson r = {lr.rvalue:.3f}, p = {lr.pvalue:.3e}, N = {x.size}")


def main():
    out_dir = "/mnt/e/Neuroimage/takehome/compare"
    target_density = "10k"

    n_perm = 100
    seed = 1234

    zero_tol = 0.0
    mask_zero_union_for_scatter = True

    direct_res = [
        os.path.join(out_dir, "direct_lh.func.gii"),
        os.path.join(out_dir, "direct_rh.func.gii"),
    ]
    indirect_res = [
        os.path.join(out_dir, "lh_smooth6mm_mean.func.gii"),
        os.path.join(out_dir, "rh_smooth6mm_mean.func.gii"),
    ]

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

    # Extract arrays for zero-fraction + scatter (downsampled)
    direct_lh_data = np.asarray(direct_ds[0].agg_data(), dtype=np.float32)
    direct_rh_data = np.asarray(direct_ds[1].agg_data(), dtype=np.float32)
    indirect_lh_data = np.asarray(indirect_ds[0].agg_data(), dtype=np.float32)
    indirect_rh_data = np.asarray(indirect_ds[1].agg_data(), dtype=np.float32)

    report_zero_fraction("Direct", direct_lh_data, direct_rh_data, tol=zero_tol)
    report_zero_fraction("Indirect", indirect_lh_data, indirect_rh_data, tol=zero_tol)

    # 3) Save downsampled files (optional but convenient for compare_images)
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

    # 4) Scatter plot (observed)
    direct_all = np.concatenate([direct_lh_data, direct_rh_data])
    indirect_all = np.concatenate([indirect_lh_data, indirect_rh_data])

    scatter_path = os.path.join(out_dir, f"direct_vs_indirect_scatter_{target_density}.png")
    plot_scatter(
        direct_all,
        indirect_all,
        out_file=scatter_path,
        title=f"Direct vs Indirect ({target_density})",
        mask_zero_union=mask_zero_union_for_scatter,
        point_size=5.0,
        alpha=0.25,
    )

    # 5) Spin-based nulls (Alexander-Bloch) on downsampled data
    direct_data_ds = images.load_data(direct_ds).astype(np.float32)
    rotated = nulls.alexander_bloch(
        direct_data_ds,
        atlas="fsaverage",
        density=target_density,
        n_perm=n_perm,
        seed=seed,
    )

    # 6) Compare images with spin nulls
    corr, pval = compare_images(
        list(direct_ds_paths),
        list(indirect_ds_paths),
        metric="pearsonr",
        nulls=rotated,
    )
    print(f"Correlation ({target_density}) with spin test: r = {corr:.03f}, p = {pval:.04f}")


if __name__ == "__main__":
    main()
