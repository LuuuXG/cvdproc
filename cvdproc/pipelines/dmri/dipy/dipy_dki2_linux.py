#!/usr/bin/env python3

# ---- Limit threads before importing numpy/scipy/dipy ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, time, faulthandler, json, traceback
import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.core.sphere import Sphere

# ====== EDIT YOUR PATHS HERE ======
DWI_PATH  = "/mnt/f/BIDS/7T_SVD/derivatives/dwi_pipeline/sub-SVD7TNEW02/ses-01/sub-SVD7TNEW02_ses-01_acq-DKIb2000_dir-PA_space-preprocdwi_desc-preproc_dwi.nii.gz"
BVAL_PATH = "/mnt/f/BIDS/7T_SVD/derivatives/dwi_pipeline/sub-SVD7TNEW02/ses-01/sub-SVD7TNEW02_ses-01_acq-DKIb2000_dir-PA_space-preprocdwi_desc-preproc_dwi.bval"
BVEC_PATH = "/mnt/f/BIDS/7T_SVD/derivatives/dwi_pipeline/sub-SVD7TNEW02/ses-01/sub-SVD7TNEW02_ses-01_acq-DKIb2000_dir-PA_space-preprocdwi_desc-preproc_dwi.bvec"
MASK_PATH = "/mnt/f/BIDS/7T_SVD/derivatives/dwi_pipeline/sub-SVD7TNEW02/ses-01/sub-SVD7TNEW02_ses-01_acq-DKIb2000_dir-PA_space-preprocdwi_desc-brain_mask.nii.gz"
OUT_DIR   = os.path.join(os.path.dirname(DWI_PATH), "dki_outputs")
QUICK_TEST = False
# ==================================

faulthandler.enable()
np.seterr(all="raise")

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save_like(ref_img: nib.Nifti1Image, data, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), ref_img.affine, hdr)
    nib.save(img, out_path)
    log(f"Saved -> {out_path}")

def call_metric(obj, name, **kwargs):
    """Call obj.name(**kwargs) if callable; else return attribute as-is."""
    if not hasattr(obj, name):
        return None
    attr = getattr(obj, name)
    if callable(attr):
        try:
            return attr(**kwargs)
        except TypeError:
            # For older versions that don't accept kwargs
            return attr()
    return attr

def save_from_dki_fit(dkifit, ref_img, out_dir, mask=None):
    os.makedirs(out_dir, exist_ok=True)

    # 1) DTI-like metrics (from DKI fit)
    for name, fname in [("fa", "FA.nii.gz"), ("md", "MD.nii.gz"),
                        ("ad", "AD.nii.gz"), ("rd", "RD.nii.gz")]:
        arr = call_metric(dkifit, name)
        if arr is not None:
            save_like(ref_img, arr, os.path.join(out_dir, fname))

    # 2) Kurtosis scalars: use numerical estimators to avoid analytical bug
    for name, fname in [("mk", "MK.nii.gz"), ("ak", "AK.nii.gz"), ("rk", "RK.nii.gz")]:
        arr = call_metric(dkifit, name, analytical=False)  # <--- key change
        if arr is not None:
            save_like(ref_img, arr, os.path.join(out_dir, fname))

    # 3) Axis AKC (Kxxxx, Kyyyy, Kzzzz)
    sphere = Sphere(xyz=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=float))
    K_axes = dkifit.akc(sphere)  # (X,Y,Z,3) or (Nmask,3)
    if K_axes.ndim == 2 and K_axes.shape[-1] == 3:
        if mask is None:
            raise RuntimeError("akc returned flattened array but mask is None.")
        vol = np.zeros(mask.shape + (3,), dtype=np.float32)
        vol[mask] = K_axes.astype(np.float32)
        K_axes = vol
    save_like(ref_img, K_axes,         os.path.join(out_dir, "K_axes.nii.gz"))
    save_like(ref_img, K_axes[..., 0], os.path.join(out_dir, "Kxxxx.nii.gz"))
    save_like(ref_img, K_axes[..., 1], os.path.join(out_dir, "Kyyyy.nii.gz"))
    save_like(ref_img, K_axes[..., 2], os.path.join(out_dir, "Kzzzz.nii.gz"))

    # 4) Diffusion tensor 3x3 -> 6 components
    D = call_metric(dkifit, "quadratic_form")
    if D is not None and D.ndim >= 2:
        D6 = np.stack([D[...,0,0], D[...,1,1], D[...,2,2],
                       D[...,0,1], D[...,0,2], D[...,1,2]], axis=-1)
        save_like(ref_img, D6, os.path.join(out_dir, "D6_from_dki.nii.gz"))

    # 5) Kurtosis tensor 15 comps (Voigt) if available
    W15 = call_metric(dkifit, "kt")
    meta = {
        "dipy_version": __import__("dipy").__version__,
        "numpy_version": np.__version__,
        "units": {"D": "mm^2/s", "K": "dimensionless"},
        "D6_order": ["Dxx","Dyy","Dzz","Dxy","Dxz","Dyz"],
        "W15_voigt_order_hint": [
            "Wxxxx","Wyyyy","Wzzzz","Wxxxy","Wxxxz",
            "Wyyyx","Wyyyz","Wzzzx","Wzzzy","Wxxyy",
            "Wxxzz","Wyyzz","Wxyxy","Wxzxz","Wyzyz"
        ],
        "K_axes_dirs": [[1,0,0],[0,1,0],[0,0,1]]
    }
    if W15 is not None:
        save_like(ref_img, W15.astype(np.float32), os.path.join(out_dir, "W15_voigt.nii.gz"))
    with open(os.path.join(out_dir, "dki_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("Meta saved.")

def main():
    log("Loading data...")
    data, _ = load_nifti(DWI_PATH)
    bvals, bvecs = read_bvals_bvecs(BVAL_PATH, BVEC_PATH)
    gtab = gradient_table(bvals, bvecs=bvecs)
    mask = nib.load(MASK_PATH).get_fdata().astype(bool)
    ref_img = nib.load(DWI_PATH)
    log(f"data shape: {data.shape}, mask shape: {mask.shape}, #bvals: {bvals.size}")

    if QUICK_TEST:
        z0, z1 = data.shape[2]//2 - 2, data.shape[2]//2 + 3
        data = data[:, :, z0:z1, :]
        mask = mask[:, :, z0:z1]
        log(f"[QUICK] cropped to z={z0}:{z1}, new shape: {data.shape}")

    assert data.shape[:3] == mask.shape, "Mask shape mismatch"
    assert data.shape[-1] == bvals.size, "Gradient length mismatch"

    data = np.ascontiguousarray(data.astype(np.float64))
    mask = np.ascontiguousarray(mask)

    log("Fitting DiffusionKurtosisModel...")
    t0 = time.time()
    dkimodel = DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data, mask=mask)
    log(f"DKI fit done in {time.time()-t0:.1f}s")

    log("Saving outputs...")
    save_from_dki_fit(dkifit, ref_img, OUT_DIR, mask=mask)
    log("All done.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("ERROR encountered:")
        traceback.print_exc()
        sys.exit(1)
