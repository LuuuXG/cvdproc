import os
import sys, time, json, traceback
import numpy as np
import nibabel as nib
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.core.sphere import Sphere


class DKIFitInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc="Path to the DWI file")
    bval_file = File(exists=True, mandatory=True, desc="Path to the bval file")
    bvec_file = File(exists=True, mandatory=True, desc="Path to the bvec file")
    mask_file = File(exists=True, mandatory=True, desc="Path to the brain mask file")
    output_dir = Directory(mandatory=True, desc="Output directory for the DKI metrics")
    bids_rename = traits.Bool(True, usedefault=True, desc="Rename outputs to BIDS style after fitting")
    overwrite = traits.Bool(False, usedefault=True, desc="Overwrite destination files if they already exist")
    num_threads = traits.Int(4, usedefault=True, desc="Number of threads for OpenMP/MKL/OpenBLAS")


class DKIFitOutputSpec(TraitedSpec):
    fa_file = File(desc="Path to the output FA map")
    md_file = File(desc="Path to the output MD map")
    ad_file = File(desc="Path to the output AD map")
    rd_file = File(desc="Path to the output RD map")
    mk_file = File(desc="Path to the output MK map")
    ak_file = File(desc="Path to the output AK map")
    rk_file = File(desc="Path to the output RK map")
    kxxxx_file = File(desc="Path to the output Kxxxx map")
    kyyyy_file = File(desc="Path to the output Kyyyy map")
    kzzzz_file = File(desc="Path to the output Kzzzz map")
    diffusion_tensor_file = File(desc="Path to the output diffusion tensor map")
    kurtosis_tensor_file = File(desc="Path to the output kurtosis tensor map")


class DKIFit(BaseInterface):
    """Fit Diffusion Kurtosis Imaging (DKI) model using Dipy and save scalar maps."""

    input_spec = DKIFitInputSpec
    output_spec = DKIFitOutputSpec

    @staticmethod
    def _call_metric(obj, name, **kwargs):
        """Call obj.<name>(**kwargs) if callable, else return attribute as-is."""
        if not hasattr(obj, name):
            return None
        attr = getattr(obj, name)
        if callable(attr):
            try:
                return attr(**kwargs)
            except TypeError:
                return attr()
        return attr

    @staticmethod
    def _save_nifti(data, ref_img, out_path):
        """Save a Nifti1Image using ref_img's affine, forcing float32."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        hdr = ref_img.header.copy()
        hdr.set_data_dtype(np.float32)
        img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), ref_img.affine, hdr)
        nib.save(img, out_path)

    def _ensure_outdir(self):
        os.makedirs(self.inputs.output_dir, exist_ok=True)

    def _bids_dest_files(self):
        from cvdproc.bids_data.rename_bids_file import rename_bids_file
        out_dir = os.path.abspath(self.inputs.output_dir)
        dest = {}
        for param in ["fa", "md", "ad", "rd", "mk", "ak", "rk",
                       "kxxxx", "kyyyy", "kzzzz", "tensor", "ktensor"]:
            fname = rename_bids_file(self.inputs.dwi_file,
                                     {"desc": None, "model": "dki", "param": param},
                                     "dwimap", ".nii.gz")
            dest[param] = os.path.join(out_dir, fname)
        return dest

    def _check_destination(self, path):
        if os.path.exists(path):
            if bool(self.inputs.overwrite):
                os.remove(path)
            else:
                raise FileExistsError(f"Destination exists: {path}")

    def _save_from_dki_fit(self, dkifit, ref_img, out_dir, mask, bids_dest):
        """Save all DKI-derived scalar maps. Uses bids_dest paths when bids_rename=True."""
        # 1) DTI-like metrics (from DKI fit)
        for name in ["fa", "md", "ad", "rd"]:
            arr = self._call_metric(dkifit, name)
            if arr is not None:
                dst = bids_dest[name]
                self._check_destination(dst)
                self._save_nifti(arr, ref_img, dst)

        # 2) Kurtosis scalars — use numerical estimators (analytical=False) to avoid Dipy bug
        for name in ["mk", "ak", "rk"]:
            arr = self._call_metric(dkifit, name, analytical=False)
            if arr is not None:
                dst = bids_dest[name]
                self._check_destination(dst)
                self._save_nifti(arr, ref_img, dst)

        # 3) Axis kurtosis along principal directions (Kxxxx, Kyyyy, Kzzzz)
        sphere = Sphere(xyz=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=float))
        K_axes = dkifit.akc(sphere)
        if K_axes.ndim == 2 and K_axes.shape[-1] == 3:
            vol = np.zeros(mask.shape + (3,), dtype=np.float32)
            vol[mask] = K_axes.astype(np.float32)
            K_axes = vol
        for i, (name, axis_label) in enumerate(zip(["kxxxx", "kyyyy", "kzzzz"], ["Kxxxx", "Kyyyy", "Kzzzz"])):
            dst = bids_dest[name]
            self._check_destination(dst)
            self._save_nifti(K_axes[..., i], ref_img, dst)

        # 4) Diffusion tensor 6 components (Voigt notation: Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
        D = self._call_metric(dkifit, "quadratic_form")
        if D is not None and D.ndim >= 2:
            D6 = np.stack([D[..., 0, 0], D[..., 1, 1], D[..., 2, 2],
                           D[..., 0, 1], D[..., 0, 2], D[..., 1, 2]], axis=-1)
            dst = bids_dest["tensor"]
            self._check_destination(dst)
            self._save_nifti(D6, ref_img, dst)

        # 5) Kurtosis tensor 15 components (Voigt notation)
        W15 = self._call_metric(dkifit, "kt")
        if W15 is not None:
            dst = bids_dest["ktensor"]
            self._check_destination(dst)
            self._save_nifti(W15.astype(np.float32), ref_img, dst)

    def _run_interface(self, runtime):
        self._ensure_outdir()
        out_dir = os.path.abspath(self.inputs.output_dir)

        nthreads = str(self.inputs.num_threads)
        os.environ["OMP_NUM_THREADS"] = nthreads
        os.environ["MKL_NUM_THREADS"] = nthreads
        os.environ["OPENBLAS_NUM_THREADS"] = nthreads
        os.environ["NUMEXPR_NUM_THREADS"] = nthreads

        if bool(self.inputs.bids_rename):
            bids_dest = self._bids_dest_files()
            # quick check: if a key output already exists, skip
            if os.path.exists(bids_dest["mk"]) and not bool(self.inputs.overwrite):
                print(f"[DKIFit] MK image already exists: {bids_dest['mk']}. Skipping DKI fit.")
                return runtime
        else:
            bids_dest = {}
            base = os.path.join(out_dir, "dki")
            for param in ["fa", "md", "ad", "rd", "mk", "ak", "rk",
                          "kxxxx", "kyyyy", "kzzzz", "tensor", "ktensor"]:
                bids_dest[param] = f"{base}_{param}.nii.gz"

        print(f"[DKIFit] Loading data from {self.inputs.dwi_file} ...")
        data, _ = load_nifti(self.inputs.dwi_file)
        bvals, bvecs = read_bvals_bvecs(self.inputs.bval_file, self.inputs.bvec_file)
        gtab = gradient_table(bvals, bvecs=bvecs)
        mask = nib.load(self.inputs.mask_file).get_fdata().astype(bool)
        ref_img = nib.load(self.inputs.dwi_file)
        print(f"[DKIFit] data: {data.shape}, mask: {mask.shape}, #bvals: {bvals.size}")

        assert data.shape[:3] == mask.shape, "Mask shape mismatch"
        assert data.shape[-1] == bvals.size, "Gradient length mismatch"

        data = np.ascontiguousarray(data.astype(np.float64))
        mask_arr = np.ascontiguousarray(mask)

        print("[DKIFit] Fitting DiffusionKurtosisModel (this may take a while)...")
        t0 = time.time()
        dkimodel = DiffusionKurtosisModel(gtab)
        dkifit = dkimodel.fit(data, mask=mask_arr)
        print(f"[DKIFit] DKI fit done in {time.time() - t0:.1f}s")

        print("[DKIFit] Saving outputs...")
        self._save_from_dki_fit(dkifit, ref_img, out_dir, mask_arr, bids_dest)

        # Save metadata
        meta = {
            "dipy_version": __import__("dipy").__version__,
            "numpy_version": np.__version__,
            "units": {"D": "mm^2/s", "K": "dimensionless"},
            "D6_order": ["Dxx", "Dyy", "Dzz", "Dxy", "Dxz", "Dyz"],
            "W15_voigt_order_hint": [
                "Wxxxx", "Wyyyy", "Wzzzz", "Wxxxy", "Wxxxz",
                "Wyyyx", "Wyyyz", "Wzzzx", "Wzzzy", "Wxxyy",
                "Wxxzz", "Wyyzz", "Wxyxy", "Wxzxz", "Wyzyz"
            ],
            "K_axes_dirs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }
        meta_path = os.path.join(out_dir, "dki_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print("[DKIFit] All done.")
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if bool(self.inputs.bids_rename):
            dst = self._bids_dest_files()
        else:
            base = os.path.join(os.path.abspath(self.inputs.output_dir), "dki")
            dst = {param: f"{base}_{param}.nii.gz" for param in [
                "fa", "md", "ad", "rd", "mk", "ak", "rk",
                "kxxxx", "kyyyy", "kzzzz", "tensor", "ktensor"
            ]}

        outputs["fa_file"] = dst["fa"]
        outputs["md_file"] = dst["md"]
        outputs["ad_file"] = dst["ad"]
        outputs["rd_file"] = dst["rd"]
        outputs["mk_file"] = dst["mk"]
        outputs["ak_file"] = dst["ak"]
        outputs["rk_file"] = dst["rk"]
        outputs["kxxxx_file"] = dst["kxxxx"]
        outputs["kyyyy_file"] = dst["kyyyy"]
        outputs["kzzzz_file"] = dst["kzzzz"]
        outputs["diffusion_tensor_file"] = dst["tensor"]
        outputs["kurtosis_tensor_file"] = dst["ktensor"]
        return outputs
