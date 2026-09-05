#!/usr/bin/env python3
"""Estimate PVeD with ANTs Atropos from MNI-space dtifit maps."""

import argparse
import json
import os
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output
from scipy import ndimage
from scipy.signal import find_peaks


def package_lv_mask():
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(package_dir, "data", "standard", "MNI152", "LV_mask_for_Fazekas.nii.gz")


def resample_to_2mm(image, is_label=False):
    return resample_to_output(image, voxel_sizes=(2.0, 2.0, 2.0), order=0 if is_label else 1)


def resample_lv_mask(md_2mm, lv_mask_path):
    lv_1mm = nib.load(lv_mask_path)
    lv_2mm = resample_from_to(lv_1mm, md_2mm, order=0)
    lv_binary = (lv_2mm.get_fdata() > 0.5).astype(np.float32)
    lv_image = nib.Nifti1Image(lv_binary, md_2mm.affine, md_2mm.header)
    print(f"  LV mask: {int(lv_binary.sum())} voxels")
    return lv_binary, lv_image


def create_brain_mask(fa_2mm, fa_threshold=0.12):
    mask = fa_2mm.get_fdata() > fa_threshold
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)), iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3, 3)), iterations=2)
    print(f"  Brain mask: {int(mask.sum())} voxels")
    return nib.Nifti1Image(mask.astype(np.uint8), fa_2mm.affine, fa_2mm.header)


def atropos_csf_segmentation(md_2mm, brain_mask, work_dir):
    md = np.clip(md_2mm.get_fdata(), 0.0, None)
    scaled_path = os.path.join(work_dir, "md_scaled.nii.gz")
    mask_path = os.path.join(work_dir, "brain_mask.nii.gz")
    segmentation_path = os.path.join(work_dir, "atropos_segmentation.nii.gz")
    probability_pattern = os.path.join(work_dir, "atropos_probability%02d.nii.gz")
    nib.save(nib.Nifti1Image((md * 10000.0).astype(np.float32), md_2mm.affine, md_2mm.header), scaled_path)
    nib.save(brain_mask, mask_path)
    subprocess.run([
        "Atropos", "-d", "3", "-a", scaled_path, "-x", mask_path, "-i", "KMeans[3]", "-m", "[0.2,1x1x1]", "-c", "[25,0.001]",
        "-o", f"[{segmentation_path},{probability_pattern}]",
    ], check=True)
    segmentation = nib.load(segmentation_path).get_fdata()
    class_means = [md[segmentation == label].mean() for label in (1, 2, 3)]
    csf_label = int(np.nanargmax(class_means)) + 1
    probability_path = probability_pattern.replace("%02d", f"{csf_label:02d}")
    csf_probability = nib.load(probability_path).get_fdata()
    csf_binary = (csf_probability > 0.3).astype(np.float32)
    print(f"  MD class means: {class_means}; CSF class: {csf_label}")
    print(f"  CSF mask: {int(csf_binary.sum())} voxels")
    return csf_binary


def compute_ttr(tensor_components):
    dxx, dyy, dzz = tensor_components[0], tensor_components[3], tensor_components[5]
    norm = np.sqrt(dxx ** 2 + dyy ** 2 + dzz ** 2)
    ttr = np.divide(dxx, norm, out=np.zeros_like(dxx), where=norm != 0)
    return np.nan_to_num(ttr, nan=0.0, posinf=0.0, neginf=0.0)


def create_pvr_mask_adapt(lv_binary):
    pvr_mask = np.zeros_like(lv_binary)
    lv_weight = lv_binary.sum(axis=0).sum(axis=0)
    if lv_weight.max() > lv_weight.min():
        lv_weight = (lv_weight - lv_weight.min()) / (lv_weight.max() - lv_weight.min())
    lv_weight = lv_weight ** 0.3
    nonzero = np.where(lv_weight > 0)[0]
    if len(nonzero) == 0:
        return pvr_mask
    midpoint = int(round(nonzero.min() + len(nonzero) / 2))
    x_values = np.linspace(-4, 4, len(nonzero)) + midpoint
    weights = np.zeros_like(lv_weight)
    weights[nonzero] = 1.0 / (1.0 + np.exp(-2 * (x_values - midpoint)))
    lv_weight *= weights

    lv_weight_y = lv_binary.sum(axis=0).sum(axis=1)
    if lv_weight_y.max() > lv_weight_y.min():
        lv_weight_y = (lv_weight_y - lv_weight_y.min()) / (lv_weight_y.max() - lv_weight_y.min())
    lv_weight_y = ndimage.gaussian_filter1d(lv_weight_y.astype(np.float64), sigma=2)
    peaks, _ = find_peaks(lv_weight_y, height=0.05)
    vec_y = np.zeros_like(lv_weight_y)
    if len(peaks) >= 2:
        top_two = np.argsort(lv_weight_y[peaks])[-2:]
        first, second = sorted([peaks[top_two[0]], peaks[top_two[1]]])
        vec_y[first:second] = 1.0
    else:
        nonzero_y = np.where(lv_weight_y > 0.1)[0]
        if len(nonzero_y) > 0:
            vec_y[nonzero_y.min():nonzero_y.max()] = 1.0
    vec_y = ndimage.gaussian_filter1d(vec_y.astype(np.float64), sigma=5)

    lv_bound = (lv_binary.max(axis=1).sum(axis=0) != 0).astype(int).sum()
    dim_x, dim_y, dim_z = lv_binary.shape
    for z in range(dim_z):
        image_slice = lv_binary[:, :, z]
        if image_slice.sum() == 0:
            continue
        dilated = np.zeros_like(image_slice)
        for y in range(dim_y):
            length = int(np.round(lv_bound * (lv_weight[z] ** 3) * vec_y[y]))
            length = min(max(length, 1), 20)
            column = image_slice[:, y].reshape(-1, 1)
            dilated[:, y] = ndimage.binary_dilation(column, structure=np.ones((length, 1)), iterations=1).astype(np.float64).ravel()

        half_x = dim_x // 2
        top, bottom = image_slice[:half_x, :], image_slice[half_x:, :]
        filter_top, filter_bottom = 1.0 - top, 1.0 - bottom
        for index in range(filter_top.shape[0]):
            if index > 0:
                filter_bottom[index, :] *= filter_bottom[index - 1, :]
        for index in range(filter_top.shape[0]):
            reverse_index = half_x - 1 - index
            if 0 <= reverse_index < half_x - 1:
                filter_top[reverse_index, :] *= filter_top[reverse_index + 1, :]
        anatomical_filter = 1.0 - np.concatenate([filter_top, filter_bottom], axis=0)
        pvr_mask[:, :, z] = (1.0 - image_slice) * dilated * anatomical_filter

    pvr_mask = np.nan_to_num(pvr_mask, nan=0.0, posinf=0.0, neginf=0.0)
    pvr_mask = (pvr_mask > 0).astype(np.float32)
    print(f"  PVR mask: {int(pvr_mask.sum())} voxels")
    return pvr_mask


def compute_pved(ttr, pvr_mask, csf_binary, lv_binary):
    final_mask = pvr_mask * (1.0 - csf_binary) * (1.0 - lv_binary)
    final_mask = final_mask > 0.5
    ttr_masked = ttr.copy()
    ttr_masked[~final_mask] = np.nan
    midpoint = ttr.shape[0] // 2
    pved_l = float(np.nanmedian(ttr_masked[midpoint:, :, :]))
    pved_r = float(np.nanmedian(ttr_masked[:midpoint, :, :]))
    pved_total = float(np.nanmedian(ttr_masked))
    dim_x, dim_y, dim_z = ttr.shape
    qa_region = ttr[int(dim_x * 0.25):int(dim_x * 0.75), int(dim_y * 0.25):int(dim_y * 0.75), int(dim_z * 0.45):int(dim_z * 0.55)]
    qa_values = qa_region[np.isfinite(qa_region)]
    qa_index = float(np.nanmean(qa_values))
    metrics = {
        "PVeD_L": round(pved_l, 6), "PVeD_R": round(pved_r, 6), "PVeD_total": round(pved_total, 6), "QA_index": round(qa_index, 6),
        "n_vox_final_mask": int(final_mask.sum()), "n_vox_pvr": int(pvr_mask.sum()), "n_vox_csf": int(csf_binary.sum()), "n_vox_lv": int(lv_binary.sum()),
    }
    return metrics, final_mask


def save_outputs(output_dir, subject, session, reference, md_2mm, ttr, lv_binary, csf_binary, pvr_mask, final_mask, metrics):
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{subject}_{session}_space-MNI152NLin6ASym_res-2mm"

    def save(array, suffix):
        image = nib.Nifti1Image(array.astype(np.float32), reference.affine, reference.header)
        nib.save(image, os.path.join(output_dir, f"{prefix}_{suffix}.nii.gz"))

    save(ttr, "param-ttr_dwimap")
    save(lv_binary, "label-LateralVentricle_mask")
    save(csf_binary, "label-CSF_desc-Atropos_mask")
    save(pvr_mask, "label-PeriventricularArea_mask")
    save(final_mask, "label-PeriventricularArea_desc-CSFExcluded_mask")
    nib.save(md_2mm, os.path.join(output_dir, f"{prefix}_model-tensor_param-md_dwimap.nii.gz"))
    with open(os.path.join(output_dir, "PVeD_metrics.csv"), "w", encoding="utf-8") as handle:
        handle.write("image_id,PVeD_L,PVeD_R,PVeD_total,QA_index\n")
        handle.write(f"{subject}_{session},{metrics['PVeD_L']},{metrics['PVeD_R']},{metrics['PVeD_total']},{metrics['QA_index']}\n")
    with open(os.path.join(output_dir, "PVeD_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fa", required=True)
    parser.add_argument("--md", required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    if shutil.which("Atropos") is None:
        raise RuntimeError("PVeD method='v2' requires the ANTs Atropos command to be available on PATH.")
    lv_mask = package_lv_mask()
    for path in (args.fa, args.md, args.tensor, lv_mask):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    started = time.time()
    fa_2mm = resample_to_2mm(nib.load(args.fa))
    md_native = nib.load(args.md)
    md_2mm = resample_to_2mm(md_native)
    tensor_image = nib.load(args.tensor)
    tensor_data = tensor_image.get_fdata()
    if tensor_data.ndim != 4 or tensor_data.shape[-1] != 6:
        raise ValueError(f"Expected a four-dimensional tensor image with six components; got {tensor_data.shape}")
    tensor_components = []
    os.makedirs(args.output_dir, exist_ok=True)
    for index in range(6):
        component = nib.Nifti1Image(tensor_data[..., index].astype(np.float32), tensor_image.affine, tensor_image.header)
        tensor_components.append(resample_to_2mm(component).get_fdata())
    lv_binary, _ = resample_lv_mask(md_2mm, lv_mask)
    work_dir = os.path.join(args.output_dir, "_pved_atropos_work")
    os.makedirs(work_dir, exist_ok=True)
    try:
        csf_binary = atropos_csf_segmentation(md_2mm, create_brain_mask(fa_2mm), work_dir)
        ttr = compute_ttr(tensor_components)
        pvr_mask = create_pvr_mask_adapt(lv_binary)
        metrics, final_mask = compute_pved(ttr, pvr_mask, csf_binary, lv_binary)
        save_outputs(args.output_dir, args.subject, args.session, md_2mm, md_2mm, ttr, lv_binary, csf_binary, pvr_mask, final_mask, metrics)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    print(json.dumps(metrics, indent=2))
    print(f"PVeD completed in {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
