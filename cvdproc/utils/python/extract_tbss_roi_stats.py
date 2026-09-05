#!/usr/bin/env python3
"""Extract per-ROI mean values from TBSS/GBSS skeletonised 4D maps.

Usage:
    python extract_tbss_roi_stats.py <tbss_output_dir> [--output_dir <dir>] [--maps TBSS|GBSS|all]
"""

import os, sys, csv, argparse, time, xml.etree.ElementTree as ET
import nibabel as nib
import numpy as np

# --- Atlas definitions ---

AAL_NII = "/mnt/e/codes/cvdproc/cvdproc/data/atlas/AAL_v4/ROI_MNI_V4_1mm_MNI152.nii"
AAL_TXT = "/mnt/e/codes/cvdproc/cvdproc/data/atlas/AAL_v4/ROI_MNI_V4.txt"

JHU_LABELS_NII = "/usr/local/fsl/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz"
JHU_LABELS_XML = "/usr/local/fsl/data/atlases/JHU-labels.xml"
JHU_TRACTS_NII = "/usr/local/fsl/data/atlases/JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz"
JHU_TRACTS_XML = "/usr/local/fsl/data/atlases/JHU-tracts.xml"

TBSS_MAPS = {
    "FA": "merged_4d/FA/all_FA_skeletonised.nii.gz",
    "MD": "merged_4d/MD/all_MD_skeletonised.nii.gz",
    "NDI": "merged_4d/NDI/all_NDI_skeletonised.nii.gz",
    "ODI": "merged_4d/ODI/all_ODI_skeletonised.nii.gz",
    "ISOVF": "merged_4d/ISOVF/all_ISOVF_skeletonised.nii.gz",
}

GBSS_MAPS = {
    "FA": "merged_4d/FA/all_FA_skeletonised_GBSS.nii.gz",
    "MD": "merged_4d/MD/all_MD_skeletonised_GBSS.nii.gz",
    "NDI": "merged_4d/NDI/all_NDI_skeletonised_GBSS.nii.gz",
    "ODI": "merged_4d/ODI/all_ODI_skeletonised_GBSS.nii.gz",
    "ISOVF": "merged_4d/ISOVF/all_ISOVF_skeletonised_GBSS.nii.gz",
    "GM_fraction": "merged_4d/GM_fraction/all_GM_fraction_skeletonised_GBSS.nii.gz",
}


def load_aal_labels():
    """Parse AAL_v4 label file. Format: abbreviation<TAB>name<TAB>index"""
    labels = {}
    with open(AAL_TXT, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                idx = int(parts[2])
                labels[idx] = parts[1]
    return labels


def load_jhu_xml_labels(xml_path):
    """Parse JHU XML label definitions. Returns {index: name}."""
    labels = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for label_elem in root.findall(".//label"):
        name = label_elem.text.strip() if label_elem.text else ""
        idx_str = label_elem.get("index")
        if idx_str is not None:
            labels[int(idx_str)] = name
    if not labels:
        for i, label_elem in enumerate(root.findall(".//label")):
            name = label_elem.text.strip() if label_elem.text else ""
            labels[i] = name
    return labels


def _make_3d_ref(ref_img):
    """Extract a 3D spatial reference image from a (potentially 4D) reference.

    The 4D skeletonised files have a 5x5 affine; resample_from_to requires
    a proper 3D image with a 4x4 affine.
    """
    if ref_img.ndim < 4:
        return ref_img
    # Build a 3D reference from the spatial portion of the 4D image
    if hasattr(ref_img, 'slicer'):
        ref_3d_img = ref_img.slicer[..., 0]
        # slicer returns an image with proper 4x4 affine
        if ref_3d_img.affine.shape == (4, 4):
            return ref_3d_img
    # Fallback: manually construct a 3D Nifti1Image
    dummy_data = np.zeros(ref_img.shape[:3], dtype=np.int16)
    affine_4x4 = np.eye(4)
    affine_4x4[:3, :3] = ref_img.affine[:3, :3]
    affine_4x4[:3, 3] = ref_img.affine[:3, 3]
    return nib.Nifti1Image(dummy_data, affine_4x4)


def resample_atlas_to_ref(atlas_nii_path, ref_img):
    """Resample atlas to match reference image grid using affine-aware resampling.

    Uses nibabel's resample_from_to which properly accounts for the affine
    transformation between the two images (origin, orientation, voxel size).
    Falls back to scipy.ndimage.zoom if resample_from_to is unavailable.
    """
    atlas_img = nib.load(atlas_nii_path)
    ref_3d = _make_3d_ref(ref_img)
    atlas_shape = atlas_img.shape[:3]
    ref_shape = ref_3d.shape[:3]

    # Quick check: same grid and same affine -> no resampling needed
    if atlas_shape == ref_shape and np.allclose(atlas_img.affine, ref_3d.affine, atol=0.01):
        return atlas_img.get_fdata().astype(np.int32)

    try:
        from nibabel.processing import resample_from_to
        print(f"    Resampling atlas {atlas_shape} -> {ref_shape} (affine-aware)...", flush=True)
        resampled_img = resample_from_to(atlas_img, ref_3d, order=0, mode='constant', cval=0)
        return resampled_img.get_fdata().astype(np.int32)
    except ImportError:
        pass

    # Fallback: simple zoom (only correct when images differ only in resolution,
    # not in origin or orientation)
    print(f"    WARNING: nibabel.processing.resample_from_to not available, using simple zoom", flush=True)
    print(f"    Atlas affine: {atlas_img.affine.diagonal()[:3]}", flush=True)
    print(f"    Ref affine:   {ref_3d.affine.diagonal()[:3]}", flush=True)
    from scipy.ndimage import zoom as scipy_zoom
    atlas_data = atlas_img.get_fdata().astype(np.float64)
    zoom_factors = [r / a for r, a in zip(ref_shape, atlas_shape)]
    resampled = scipy_zoom(atlas_data, zoom_factors, order=0)
    return np.round(resampled).astype(np.int32)


# Cache for resampled atlases keyed by (atlas_path, ref_shape_hash, ref_affine_hash)
_atlas_cache = {}


def get_resampled_atlas(atlas_nii_path, ref_img):
    """Return resampled atlas data, using cache to avoid repeated resampling."""
    key = (atlas_nii_path, ref_img.shape[:3], ref_img.affine.tobytes())
    if key not in _atlas_cache:
        _atlas_cache[key] = resample_atlas_to_ref(atlas_nii_path, ref_img)
    return _atlas_cache[key]


def extract_roi_means(skel_4d_path, atlas_data, atlas_labels, subject_ids):
    """Extract per-ROI per-subject mean values from a skeletonised 4D file.

    Loads the entire 4D array into memory for performance (sequential gzip
    read is much faster than batched random-access slices).
    """
    if not os.path.exists(skel_4d_path):
        print(f"  SKIP: {skel_4d_path} not found", flush=True)
        return []

    t_start = time.time()
    skel_img = nib.load(skel_4d_path)
    n_subjects = skel_img.shape[3]
    print(f"    Loading 4D data ({skel_img.shape}, {skel_img.get_data_dtype()})...", flush=True)

    if n_subjects != len(subject_ids):
        print(f"    WARNING: n_subjects={n_subjects} but {len(subject_ids)} IDs; truncating", flush=True)
        subject_ids = subject_ids[:n_subjects]

    full_data = np.asarray(skel_img.dataobj, dtype=np.float32)
    if full_data.ndim == 3:
        full_data = full_data[..., np.newaxis]
    print(f"    Loaded in {time.time() - t_start:.1f}s, memory={full_data.nbytes / 1e9:.2f} GB", flush=True)

    # Build ROI masks
    roi_indices = sorted(set(atlas_data.flatten()))
    roi_indices = [r for r in roi_indices if r > 0 and r in atlas_labels]
    if not roi_indices:
        print(f"    WARNING: no valid ROI labels found", flush=True)
        return []

    masks = {}
    for roi_idx in roi_indices:
        m = (atlas_data == roi_idx)
        if m.sum() > 0:
            masks[roi_idx] = m
    roi_list = sorted(masks.keys())
    n_rois = len(roi_list)
    if n_rois == 0:
        return []

    # Pre-compute ROI means for all subjects at once
    roi_col = {r: i for i, r in enumerate(roi_list)}
    all_means = np.full((n_subjects, n_rois), np.nan, dtype=np.float32)

    for roi_idx in roi_list:
        col = roi_col[roi_idx]
        all_means[:, col] = full_data[masks[roi_idx], :].mean(axis=0)

    # Build results
    results = []
    for t in range(n_subjects):
        for roi_idx in roi_list:
            mv = all_means[t, roi_col[roi_idx]]
            results.append({
                "subject_id": subject_ids[t],
                "roi_name": atlas_labels[roi_idx],
                "roi_index": roi_idx,
                "value": float(mv) if not np.isnan(mv) else 0.0,
            })
    print(f"    Extracted {len(roi_list)} ROIs x {n_subjects} subjects in {time.time() - t_start:.1f}s", flush=True)
    return results


def load_subject_order(tbss_dir):
    """Load subject order from merge_dir."""
    order_file = os.path.join(tbss_dir, "merged_4d", "subject_session_order.txt")
    if not os.path.exists(order_file):
        order_file = os.path.join(tbss_dir, "logs", "common_ids.txt")
    if not os.path.exists(order_file):
        print("ERROR: cannot find subject order file")
        sys.exit(1)
    with open(order_file) as f:
        return [line.strip() for line in f if line.strip()]


def process_atlas(tbss_dir, atlas_name, atlas_nii, atlas_labels, ref_img, map_dict, subject_ids, out_dir, analysis_label):
    """Process one atlas against all maps."""
    print(f"\n{'='*60}")
    print(f"Atlas: {atlas_name} ({analysis_label})")
    print(f"{'='*60}")

    atlas_data = get_resampled_atlas(atlas_nii, ref_img)
    n_unique = len(set(atlas_data.flatten())) - 1
    print(f"  Atlas shape: {atlas_data.shape}, unique ROIs (non-zero): {n_unique}")

    for map_name, map_relpath in map_dict.items():
        map_path = os.path.join(tbss_dir, map_relpath)

        # Skip if output already exists
        out_csv = os.path.join(out_dir, f"{analysis_label}_{atlas_name}_{map_name}.csv")
        if os.path.exists(out_csv):
            print(f"  {map_name}: SKIP (already exists: {out_csv})")
            continue

        print(f"  Extracting {map_name}...")
        results = extract_roi_means(map_path, atlas_data, atlas_labels, subject_ids)
        if not results:
            print(f"    No results for {map_name}, skipping")
            continue

        pivot = {}
        roi_order = []
        for r in results:
            key = f"{r['roi_name']}__{r['roi_index']}"
            if key not in pivot:
                pivot[key] = {}
                roi_order.append(key)
            pivot[key][r['subject_id']] = r['value']

        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject_id"] + [k.split("__")[0] for k in roi_order])
            for sid in subject_ids:
                writer.writerow([sid] + [pivot[k].get(sid, "") for k in roi_order])
        print(f"    -> {out_csv} ({len(subject_ids)} subjects x {len(roi_order)} ROIs)")

        out_long = out_csv.replace(".csv", "_long.csv")
        with open(out_long, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject_id", "roi_name", "roi_index", "value"])
            for r in results:
                writer.writerow([r["subject_id"], r["roi_name"], r["roi_index"], r["value"]])
        print(f"    -> {out_long} (long format)")


def main():
    parser = argparse.ArgumentParser(description="Extract per-ROI mean values from TBSS/GBSS skeletonised maps")
    parser.add_argument("tbss_dir", help="Path to tbss_gbss output directory")
    parser.add_argument("--output_dir", "-o", default=None, help="Output directory for CSVs (default: tbss_dir/roi_stats)")
    parser.add_argument("--maps", default="all", choices=["TBSS", "GBSS", "all"], help="Which skeleton maps to process")
    args = parser.parse_args()

    tbss_dir = args.tbss_dir
    out_dir = args.output_dir or os.path.join(tbss_dir, "roi_stats")
    os.makedirs(out_dir, exist_ok=True)

    subject_ids = load_subject_order(tbss_dir)
    print(f"Subjects: {len(subject_ids)}")

    aal_labels = load_aal_labels()
    print(f"AAL labels: {len(aal_labels)}")
    jhu_label_dict = load_jhu_xml_labels(JHU_LABELS_XML)
    print(f"JHU labels: {len(jhu_label_dict)}")
    jhu_tract_dict = load_jhu_xml_labels(JHU_TRACTS_XML)
    print(f"JHU tracts labels: {len(jhu_tract_dict)}")

    atlases = [
        ("AAL", AAL_NII, aal_labels),
        ("JHU_labels", JHU_LABELS_NII, jhu_label_dict),
        ("JHU_tracts", JHU_TRACTS_NII, jhu_tract_dict),
    ]

    # --- TBSS ---
    if args.maps in ("TBSS", "all"):
        fa_tbss_path = os.path.join(tbss_dir, TBSS_MAPS["FA"])
        if os.path.exists(fa_tbss_path):
            ref_img = nib.load(fa_tbss_path)
            print(f"\n{'='*60}")
            print(f"TBSS (WM Skeleton) - ref shape={ref_img.shape[:3]}, voxel={ref_img.header.get_zooms()[:3]}")
            print(f"{'='*60}")
            for atlas_name, atlas_nii, atlas_labels in atlases:
                process_atlas(tbss_dir, atlas_name, atlas_nii, atlas_labels, ref_img, TBSS_MAPS, subject_ids, out_dir, "TBSS")
        else:
            print(f"TBSS FA skeleton not found: {fa_tbss_path}")

    # --- GBSS ---
    if args.maps in ("GBSS", "all"):
        gbss_ref_rel = GBSS_MAPS.get("GM_fraction")
        if gbss_ref_rel:
            gbss_ref_path = os.path.join(tbss_dir, gbss_ref_rel)
            if os.path.exists(gbss_ref_path):
                ref_img_gbss = nib.load(gbss_ref_path)
                print(f"\n{'='*60}")
                print(f"GBSS (GM Skeleton) - ref shape={ref_img_gbss.shape[:3]}, voxel={ref_img_gbss.header.get_zooms()[:3]}")
                print(f"{'='*60}")
                for atlas_name, atlas_nii, atlas_labels in atlases:
                    process_atlas(tbss_dir, atlas_name, atlas_nii, atlas_labels, ref_img_gbss, GBSS_MAPS, subject_ids, out_dir, "GBSS")
            else:
                print(f"GBSS reference not found: {gbss_ref_path}")

    print(f"\nDone. All CSVs saved in: {out_dir}")


if __name__ == "__main__":
    main()
