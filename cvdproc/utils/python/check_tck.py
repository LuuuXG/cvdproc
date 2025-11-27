import nibabel as nib
import numpy as np
from nibabel.streamlines import load, Tractogram, save

def _apply_axis_flips_mm(streamlines, img_affine, img_shape,
                         flip_x=False, flip_y=False, flip_z=False):
    """
    Apply optional left-right (X), anterior-posterior (Y), and
    inferior-superior (Z) flips in mm space, using the image bounding box.

    The reflection is applied about the center of the bounding box
    for each axis, so coordinates remain inside the image extent.
    """

    # Compute 8 corners of the image in voxel space
    nx, ny, nz = img_shape[:3]
    corners_ijk = np.array(
        [
            [0, 0, 0],
            [nx - 1, 0, 0],
            [0, ny - 1, 0],
            [0, 0, nz - 1],
            [nx - 1, ny - 1, 0],
            [nx - 1, 0, nz - 1],
            [0, ny - 1, nz - 1],
            [nx - 1, ny - 1, nz - 1],
        ],
        dtype=float,
    )

    # Convert voxel corners to mm using the image affine
    corners_mm = nib.affines.apply_affine(img_affine, corners_ijk)

    xmin, ymin, zmin = np.min(corners_mm, axis=0)
    xmax, ymax, zmax = np.max(corners_mm, axis=0)

    # Reflection about the middle plane of the bounding box
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    flipped_streamlines = []
    for s in streamlines:
        if s.size == 0:
            flipped_streamlines.append(s)
            continue

        coords = s.copy()
        if flip_x:
            coords[:, 0] = 2 * cx - coords[:, 0]
        if flip_y:
            coords[:, 1] = 2 * cy - coords[:, 1]
        if flip_z:
            coords[:, 2] = 2 * cz - coords[:, 2]

        flipped_streamlines.append(coords)

    return flipped_streamlines


def convert_tck_to_dsi_mm_space(
    tck_path,
    fa1_path,
    fa_dsi_path,
    output_path,
    flip_x=False,
    flip_y=False,
    flip_z=False,
):
    """
    Convert a TCK (in FA1 RAS mm space) to DSI Studio-compatible mm space
    based on FA1 and DSI FA images.

    Parameters
    ----------
    tck_path : str
        Path to input TCK file (streamlines in FA1 RAS mm space).
    fa1_path : str
        Path to FA image used in tractography (same space as TCK).
    fa_dsi_path : str
        Path to DSI Studio FA image (target space).
    output_path : str
        Path to output TCK file in DSI-compatible space.
    flip_x : bool, optional
        If True, apply left-right flip in mm space.
    flip_y : bool, optional
        If True, apply anterior-posterior flip in mm space.
    flip_z : bool, optional
        If True, apply inferior-superior flip in mm space.
    """

    # Load TCK streamlines (assumed to be in FA1 RAS mm space)
    tck = load(tck_path)
    streamlines = list(tck.streamlines)

    # Load FA images
    fa1_img = nib.load(fa1_path)
    fa1_affine = fa1_img.affine
    fa1_affine_inv = np.linalg.inv(fa1_affine)

    fa_dsi_img = nib.load(fa_dsi_path)
    fa_dsi_affine = fa_dsi_img.affine
    fa_dsi_header = fa_dsi_img.header
    fa_dsi_voxel_size = fa_dsi_header.get_zooms()[:3]
    fa_dsi_shape = fa_dsi_img.shape

    # Step 1: convert RAS mm (FA1 space) -> voxel (FA1 space)
    streamlines_fa1_voxel = [
        nib.affines.apply_affine(fa1_affine_inv, s) for s in streamlines
    ]

    # Step 2: convert FA1 voxel -> DSI mm (using FA_DSI affine)
    streamlines_dsi_mm = [
        nib.affines.apply_affine(fa_dsi_affine, s_vox)
        for s_vox in streamlines_fa1_voxel
    ]

    # Step 3: fix Z scaling (DSI Studio assumes isotropic spacing)
    scale_factor = fa_dsi_voxel_size[2] / fa_dsi_voxel_size[0]
    print(f"Z scaling factor applied: {scale_factor:.3f}")

    streamlines_scaled = [
        s * np.array([1.0, 1.0, scale_factor], dtype=float)
        for s in streamlines_dsi_mm
    ]

    # Step 4: optional axis flips to match DSI Studio orientation
    if flip_x or flip_y or flip_z:
        streamlines_final = _apply_axis_flips_mm(
            streamlines_scaled,
            img_affine=fa_dsi_affine,
            img_shape=fa_dsi_shape,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
        )
    else:
        streamlines_final = streamlines_scaled

    # Save final TCK in DSI-compatible space (coordinates already in mm)
    tractogram = Tractogram(streamlines_final, affine_to_rasmm=np.eye(4))
    save(tractogram, output_path)

    print(f"Saved transformed TCK to: {output_path}")
    print(f"Axis flips applied: flip_x={flip_x}, flip_y={flip_y}, flip_z={flip_z}")

# Example usage:
convert_tck_to_dsi_mm_space(
    tck_path="/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/dwi_pipeline/sub-SSI0248/ses-baseline/mrtrix3/tracks_1M.tck",
    fa1_path="/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/dwi_pipeline/sub-SSI0248/ses-baseline/dtifit/sub-SSI0248_ses-baseline_acq-NODDIb2500_dir-AP_space-preprocdwi_model-tensor_param-fa_dwimap.nii.gz",
    fa_dsi_path="/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/dwi_pipeline/sub-SSI0248/ses-baseline/dsistudio/sub-SSI0248_ses-baseline_acq-NODDIb2500_dir-AP_space-preprocdwi_desc-preproc_dwi_dti_fa.nii.gz",
    output_path="/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/dwi_pipeline/sub-SSI0248/ses-baseline/mrtrix3/tracked_in_dsi_space.tck",
    flip_x=False,
    flip_y=True,
    flip_z=False
)
