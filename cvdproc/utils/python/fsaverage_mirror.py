import nibabel as nib
import numpy as np

def copy_lh_to_rh_with_rh_header(lh_mgh_path, rh_mgh_path, output_path_rh):
    # Load left hemisphere (for data)
    lh_img = nib.load(lh_mgh_path)
    lh_data = lh_img.get_fdata().squeeze()

    # Load right hemisphere (for header and shape reference)
    rh_img = nib.load(rh_mgh_path)
    rh_data = rh_img.get_fdata().squeeze()

    # Sanity check: shape match
    if lh_data.shape != rh_data.shape:
        raise ValueError(f"Data shape mismatch: LH {lh_data.shape} vs RH {rh_data.shape}")

    # Expand dimensions to match original .mgh shape if needed
    # .mgh expects shape (N, 1, 1) rather than (N,)
    new_data = lh_data.astype(np.float32)
    while new_data.ndim < 3:
        new_data = np.expand_dims(new_data, axis=-1)

    # Save using RH affine + header
    new_rh_img = nib.MGHImage(new_data, rh_img.affine, rh_img.header)
    nib.save(new_rh_img, output_path_rh)

    print(f"Saved mirrored RH file (data from LH, header from RH) to: {output_path_rh}")

if __name__ == "__main__":
    # Example usage
    lh_mgh_path = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0095/ses-01/probtrackx_output_PathLengthCorrected/lh.HighConn_fsaverage_sym.mgh"
    rh_mgh_path = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0095/ses-01/probtrackx_output_PathLengthCorrected/rh.HighConn_fsaverage_sym.mgh"
    output_path_rh = "/mnt/f/BIDS/SVD_BIDS/derivatives/fdt/sub-SVD0095/ses-01/probtrackx_output_PathLengthCorrected/rh.HighConn_fsaverage_sym_mirrored.mgh"

    copy_lh_to_rh_with_rh_header(lh_mgh_path, rh_mgh_path, output_path_rh)