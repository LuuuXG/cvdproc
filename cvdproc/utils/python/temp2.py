import os
import numpy as np
import nibabel as nib


dsi_file = "/mnt/e/Neuroimage/workdir/fix/sub-TI001_ses-6m_acq-DSIb4000_dir-AP_space-preprocdwi_dwiref.nii.gz"
ref_file = "/mnt/e/Neuroimage/workdir/fix/b0_ref.nii.gz"
out_file = "/mnt/e/Neuroimage/workdir/fix/sub-TI001_ses-6m_acq-DSIb4000_dir-AP_space-preprocdwi_dwiref_physfixed.nii.gz"


def main():
    dsi_img = nib.load(dsi_file)
    ref_img = nib.load(ref_file)

    dsi_data = np.asanyarray(dsi_img.dataobj)

    if dsi_img.shape != ref_img.shape:
        raise ValueError(
            f"Shape mismatch: DSI Studio image shape = {dsi_img.shape}, "
            f"reference image shape = {ref_img.shape}"
        )

    # Flip x and y axes
    fixed_data = dsi_data[:, ::-1, :]

    header = ref_img.header.copy()
    header.set_data_shape(fixed_data.shape)
    header.set_data_dtype(fixed_data.dtype)

    fixed_img = nib.Nifti1Image(
        fixed_data,
        affine=ref_img.affine,
        header=header,
    )

    fixed_img.set_qform(ref_img.affine, code=1)
    fixed_img.set_sform(ref_img.affine, code=1)

    nib.save(fixed_img, out_file)

    print("Saved fixed image:")
    print(out_file)


if __name__ == "__main__":
    main()