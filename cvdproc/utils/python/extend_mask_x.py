#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib


def extend_mask_x(
    mask_nii: str,
    out_nii: str,
    extend_part_nii: str,
    n_vox: int = 3,
    direction: str = "plus",
) -> None:
    img = nib.load(mask_nii)
    data = img.get_fdata() > 0

    out = data.copy()
    extend_part = np.zeros_like(data, dtype=bool)

    if direction not in ["plus", "minus"]:
        raise ValueError("direction must be 'plus' or 'minus'")

    for i in range(1, n_vox + 1):
        shifted = np.zeros_like(data, dtype=bool)
        if direction == "plus":
            # x plus: shift towards increasing x index
            shifted[i:, :, :] = data[:-i, :, :]
        else:
            # x minus: shift towards decreasing x index
            shifted[:-i, :, :] = data[i:, :, :]

        # record only newly added voxels
        new_part = shifted & (~out)
        extend_part |= new_part

        # update union mask
        out |= shifted

    # Save union mask
    out_img = nib.Nifti1Image(out.astype(np.uint8), img.affine, img.header)
    nib.save(out_img, out_nii)

    # Save extension-only mask
    extend_img = nib.Nifti1Image(extend_part.astype(np.uint8), img.affine, img.header)
    nib.save(extend_img, extend_part_nii)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_mask", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--extend_part", required=True)
    ap.add_argument("--n_vox", type=int, default=3)
    ap.add_argument("--direction", choices=["plus", "minus"], default="plus")
    args = ap.parse_args()

    extend_mask_x(
        mask_nii=args.in_mask,
        out_nii=args.out_mask,
        extend_part_nii=args.extend_part,
        n_vox=args.n_vox,
        direction=args.direction,
    )


if __name__ == "__main__":
    main()
