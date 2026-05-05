#!/usr/bin/env python3
import argparse
import os
import SimpleITK as sitk


def invert_displacement_h5(
    forward_h5: str,
    inverse_reference_image: str,
    output_h5: str,
    subsampling_factor: int = 16,
):
    if not os.path.isfile(forward_h5):
        raise FileNotFoundError(f"Forward transform not found: {forward_h5}")
    if not os.path.isfile(inverse_reference_image):
        raise FileNotFoundError(f"Inverse reference image not found: {inverse_reference_image}")

    print(f"Reading forward transform: {forward_h5}")
    tx = sitk.ReadTransform(forward_h5)

    tx_name = tx.GetName()
    print(f"Transform type: {tx_name}")
    if tx_name != "DisplacementFieldTransform":
        raise ValueError(f"Expected a DisplacementFieldTransform, got: {tx_name}")

    print("Extracting displacement field from forward transform...")
    forward_field = tx.GetDisplacementField()

    print(f"Reading inverse reference image: {inverse_reference_image}")
    ref_img = sitk.ReadImage(inverse_reference_image)

    print("Computing inverse displacement field...")

    inv_filter = sitk.InverseDisplacementFieldImageFilter()
    inv_filter.SetReferenceImage(ref_img)
    inv_filter.SetSubsamplingFactor(subsampling_factor)

    inv_field = inv_filter.Execute(forward_field)

    print("Wrapping inverse field as a displacement transform...")
    inv_tx = sitk.DisplacementFieldTransform(inv_field)

    print(f"Writing inverse transform: {output_h5}")
    sitk.WriteTransform(inv_tx, output_h5)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Invert a validated forward displacement H5 transform."
    )
    parser.add_argument("--forward_h5", required=True, help="Validated forward displacement transform (.h5)")
    parser.add_argument("--inverse_reference", required=True, help="Reference image defining inverse transform domain")
    parser.add_argument("--output_h5", required=True, help="Output inverse displacement transform (.h5)")
    parser.add_argument("--subsampling_factor", type=int, default=16, help="Subsampling factor (default: 16)")
    args = parser.parse_args()

    invert_displacement_h5(
        forward_h5=args.forward_h5,
        inverse_reference_image=args.inverse_reference,
        output_h5=args.output_h5,
        subsampling_factor=args.subsampling_factor,
    )


if __name__ == "__main__":
    main()