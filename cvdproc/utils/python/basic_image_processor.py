import os
import argparse
import nibabel as nib
import numpy as np
import csv
from nipype.utils.filemanip import fname_presuffix

############################
# Image Loading and Saving #
############################

def load_image(image_path):
    """Load image data based on file extension."""
    ext = os.path.splitext(image_path)[-1].lower()
    if ext in ['.nii', '.gz', '.mgh']:
        img = nib.load(image_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return img

def save_image(data, affine, header, output_path):
    """Save image data to file with float32 type."""
    ext = os.path.splitext(output_path)[-1].lower()
    data = data.astype(np.float32)
    if ext in ['.nii', '.gz']:
        img_to_save = nib.Nifti1Image(data, affine, header)
    elif ext == '.mgh':
        img_to_save = nib.MGHImage(data, affine, header)
    else:
        raise ValueError(f"Unsupported output file format: {ext}")
    nib.save(img_to_save, output_path)

##################################################
# Functions for handling single image processing #
##################################################

def threshold_binarize_image(image_path, threshold, output_path):
    img = load_image(image_path)
    data = img.get_fdata()
    binarized_data = (data > threshold).astype(np.float32)

    if output_path is None:
        base_name, ext = os.path.splitext(os.path.basename(image_path))
        folder = os.path.dirname(image_path) or "."
        threshold_str = f"{threshold:.2e}".replace(".", "p").replace("+", "")
        output_filename = f"binarized_thr{threshold_str}_{base_name}{ext}"
        output_path = os.path.join(folder, output_filename)

    save_image(binarized_data, img.affine, img.header, output_path)
    print(f"Binarized image saved to {output_path}")

def mask_image(input_image, mask_image, binarize_mask, output_path):
    img = load_image(input_image)
    mask = load_image(mask_image)
    mask_data = mask.get_fdata()

    if binarize_mask:
        mask_data = (mask_data > 0).astype(np.float32)
    else:
        if not np.all(np.isclose(mask_data, 0) | np.isclose(mask_data, 1)):
            raise ValueError("Mask data must be binary (0 and 1). Use --binarize to force binarization.")
        mask_data = mask_data.astype(np.float32)

    if img.shape != mask.shape:
        raise ValueError("Input image and mask must have the same dimensions.")

    masked_data = img.get_fdata() * mask_data

    if output_path is None:
        base_name, ext = os.path.splitext(os.path.basename(input_image))
        folder = os.path.dirname(input_image) or "."
        output_filename = f"masked_{base_name}{ext}"
        output_path = os.path.join(folder, output_filename)

    save_image(masked_data, img.affine, img.header, output_path)
    print(f"Masked image saved to {output_path}")

def extract_roi_from_image(input_image, roi_list, binarize, output_path):
    img = load_image(input_image)
    img_data = img.get_fdata()

    roi_mask = np.isin(img_data, roi_list)
    new_data = np.zeros(img_data.shape, dtype=img_data.dtype)

    if binarize:
        new_data[roi_mask] = 1
    else:
        new_data[roi_mask] = img_data[roi_mask]
    
    if output_path is None:
        base_name, ext = os.path.splitext(os.path.basename(input_image))
        folder = os.path.dirname(input_image) or "."
        output_filename = f"roi_{base_name}{ext}"
        output_path = os.path.join(folder, output_filename)
    
    save_image(new_data, img.affine, img.header, output_path)
    print(f"Extracted ROI saved to {output_path}")

    return output_path

def extract_roi_means(input_image, roi_image, ignore_background, output_path):
    img = load_image(input_image)
    roi = load_image(roi_image)

    if img.shape != roi.shape:
        raise ValueError("Input image and ROI mask must have the same dimensions.")

    img_data = img.get_fdata()
    roi_data = roi.get_fdata().astype(int)
    roi_labels = np.unique(roi_data)

    roi_means = []
    for label in roi_labels:
        mask = roi_data == label
        if ignore_background:
            mask &= img_data != 0
        mean_val = img_data[mask].mean() if np.any(mask) else np.nan
        roi_means.append([label, mean_val])

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_image))[0]
        folder = os.path.dirname(input_image) or "."
        output_filename = f"roi_means_{base_name}.csv"
        output_path = os.path.join(folder, output_filename)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ROI', 'MeanValue'])
        for label, mean_val in roi_means:
            writer.writerow([label, mean_val])

    print(f"ROI means saved to {output_path}")

def change_dtype(input_image, target_dtype, output_path):
    img = load_image(input_image)
    data = img.get_fdata().astype(np.dtype(target_dtype))

    if output_path is None:
        base_name, ext = os.path.splitext(os.path.basename(input_image))
        folder = os.path.dirname(input_image) or "."
        output_filename = f"{base_name}_{target_dtype}.nii.gz"
        output_path = os.path.join(folder, output_filename)

    save_image(data, img.affine, img.header, output_path)
    print(f"Image saved to {output_path}")

def change_affine(file_1, file_2, output_path, axis=None):
    """
    Update the affine of file_2 to match file_1, with optional flipping of the image data.

    Args:
        file_1: Path to the reference NIfTI file.
        file_2: Path to the NIfTI file to be updated.
        output_path: Path to save the updated file.
        flip: Whether to flip the image data along a specified axis.
        axis: Axis along which to flip the image data (0, 1, or 2 for x, y, z).
    """
    nii_1 = nib.load(file_1)
    nii_2 = nib.load(file_2)

    # Prepare new file name
    orig_file_name = os.path.basename(file_2)
    if orig_file_name.endswith('.gz'):
        orig_file_name = orig_file_name[:-7]
        orig_file_ext = '.nii.gz'
    else:
        orig_file_name = orig_file_name[:-4]
        orig_file_ext = '.nii'

    root_folder = os.path.dirname(file_2)
    if output_path is None:
        output_path = os.path.join(root_folder, f"newaffine_{orig_file_name}{orig_file_ext}")

    # Get original data
    img_data = nii_2.get_fdata()

    # Optionally flip the image data
    ## NOTE: I am not sure if this is the correct way to update the affine matrix
    if axis is not None:
        if axis not in [0, 1, 2]:
            raise ValueError("Invalid axis value. Must be 0, 1, or 2 for x, y, z.")
        print(f"Flipping the image along axis {axis}")
        img_data = np.flip(img_data, axis=axis)
        # Update the affine matrix to reflect the flip
        flip_affine = np.eye(4)
        flip_affine[axis, axis] = -1
        flip_affine[axis, 3] = nii_2.shape[axis] - 1
        new_affine = nii_1.affine @ flip_affine
    else:
        new_affine = nii_1.affine

    # Update affine matrix and ensure qform/sform codes match
    nii_2_updated = nib.Nifti1Image(img_data, affine=new_affine, header=nii_2.header)
    nii_2_updated.header.set_qform(new_affine, code=1)
    nii_2_updated.header.set_sform(new_affine, code=1)

    # Save the new file
    nib.save(nii_2_updated, output_path)
    print(f"Updated file saved at: {output_path}")

# Fix affine (similar to change_affine)
# QSIPrep
def fix_affine(image_to_fix, correct_image, output_path=None):
    if output_path is None:
        output_path = fname_presuffix(image_to_fix, suffix="fixhdr", newpath=".")

    image_to_fix_nii = nib.load(image_to_fix)
    correct_img_nii = nib.load(correct_image)

    new_axcodes = nib.aff2axcodes(correct_img_nii.affine)
    input_axcodes = nib.aff2axcodes(image_to_fix_nii.affine)

    # Is the input image oriented how we want?
    if not input_axcodes == new_axcodes:
        # Re-orient
        input_orientation = nib.orientations.axcodes2ornt(input_axcodes)
        desired_orientation = nib.orientations.axcodes2ornt(new_axcodes)
        transform_orientation = nib.orientations.ornt_transform(
            input_orientation, desired_orientation
        )
        reoriented_img = image_to_fix_nii.as_reoriented(transform_orientation)

    else:
        reoriented_img = image_to_fix_nii

    # No matter what, still use the correct affine
    nib.Nifti1Image(reoriented_img.get_fdata(), correct_img_nii.affine).to_filename(output_path)

    return output_path

# Calculate volume of a labeled nifti image
def calculate_volume(input_image, output_path=None):
    img = load_image(input_image)
    data = img.get_fdata()
    unique_labels = np.unique(data)

    # voxel size in mm^3
    voxel_size = np.abs(np.linalg.det(img.affine[:3, :3]))

    # calculate voxel count and volume per label
    results = []
    for label in unique_labels:
        voxel_count = np.sum(data == label)
        volume = voxel_count * voxel_size
        results.append((label, voxel_count, voxel_size, volume))

    # determine output path
    if output_path is None:
        base_name, ext = os.path.splitext(os.path.basename(input_image))
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]  # for .nii.gz
        folder = os.path.dirname(input_image) or "."
        output_filename = f"volume_{base_name}.csv"
        output_path = os.path.join(folder, output_filename)

    # write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Label', 'Voxel Count', 'Voxel Size (mm^3)', 'Volume (mm^3)'])
        for label, voxel_count, voxel_size, volume in results:
            writer.writerow([label, voxel_count, voxel_size, volume])

    return output_path

def calculate_roi_volume(input_image, roi_image, output_path=None):
    img = load_image(input_image)
    roi = load_image(roi_image)

    if img.shape != roi.shape:
        raise ValueError("Input image and ROI mask must have the same dimensions.")

    img_data = img.get_fdata()
    roi_data = roi.get_fdata().astype(int)
    unique_labels = np.unique(roi_data)

    voxel_size = np.abs(np.linalg.det(img.affine[:3, :3]))

    results = []
    for label in unique_labels:
        if label == 0:
            continue  # skip background if 0

        mask = roi_data == label
        voxel_count = np.sum(img_data[mask] > 0)  # only count where input_image == 1
        volume = voxel_count * voxel_size

        results.append((label, voxel_count, voxel_size, volume))

    if output_path is None:
        base_name, ext = os.path.splitext(os.path.basename(input_image))
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]
        folder = os.path.dirname(input_image) or "."
        output_filename = f"roi_volume_{base_name}.csv"
        output_path = os.path.join(folder, output_filename)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Label', 'Voxel Count', 'Voxel Size (mm^3)', 'Volume (mm^3)'])
        for label, voxel_count, voxel_size, volume in results:
            writer.writerow([label, voxel_count, voxel_size, volume])

    return output_path

#####################################################
# Function for handling multiple images in a folder #
#####################################################

def calculate_aggregate_image(input_folder, method, output_path):
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                  if f.endswith(('.nii', '.nii.gz', '.mgh'))]

    if not file_paths:
        raise ValueError("No supported image files found in the specified folder.")

    first_img = load_image(file_paths[0])
    data_shape = first_img.shape
    all_data = []

    for fp in file_paths:
        img = load_image(fp)
        if img.shape != data_shape:
            raise ValueError("All images must have the same dimensions.")
        all_data.append(img.get_fdata().astype(np.float32))

    stacked_data = np.stack(all_data, axis=-1)

    if method == 'mean':
        agg_data = np.mean(stacked_data, axis=-1)
    elif method == 'median':
        agg_data = np.median(stacked_data, axis=-1)
    else:
        raise ValueError("Unsupported aggregation method.")

    if output_path is None:
        folder_name = os.path.basename(os.path.normpath(input_folder))
        output_filename = f"{method}_{folder_name}.nii.gz"
        output_path = os.path.join(input_folder, output_filename)

    save_image(agg_data, first_img.affine, first_img.header, output_path)
    print(f"{method.capitalize()} image saved to {output_path}")

############################################
# Main function for command line interface #
############################################

def main():
    parser = argparse.ArgumentParser(description="Image calculator for thresholding/binarizing, masking, mean, and median calculation (NIfTI/MGH).")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-in", "--input", type=str, nargs='+', help="Path to the input image (.nii, .nii.gz, .mgh).")
    input_group.add_argument("-indir", "--input_folder", type=str, help="Path to the input folder containing images.")

    proc_group = parser.add_mutually_exclusive_group()
    proc_group.add_argument("-thr", "--threshold", type=float, help="Threshold value for binarization.")
    proc_group.add_argument("-mask", "--mask", type=str, help="Path to the mask image (.nii, .nii.gz, .mgh).")
    proc_group.add_argument("-extract_label", "--extract_label", type=int, nargs='+', help="List of integer labels to extract from the mask.")
    proc_group.add_argument("-mean", "--mean", action="store_true", help="Calculate mean image from folder.")
    proc_group.add_argument("-median", "--median", action="store_true", help="Calculate median image from folder.")
    proc_group.add_argument("-dtype", "--dtype", type=str, help="Target data type for changing image data type.")
    proc_group.add_argument("-affine", "--affine", type=str, help="Path to the reference NIfTI file for affine update.")
    proc_group.add_argument("-vol", "--volume", action="store_true", help="Calculate volume of labeled image.")
    proc_group.add_argument("-roi_vol", "--roi_vol", action="store_true", help="Calculate volume of labeled image with ROI mask.")
    
    parser.add_argument("-axis", "--axis", type=int, choices=[0, 1, 2], help="Axis for flipping the image data.")
    parser.add_argument("-extract", "--extract", type=str, help="Path to ROI mask file containing integer labels.")
    parser.add_argument("-ign", "--ignore_background", action="store_true")
    parser.add_argument("-bi", "--binarize", action="store_true", help="Force binarization of a image.")
    # in mask_image: if binarize_mask: mask_data = (mask_data > 0).astype(np.float32)
    # in extract_roi_from_image: if binarize: new_data[roi_mask] = 1
    parser.add_argument("-out", "--output", type=str, default=None, nargs='+', help="Output path. Defaults to input directory.")

    args = parser.parse_args()

    if args.input:
        if args.threshold is not None:
            threshold_binarize_image(args.input, args.threshold, args.output)
        elif args.mask is not None:
            mask_image(args.input, args.mask, args.binarize, args.output)
        elif args.dtype is not None:
            change_dtype(args.input, args.dtype, args.output)
        elif args.affine is not None:
            #change_affine(args.affine, args.input, args.output, args.axis)
            # use fix_affine instead
            fix_affine(args.input, args.affine, args.output)
        elif args.volume is not None:
            calculate_volume(args.input, args.output)
        elif args.roi_vol is not None:
            calculate_roi_volume(args.input, args.extract, args.output)
        elif args.swap_lr is not None:
            swap_lr(args.input[0], args.input[1], args.output[0], args.output[1])
        elif args.extract is not None:
            extract_roi_means(args.input, args.extract, args.ignore_background, args.output)
        elif args.extract_label is not None:
            extract_roi_from_image(args.input, args.extract_label, args.binarize, args.output)
        else:
            raise ValueError("For single input file, either --threshold, --mask, or --dtype must be specified.")
    elif args.input_folder and (args.mean or args.median):
        method = 'mean' if args.mean else 'median'
        calculate_aggregate_image(args.input_folder, method, args.output)
    else:
        raise ValueError("Invalid combination of arguments.")

if __name__ == "__main__":
    main()
