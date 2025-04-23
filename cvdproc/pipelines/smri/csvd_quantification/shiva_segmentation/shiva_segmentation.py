# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

# modified by Youjie Wang, 2025-01-09

import os
import nibabel as nib
import numpy as np
import gc
import time
from pathlib import Path
#import tensorflow as tf
from skimage import measure

def load_image(filename):
    """
    Loads a NIfTI image and returns the image data and affine matrix.

    :param filename: str, path to the NIfTI image file
    :return: tuple (image_data, affine)
    """
    dataNii = nib.load(filename)
    # load file and add dimension for the modality
    image = dataNii.get_fdata(dtype=np.float32)[..., np.newaxis]
    image = np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return image, dataNii.affine

def crop_or_pad_image(original_path, cropped_image_path, target_shape, percentage=(0.5, 0.5, 0.5)):
    """
    Crops or pads an image to the target shape according to the given percentages for each dimension.

    :param original_path: str, path to the original image file
    :param cropped_image_path: str, path where the cropped/padded image will be saved
    :param target_shape: tuple, desired shape of the output image
    :param percentage: tuple of 3 floats between 0 and 1. The percentage of slices in each dimension to crop or pad. For
    #             example, if percentage=(0.5, 0.5, 0.25), the function will crop or pad 50% of the slices in each side
    #             (left and right) of the first and second dimensions, and 25% of the slices in the upper side and 75%
    #             of the slices in the lower side of the third dimension.
    """

    # Load the NIfTI image
    img = nib.load(original_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header.copy()

    current_shape = data.shape
    new_data = np.zeros(target_shape, dtype=data.dtype)  # Create a zero array with the target shape

    # Calculate start indices for slicing
    start_idx = []
    end_idx = []
    new_start_idx = []
    new_end_idx = []

    for i, (cs, ts, p) in enumerate(zip(current_shape, target_shape, percentage)):
        if cs > ts:
            # Need to crop
            crop_before = int((cs - ts) * (1 - p))  # More cropping towards the "lower" side if p < 0.5
            crop_after = cs - ts - crop_before
            start_idx.append(crop_before)
            end_idx.append(cs - crop_after)
            new_start_idx.append(0)
            new_end_idx.append(ts)
        else:
            # Need to pad
            pad_total = ts - cs
            pad_before = int(pad_total * (1 - p))  # More padding towards the "lower" side if p < 0.5
            pad_after = pad_total - pad_before
            start_idx.append(0)
            end_idx.append(cs)
            new_start_idx.append(pad_before)
            new_end_idx.append(pad_before + cs)

    # Crop or pad data
    slices_from = tuple(slice(si, ei) for si, ei in zip(start_idx, end_idx))
    slices_to = tuple(slice(nsi, nei) for nsi, nei in zip(new_start_idx, new_end_idx))
    new_data[slices_to] = data[slices_from]

    # Adjust the affine transformation matrix to account for the new data offset
    new_affine = affine.copy()
    voxel_sizes = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    offset_diff = [((start_idx[i] - new_start_idx[i]) * voxel_sizes[i]) for i in range(3)]
    new_affine[:3, 3] -= offset_diff  # Subtract because positive values mean shifting in negative space

    # Save the new NIfTI image
    cropped_img = nib.Nifti1Image(new_data, new_affine, header)
    nib.save(cropped_img, cropped_image_path)

    print(f"Image has been cropped/padded and saved to {cropped_image_path}.")

def predict_one_file(image_to_predict, predictor_files, crop_or_pad_percentage, threshold, prefix, output_path0,
                     prediction_file, thresholded_file, binary_file, target_shape = (160, 214, 176), save_intermediate_image = True):
    """
    Performs prediction on multiple image files (modalities) using the given model(s).
    
    :param image_to_predict: list of str, paths to the image files (multiple modalities)
    :param predictor_files: list of Path objects, paths to the trained model files
    :param crop_or_pad_percentage: tuple, percentage to crop or pad the image
    :param threshold: float, probability threshold for segmentation
    :param prefix: str, prefix for the output file names
    :param output_path0: str, output directory path
    :param target_shape: tuple, desired shape of the output image
    :param save_intermediate_image: bool, whether to save intermediate images
    """
    _VERBOSE = True
    import os
    import nibabel as nib
    import numpy as np
    import gc
    import time
    from pathlib import Path
    #import tensorflow as tf
    from skimage import measure
    import tensorflow as tf

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    base_filename = Path(image_to_predict[0]).name.split('.')[0]  # Use the first image name for the base filename

    # Store paths for cropped images
    cropped_image_paths = []

    # Process each original image (modality)
    for original_image in image_to_predict:
        original_shape = nib.load(original_image).shape
        cropped_image_path = Path(output_path0) / ('crop_' + Path(original_image).name)

        # Crop or pad the image
        crop_or_pad_image(original_image, cropped_image_path, target_shape, crop_or_pad_percentage)
        cropped_image_paths.append(cropped_image_path)

    # Load and process the modalities
    modalities = cropped_image_paths
    brainmask = None
    output_path = Path(output_path0) / (f'{base_filename}_{prefix}_temp.nii.gz')

    affine = None
    image_shape = None
    images = []

    # Load image data and normalize
    for modality in modalities:
        image, aff = load_image(modality)
        image_min = image.min()
        image = (image - image_min) / (image.max() - image_min)

        if affine is None:
            affine = aff
        if image_shape is None:
            image_shape = image.shape
        elif image.shape != image_shape:
            raise ValueError(f'Images have different shape {image_shape} vs {image.shape} in {modality}')

        if brainmask is not None:
            image *= brainmask
        images.append(image)

    # Concatenate all modalities
    images = np.concatenate(images, axis=-1)
    images = np.reshape(images, (1,) + images.shape)  # Add batch dimension

    chrono0 = time.time()

    # Load models and predict
    predictions = []
    predictor_files = [Path(f) for f in predictor_files]
    for predictor_file in predictor_files:
        tf.keras.backend.clear_session()
        gc.collect()

        try:
            model = tf.keras.models.load_model(predictor_file, compile=False, custom_objects={"tf": tf})
        except Exception as err:
            print(f'\n\tWARNING: Exception loading model: {predictor_file}\n{err}')
            continue

        print(f'INFO: Predicting fold: {predictor_file.stem}')
        prediction = model.predict(images, batch_size=1)
        if brainmask is not None:
            prediction *= brainmask
        predictions.append(prediction)

    # Average predictions from all folds
    predictions = np.mean(predictions, axis=0)

    chrono1 = (time.time() - chrono0) / 60.
    if _VERBOSE:
        print(f'Inference time: {chrono1} sec.')

    # Save prediction as soft segmentation
    predictions = predictions.squeeze()
    nifti = nib.Nifti1Image(predictions, affine=affine)
    nifti.to_filename(output_path)

    # Thresholding and segmentation for 3D slicer compatibility
    # use 0pXX in file name
    thr_string = f'{threshold:.2f}'.replace('.', 'p')
    thr_path = Path(output_path0) / Path(f'{base_filename}_{prefix}_labeled_thr{thr_string}_temp.nii.gz')

    thresholded = (predictions > threshold).astype(int).squeeze()
    clusters, num_clusters = measure.label(thresholded, background=0, connectivity=3, return_num=True)

    probas = [predictions[clusters == i].max() for i in range(1, num_clusters)]
    sort_index = np.argsort(probas)
    sort_index = np.concatenate([[0], sort_index[::-1] + 1])

    cluster_2 = np.zeros_like(clusters)
    for i_cluster in range(num_clusters):
        cluster_2[clusters == sort_index[i_cluster]] = i_cluster
    nifti = nib.Nifti1Image(cluster_2.astype(np.short), affine=affine)
    nifti.to_filename(thr_path)

    # Create binary thresholded result
    binary_mask = (thresholded > 0).astype(int)

    # Save binary thresholded result
    binary_thr_path = Path(output_path0) / (f'{base_filename}_{prefix}_binary_thr{thr_string}_temp.nii.gz')
    nifti_binary = nib.Nifti1Image(binary_mask.astype(np.short), affine)
    nifti_binary.to_filename(binary_thr_path)

    # Revert to original size by cropping or padding the output files
    reverted_path_1 = str(output_path).replace('_temp.nii.gz', '.nii.gz')
    reverted_path_2 = str(thr_path).replace('_temp.nii.gz', '.nii.gz')
    reverted_path_3 = str(binary_thr_path).replace('_temp.nii.gz', '.nii.gz')

    crop_or_pad_image(output_path, reverted_path_1, original_shape, crop_or_pad_percentage)
    crop_or_pad_image(thr_path, reverted_path_2, original_shape, crop_or_pad_percentage)
    crop_or_pad_image(binary_thr_path, reverted_path_3, original_shape, crop_or_pad_percentage)

    # Delete temporary cropped images
    if not save_intermediate_image:
        for cropped_image_path in cropped_image_paths:
            os.remove(cropped_image_path)

    # Delete other temporary files
    os.remove(output_path)
    os.remove(thr_path)
    os.remove(binary_thr_path)

    os.rename(reverted_path_1, os.path.join(output_path0, prediction_file))
    os.rename(reverted_path_2, os.path.join(output_path0, thresholded_file))
    os.rename(reverted_path_3, os.path.join(output_path0, binary_file))

    print(f'\nINFO: Done with predictions -> {prediction_file}\nThresholded: {thresholded_file} and {binary_file}')


# Nipype interface
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits
import os
from pathlib import Path

# 定义输入参数
class SHIVAPredictImageInputSpec(BaseInterfaceInputSpec):
    image_to_predict = traits.List(File(exists=True), mandatory=True, desc="List of input NIfTI image files (multiple modalities)")
    predictor_files = traits.List(File(exists=True), mandatory=True, desc="List of trained model files")
    crop_or_pad_percentage = traits.Tuple(traits.Float, traits.Float, traits.Float, usedefault=True,
                                          desc="Percentage for cropping or padding (default: (0.5, 0.5, 0.5))")
    threshold = traits.Float(mandatory=True, desc="Probability threshold for segmentation")
    prefix = traits.Str(mandatory=True, desc="Prefix for output filenames")
    output_path = Directory(mandatory=True, desc="Directory to save output files")
    prediction_file = File(desc="Final predicted segmentation file")
    thresholded_file = File(desc="Thresholded segmentation file")
    binary_file = File(desc="Binary thresholded segmentation file")
    target_shape = traits.Tuple(traits.Int, traits.Int, traits.Int, usedefault=True,
                                desc="Target shape for image resizing (default: (160, 214, 176))")
    save_intermediate_image = traits.Bool(default_value=True, usedefault=True,
                                          desc="Whether to save intermediate images")

# 定义输出参数
class SHIVAPredictImageOutputSpec(TraitedSpec):
    prediction_file = File(desc="Final predicted segmentation file")
    thresholded_file = File(desc="Thresholded segmentation file")
    binary_file = File(desc="Binary thresholded segmentation file")

# 自定义 `nipype` 接口
class SHIVAPredictImage(BaseInterface):
    input_spec = SHIVAPredictImageInputSpec
    output_spec = SHIVAPredictImageOutputSpec

    def _run_interface(self, runtime):
        from pathlib import Path
        # 解析输入参数
        image_to_predict = self.inputs.image_to_predict
        predictor_files = self.inputs.predictor_files
        crop_or_pad_percentage = self.inputs.crop_or_pad_percentage
        threshold = self.inputs.threshold
        prefix = self.inputs.prefix
        output_path = self.inputs.output_path
        prediction_file = self.inputs.prediction_file
        thresholded_file = self.inputs.thresholded_file
        binary_file = self.inputs.binary_file
        target_shape = self.inputs.target_shape
        save_intermediate_image = self.inputs.save_intermediate_image

        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

        # 调用 `predict_one_file` 进行预测
        predict_one_file(
            image_to_predict=image_to_predict,
            predictor_files=predictor_files,
            crop_or_pad_percentage=crop_or_pad_percentage,
            threshold=threshold,
            prefix=prefix,
            output_path0=output_path,
            prediction_file=prediction_file,
            thresholded_file=thresholded_file,
            binary_file=binary_file,
            target_shape=target_shape,
            save_intermediate_image=save_intermediate_image
        )

        return runtime

    def _list_outputs(self):
        """返回预测函数生成的主要输出文件路径"""
        outputs = self._outputs().get()
        base_filename = Path(self.inputs.image_to_predict[0]).name.split('.')[0]
        prefix = self.inputs.prefix
        output_path = Path(self.inputs.output_path)

        thr_string = f'{self.inputs.threshold:.2f}'.replace('.', 'p')
        outputs['prediction_file'] = os.path.join(output_path, self.inputs.prediction_file)
        outputs['thresholded_file'] = os.path.join(output_path, self.inputs.thresholded_file)
        outputs['binary_file'] = os.path.join(output_path, self.inputs.binary_file)

        return outputs

if __name__ == '__main__':
    from nipype import Node

    # 创建 `PredictImage` 节点
    predict_node = Node(SHIVAPredictImage(), name="predict_image")

    # 设置输入
    predict_node.inputs.image_to_predict = [
        "/mnt/f/BIDS/UKB_AFproject/sub-AF1000077/ses-01/anat/sub-AF1000077_ses-01_T1w.nii.gz"
    ]
    predict_node.inputs.predictor_files = [
        "/mnt/e/Neuroimage/SHIVA_model/PVS_model/v1/T1.PVS/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_0_model.h5",
        "/mnt/e/Neuroimage/SHIVA_model/PVS_model/v1/T1.PVS/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_1_model.h5",
        "/mnt/e/Neuroimage/SHIVA_model/PVS_model/v1/T1.PVS/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_2_model.h5",
        "/mnt/e/Neuroimage/SHIVA_model/PVS_model/v1/T1.PVS/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_3_model.h5",
        "/mnt/e/Neuroimage/SHIVA_model/PVS_model/v1/T1.PVS/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_4_model.h5",
        "/mnt/e/Neuroimage/SHIVA_model/PVS_model/v1/T1.PVS/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_5_model.h5"
    ]
    predict_node.inputs.crop_or_pad_percentage = (0.5, 0.5, 0.5)
    predict_node.inputs.threshold = 0.5
    predict_node.inputs.prefix = "prediction"
    predict_node.inputs.output_path = "/mnt/f/BIDS/UKB_AFproject/derivatives/nipype"
    predict_node.inputs.prediction_file = "prediction.nii.gz"
    predict_node.inputs.thresholded_file = "thresholded.nii.gz"
    predict_node.inputs.binary_file = "binary.nii.gz"
    predict_node.inputs.target_shape = (160, 214, 176)
    predict_node.inputs.save_intermediate_image = True

    # 运行 `Node`
    result = predict_node.run()
