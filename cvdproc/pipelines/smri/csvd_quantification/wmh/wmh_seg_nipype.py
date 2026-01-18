import os
import subprocess
import shutil
import nibabel as nib
import time
import numpy as np
import pandas as pd
from nipype import Node, Workflow
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str, Float, Either

from scipy.ndimage import label

'''
Different methods of WMH segmentation
'''

####################
# LST Segmentation #
####################

class LSTSegmentationInputSpec(BaseInterfaceInputSpec):
    flair = File(desc="Path to the FLAIR image", exists=True, mandatory=True)
    t1w = Str(desc="Path to the T1-weighted image", mandatory=False)
    threshold = Float(desc="Threshold for WMH segmentation", mandatory=True)
    output_path = Str(desc="Output directory to save the results", mandatory=True)
    
    WMH_mask_flair = Str(desc="Name of the WMH mask file in FLAIR space")
    WMH_prob_flair = Str(desc="Name of the WMH probability map file in FLAIR space")
    WMH_mask_t1w = Str(desc="Name of the WMH mask file in T1w space")
    WMH_prob_t1w = Str(desc="Name of the WMH probability map file in T1w space")
    FLAIR_in_T1w = Str(desc="Name of the FLAIR image in T1w space")

    spm_path = Directory(desc="Path to SPM installation")
    lst_path = Str(desc="Path to MATLAB executable")
    script_path = File(desc="Path to the .m script", exists=True, mandatory=True)

class LSTSegmentationOutputSpec(TraitedSpec):
    wmh_mask_flair = Str(desc="Path to the WMH mask file in FLAIR space")
    wmh_prob_flair = Str(desc="Path to the WMH probability map file in FLAIR space")
    wmh_mask_t1w = Str(desc="Path to the WMH mask file in T1w space")
    wmh_prob_t1w = Str(desc="Path to the WMH probability map file in T1w space")

class LSTSegmentation(BaseInterface):
    input_spec = LSTSegmentationInputSpec
    output_spec = LSTSegmentationOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.script_path) as script_file:
            script_content = script_file.read()
        
        # Replace the placeholders in the script
        subject_matlab_script = os.path.join(self.inputs.output_path, 'wmh_seg_lst_script.m')
        with open(subject_matlab_script, 'w') as script_file:
            new_script_content = script_content
            new_script_content = new_script_content.replace('/this/is/for/nipype/flair', 
                                                            self.inputs.flair)
            new_script_content = new_script_content.replace('/this/is/for/nipype/t1w',
                                                            self.inputs.t1w)
            new_script_content = new_script_content.replace('/this/is/for/nipype/threshold',
                                                            str(self.inputs.threshold))
            new_script_content = new_script_content.replace('/this/is/for/nipype/output_path',
                                                            self.inputs.output_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/spm_path',
                                                            str(self.inputs.spm_path))
            new_script_content = new_script_content.replace('/this/is/for/nipype/lst_path',
                                                            str(self.inputs.lst_path))
            new_script_content = new_script_content.replace('/this/is/for/nipype/WMH_mask_flair',
                                                            str(self.inputs.WMH_mask_flair))
            new_script_content = new_script_content.replace('/this/is/for/nipype/WMH_prob_flair',
                                                            str(self.inputs.WMH_prob_flair))
            new_script_content = new_script_content.replace('/this/is/for/nipype/WMH_mask_t1w',
                                                            str(self.inputs.WMH_mask_t1w))
            new_script_content = new_script_content.replace('/this/is/for/nipype/WMH_prob_t1w',
                                                            str(self.inputs.WMH_prob_t1w))
            new_script_content = new_script_content.replace('/this/is/for/nipype/FLAIR_in_T1w_path',
                                                            str(self.inputs.FLAIR_in_T1w))
            
            script_file.write(new_script_content)
        
        self._wmh_mask_flair = os.path.join(self.inputs.output_path, self.inputs.WMH_mask_flair)
        self._wmh_prob_flair = os.path.join(self.inputs.output_path, self.inputs.WMH_prob_flair)
        self._wmh_mask_t1w = os.path.join(self.inputs.output_path, self.inputs.WMH_mask_t1w) if self.inputs.t1w != '' else ''
        self._wmh_prob_t1w = os.path.join(self.inputs.output_path, self.inputs.WMH_prob_t1w) if self.inputs.t1w != '' else ''

        if os.path.exists(self.inputs.t1w):
            target_check = self._wmh_mask_t1w
        else:
            target_check = self._wmh_mask_flair

        if not os.path.exists(target_check):
            cmd_str = f"run('{subject_matlab_script}'); exit;"
            mlab = CommandLine('matlab', args=f"-nodisplay -nosplash -nodesktop -r \"{cmd_str}\"", terminal_output='stream')
            result = mlab.run()
            
            return result.runtime
        else:
            print(f"Output file {target_check} already exists. Skipping segmentation.")
            return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['wmh_mask_flair'] = self._wmh_mask_flair
        outputs['wmh_prob_flair'] = self._wmh_prob_flair
        outputs['wmh_mask_t1w'] = self._wmh_mask_t1w if self._wmh_mask_t1w != '' else ''
        outputs['wmh_prob_t1w'] = self._wmh_prob_t1w if self._wmh_prob_t1w != '' else ''

        return outputs


##########
# LST-AI #
##########

class LSTAIInputSpec(CommandLineInputSpec):
    t1w_img = File(desc="Path to the T1-weighted image", argstr='--t1 %s')
    flair_img = File(desc="Path to the FLAIR image", argstr='--flair %s')
    output_dir = Directory(desc="Directory to save the output", argstr='--output %s')
    temp_dir = Directory(desc="Temporary directory for intermediate files", argstr='--temp %s')
    save_prob_map = Bool(desc="Save the probability map", argstr='--probability_map', default=True)
    img_stripped = Bool(desc="Use stripped images", argstr='--stripped', default=False)
    existing_seg = File(desc="Path to existing segmentation file", argstr='--existing_seg %s')
    segment_only = Bool(desc="Only segment the image", argstr='--segment_only', default=False)
    annotate_only = Bool(desc="Only annotate the image", argstr='--annotate_only', default=False)
    threshold = Float(desc="Threshold for WMH segmentation", argstr='--threshold %f', default=0.5)
    lesion_threshold = Int(desc="Lesion size threshold for WMH segmentation", argstr='--lesion_threshold %d', default=0)
    device = Either('cpu', Int, desc="Device to use for processing", argstr='--device %s', default=0)
    threads = Int(desc="Number of threads to use", argstr='--threads %d', default=1)

class LSTAIOutputSpec(TraitedSpec):
    input_t1w_img = Either(None, File(exists=True), desc="Path to the input T1-weighted image")
    input_flair_img = Either(None, File(exists=True), desc="Path to the input FLAIR image")
    wmh_mask = Str(desc="Path to the WMH mask generated by LST-AI")
    wmh_prob_map = Str(desc="Path to the WMH probability map generated by LST-AI")
    parcellated_volume = Str(desc="Path to the parcellated volume CSV file")
    parcellated_wmh_mask = Str(desc="Path to the parcellated WMH mask NIfTI file")

class LSTAI(CommandLine):
    input_spec = LSTAIInputSpec
    output_spec = LSTAIOutputSpec
    _cmd = "lst"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['input_t1w_img'] = os.path.abspath(self.inputs.t1w_img)
        outputs['input_flair_img'] = os.path.abspath(self.inputs.flair_img)
        outputs['wmh_mask'] = os.path.abspath(os.path.join(self.inputs.output_dir, 'space-flair_seg-lst.nii.gz'))
        outputs['wmh_prob_map'] = os.path.abspath(os.path.join(self.inputs.temp_dir, 'sub-X_ses-Y_space-FLAIR_seg-lst_prob.nii.gz'))
        outputs['parcellated_volume'] = os.path.abspath(os.path.join(self.inputs.output_dir, 'annotated_lesion_stats.csv'))
        outputs['parcellated_wmh_mask'] = os.path.abspath(os.path.join(self.inputs.output_dir, 'sub-X_ses-Y_space-flair_desc-annotated_seg-lst.nii.gz'))

        return outputs

if __name__ == "__main__":
    lst_ai = LSTAI()
    lst_ai.inputs.t1w_img = "/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-02/anat/T1w_brain.nii.gz"
    lst_ai.inputs.flair_img = "/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-02/anat/FLAIR_brain.nii.gz"
    lst_ai.inputs.output_dir = "/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-02/anat/lst_ai_output"
    lst_ai.inputs.temp_dir = "/mnt/f/BIDS/SVD_BIDS/sub-SVD0100/ses-02/anat/lst_ai_output"
    lst_ai.inputs.threshold = 0.25
    lst_ai.inputs.save_prob_map = True
    lst_ai.inputs.img_stripped = True
    lst_ai.inputs.device = 0

    lst_ai.run()

################
# WMH-SynthSeg #
################

from cvdproc.utils.python.basic_image_processor import extract_roi_from_image

class WMHSynthSegInputSpec(BaseInterfaceInputSpec):
    flair_img = File(mandatory=True, desc="Path to the FLAIR image")
    output_dir = Directory(mandatory=True, desc="Directory to save the output")
    output_mask_name = Str(mandatory=True, desc="Name of the output WMH mask")
    output_prob_map_name = Str(mandatory=True, desc="Name of the output WMH probability map")
    seg_name = Str(mandatory=True, desc="Name of the segmentation method")

class WMHSynthSegOutputSpec(TraitedSpec):
    wmh_mask = Str(desc="Path to the WMH mask generated by WMH-SynthSeg")
    wmh_prob_map = Str(desc="Path to the WMH probability map generated by WMH-SynthSeg")
    wmh_synthseg = Str(desc="Path to the WMH-SynthSeg output file")

class WMHSynthSeg(BaseInterface):
    input_spec = WMHSynthSegInputSpec
    output_spec = WMHSynthSegOutputSpec

    def _run_interface(self, runtime):
        # Create the output directory if it doesn't exist
        os.makedirs(self.inputs.output_dir, exist_ok=True)

        # prepare the input dir
        input_dir = os.path.join(self.inputs.output_dir, 'input4WMHSynthSeg')
        os.makedirs(input_dir, exist_ok=True)
        shutil.copy(self.inputs.flair_img, input_dir)

        vol_csv = os.path.join(self.inputs.output_dir, 'WMHSynthSegVols.csv')

        # Avoid possible CUDA imcompability issues
        # Here we use the CPU for the segmentation

        flair_filename = os.path.basename(self.inputs.flair_img).split(".")[0]
        probmap_file_old = os.path.join(self.inputs.output_dir, f'{flair_filename}_seg.lesion_probs.nii.gz')
        seg_file_old = os.path.join(self.inputs.output_dir, f'{flair_filename}_seg.nii.gz')

        seg_file = os.path.join(self.inputs.output_dir, self.inputs.seg_name)
        probmap_file = os.path.join(self.inputs.output_dir, self.inputs.output_prob_map_name)

        if os.path.exists(seg_file) and os.path.exists(probmap_file):
            print(f"Output files already exist: {seg_file}, {probmap_file}. Skipping segmentation.")
            wmh_mask = os.path.join(self.inputs.output_dir, self.inputs.output_mask_name)
        else:
            cmd = [
                'mri_WMHsynthseg',
                '--i', input_dir,
                '--o', self.inputs.output_dir,
                '--csv_vols', vol_csv,
                '--device', 'cuda',
                '--crop',
                '--save_lesion_probabilities'
            ]

            # cmd = [
            #     'mri_WMHsynthseg',
            #     '--i', input_dir,
            #     '--o', self.inputs.output_dir,
            #     '--csv_vols', vol_csv,
            #     '--device', 'cpu',
            #     '--threads', '8',
            #     '--save_lesion_probabilities'
            # ]

            subprocess.run(cmd, check=True)
            
            os.rename(probmap_file_old, probmap_file)
            os.rename(seg_file_old, seg_file)
            wmh_mask = extract_roi_from_image(seg_file, [77], binarize=True, output_path=os.path.join(self.inputs.output_dir, self.inputs.output_mask_name))

        # Set the output files
        self._wmh_mask = wmh_mask
        self._wmh_prob_map = probmap_file
        self._wmh_synthseg = seg_file

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['wmh_mask'] = os.path.abspath(self._wmh_mask)
        outputs['wmh_prob_map'] = os.path.abspath(self._wmh_prob_map)
        outputs['wmh_synthseg'] = os.path.abspath(self._wmh_synthseg)
        
        return outputs

# WMHsynthseg for a single image (command line interface)

# usage: inference.py [-h] --i I --o O [--csv_vols CSV_VOLS] [--device DEVICE]
#                     [--threads THREADS] [--save_lesion_probabilities] [--crop]

# WMH-SynthSeg: joint segmentation of anatomy and white matter hyperintensities

# optional arguments:
#   -h, --help            show this help message and exit
#   --i I                 Input image or directory.
#   --o O                 Output segmentation (or directory, if the input is a
#                         directory)
#   --csv_vols CSV_VOLS   (optional) CSV file with volumes of ROIs
#   --device DEVICE       device (cpu or cuda; optional)
#   --threads THREADS     (optional) Number of CPU cores to be used. Default is
#                         1. You can use -1 to use all available cores
#   --save_lesion_probabilities
#                         (optional) Saves lesion probability maps
#   --crop                (optional) Does two passes, to limit size to
#                         192x224x192 cuboid (needed for GPU processing)

class WMHSynthSegSingleInputSpec(CommandLineInputSpec):
    input = Str(mandatory=True, desc="Input image or directory", argstr='--i %s')
    output = Str(mandatory=True, desc="Output segmentation (or directory, if the input is a directory)", argstr='--o %s')
    csv_vols = File(desc="CSV file with volumes of ROIs", argstr='--csv_vols %s', mandatory=False)
    device = Either('cpu', 'cuda', Int, desc="Device to use for processing", argstr='--device %s', default='cuda')
    threads = Int(desc="Number of CPU cores to be used", argstr='--threads %d', default=1)
    save_lesion_probabilities = Bool(desc="Saves lesion probability maps", argstr='--save_lesion_probabilities', default=False)
    crop = Bool(desc="Does two passes, to limit size to 192x224x192 cuboid (needed for GPU processing)", argstr='--crop', default=False)

    prob_filepath = File(desc="Path to the probability map file", argstr='--prob_filepath %s', mandatory=False)
    wmh_filepath = File(desc="Path to the WMH mask file", argstr='--wmh_filepath %s', mandatory=False)

class WMHSynthSegSingleOutputSpec(TraitedSpec):
    prob_filepath = File(desc="Path to the probability map file")
    wmh_filepath = File(desc="Path to the WMH mask file")
    seg_filepath = File(desc="Path to the segmentation file")

class WMHSynthSegSingle(CommandLine):
    input_spec = WMHSynthSegSingleInputSpec
    output_spec = WMHSynthSegSingleOutputSpec
    
    _cmd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "bash", "mri_WMHsynthseg_single.sh"))

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['prob_filepath'] = self.inputs.prob_filepath
        outputs['wmh_filepath'] = self.inputs.wmh_filepath
        outputs['seg_filepath'] = self.inputs.output

        return outputs

########################
# truenet segmentation #
########################

class PrepareTrueNetDataInputSpec(CommandLineInputSpec):
    FLAIR = File(desc="FLAIR image path", argstr='--FLAIR=%s', exists=True)
    T1 = File(desc="T1 image path", argstr='--T1=%s', exists=True)
    outname = Str(desc="Output basename", mandatory=True, argstr='--outname=%s')
    manualmask = File(desc="Manual mask image path", argstr='--manualmask=%s', exists=True)
    nodistmaps = Bool(desc="Skip adding distance maps", argstr='--nodistmaps')
    keepintermediate = Bool(desc="Keep intermediate results", argstr='--keepintermediate')
    verbose = Bool(desc="Verbose output", argstr='-v')

class PrepareTrueNetDataOutputSpec(TraitedSpec):
    output_dir = Directory(desc="Output directory for processed images")
    processed_flair = Either(File, None, desc="Processed FLAIR image (or None if not provided)")
    processed_t1 = Either(File, None, desc="Processed T1 image (or None if not provided)")

class PrepareTrueNetData(CommandLine):
    input_spec = PrepareTrueNetDataInputSpec
    output_spec = PrepareTrueNetDataOutputSpec
    _cmd = "prepare_truenet_data"

    def _run_interface(self, runtime):
        # At least one of FLAIR or T1 must be provided
        if not getattr(self.inputs, "FLAIR", None) and not getattr(self.inputs, "T1", None):
            raise ValueError("At least one of FLAIR or T1 must be provided.")
        return super()._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()

        out_base = os.path.abspath(self.inputs.outname)
        out_dir = os.path.dirname(out_base) if os.path.dirname(out_base) else os.getcwd()
        outputs["output_dir"] = out_dir

        outputs["processed_flair"] = (
            self._gen_output_filename("FLAIR") if getattr(self.inputs, "FLAIR", None) else None
        )
        outputs["processed_t1"] = (
            self._gen_output_filename("T1") if getattr(self.inputs, "T1", None) else None
        )

        return outputs

    def _gen_output_filename(self, suffix: str) -> str:
        return f"{self.inputs.outname}_{suffix}.nii.gz"

class PrepareTrueNetData2InputSpec(CommandLineInputSpec):
    flair = File(desc="Skull-stripped FLAIR image path (should be aligned with T1w)", argstr='%s', position=0, exists=True)
    t1w = File(desc="Skull-stripped T1 image path (should be aligned with FLAIR)", argstr='%s', position=1, exists=True)
    brain_mask = File(desc="Brain mask path", argstr='%s', position=2, exists=True)
    synthseg_img = File(desc="SynthSeg output image path", argstr='%s', position=3, exists=True)
    output_dir = Directory(desc="Output directory for processed images", argstr='%s', position=4, exists=False, mandatory=True)
    prefix = Str(desc="Prefix for output files", argstr='%s', position=5, exists=False, mandatory=True)

class PrepareTrueNetData2OutputSpec(TraitedSpec):
    output_dir = Directory(desc="Output directory for processed images")
    processed_flair = Either(File, None, desc="Processed FLAIR image (or None if not provided)")
    processed_t1 = Either(File, None, desc="Processed T1 image (or None if not provided)")

class PrepareTrueNetData2(CommandLine):
    _cmd = 'bash ' + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "bash", "fsl", "truenet_preprocess_custom.sh"))
    input_spec = PrepareTrueNetData2InputSpec
    output_spec = PrepareTrueNetData2OutputSpec
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_dir'] = os.path.abspath(self.inputs.output_dir)
        outputs['processed_flair'] = os.path.join(outputs['output_dir'], f"{self.inputs.prefix}_FLAIR.nii.gz") if getattr(self.inputs, "flair", None) else None
        outputs['processed_t1'] = os.path.join(outputs['output_dir'], f"{self.inputs.prefix}_T1.nii.gz") if getattr(self.inputs, "t1w", None) else None
        
        return outputs

class TrueNetEvaluateInputSpec(CommandLineInputSpec):
    inp_dir = Directory(exists=True, desc="Input directory containing test images", mandatory=True, argstr='-i %s')
    model_name = Str(desc="Pre-trained model name or model path", mandatory=True, argstr='-m %s')
    output_dir = Str(desc="Output directory for saving predictions", mandatory=True, argstr='-o %s')
    use_cpu = Bool(desc="Force the model to evaluate the model on CPU", default=False, argstr='-cpu')
    num_classes = Int(desc="Number of classes in the labels used for training the model", default=2, argstr='-nclass %d')
    intermediate = Bool(desc="Save intermediate predictions", default=False, argstr='-int')
    cp_type = Str(desc="Checkpoint load type", default='last', argstr='-cp_type %s')
    cp_n = Int(desc="If -cp_type=specific, the N value", default=10, argstr='-cp_n %d')
    verbose = Bool(desc="Display debug messages", default=False, argstr='-v')

class TrueNetEvaluateOutputSpec(TraitedSpec):
    output_dir = Directory(desc="Output directory for saving predictions")
    pred_file = File(desc="Path to the predicted file")

class TrueNetEvaluate(CommandLine):
    input_spec = TrueNetEvaluateInputSpec
    output_spec = TrueNetEvaluateOutputSpec
    _cmd = "truenet evaluate"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_dir'] = os.path.abspath(self.inputs.output_dir)

        # get the prefix in the inp_dir (truenet_preprocess_FLAIR.nii.gz -> truenet_preprocess)
        flair_file = [f for f in os.listdir(self.inputs.inp_dir) if f.endswith('FLAIR.nii.gz')]
        prefix = flair_file[0].split('_FLAIR')[0] if flair_file else None

        if prefix:
            outputs['pred_file'] = os.path.join(self.inputs.output_dir, f"Predicted_probmap_truenet_{prefix}.nii.gz")
        return outputs
    
    
class TrueNetPostProcessInputSpec(BaseInterfaceInputSpec):
    preprocess_dir = Directory(exists=True, desc="Directory containing preprocessed images", mandatory=True)
    pred_file = Str(mandatory=True, desc="Path to the predicted file")
    output_dir = Directory(desc="Output directory for processed images", mandatory=True)
    threshold = Float(mandatory=True, desc="Threshold for WMH segmentation") # Here we only expect one threshold value
    output_mask_name = Str(mandatory=True, desc="Name of the output WMH mask")
    output_prob_map_name = Str(mandatory=True, desc="Name of the output WMH probability map")

class TrueNetPostProcessOutputSpec(TraitedSpec):
    wmh_mask = File(desc="Path to the WMH mask generated by TrueNet")
    wmh_prob_map = File(desc="Path to the WMH probability map generated by TrueNet")

class TrueNetPostProcess(BaseInterface):
    input_spec = TrueNetPostProcessInputSpec
    output_spec = TrueNetPostProcessOutputSpec

    def _run_interface(self, runtime):
        # Create the output directory if it doesn't exist
        os.makedirs(self.inputs.output_dir, exist_ok=True)

        # Load the TrueNet output file
        truenet_output_file = self.inputs.pred_file
        truenet_output_img = nib.load(truenet_output_file)
        truenet_output_data = truenet_output_img.get_fdata()

        # Apply thresholding to create a binary mask
        binary_mask = (truenet_output_data > self.inputs.threshold).astype(np.uint8)

        # Try to load WM mask
        wm_mask_file = [f for f in os.listdir(self.inputs.preprocess_dir) if f.endswith('_WMmask.nii.gz')]
        if wm_mask_file:
            wm_mask_file = os.path.join(self.inputs.preprocess_dir, wm_mask_file[0])
            wm_mask_img = nib.load(wm_mask_file)
            wm_mask_data = wm_mask_img.get_fdata()

            # Label WMH clusters
            labeled_mask, num_features = label(binary_mask, structure=np.ones((3, 3, 3)))
            for i in range(1, num_features + 1):
                lesion_mask = (labeled_mask == i).astype(np.uint8)
                lesion_voxels = np.sum(lesion_mask)
                if lesion_voxels == 0:
                    continue

                outside_voxels = np.sum(lesion_mask * (wm_mask_data == 0))
                outside_ratio = outside_voxels / lesion_voxels

                # Remove lesion if >30% outside WM mask
                if outside_ratio > 0.3:
                    binary_mask[labeled_mask == i] = 0

        # Save the binary mask as a NIfTI file
        wmh_mask_file = os.path.join(self.inputs.output_dir, self.inputs.output_mask_name)
        nib.save(nib.Nifti1Image(binary_mask, truenet_output_img.affine), wmh_mask_file)
        
        # Also mask probability map (only in WM)
        probmap_file = os.path.join(self.inputs.output_dir, self.inputs.output_prob_map_name)
        prob_data = truenet_output_data.astype(np.float32)
        if wm_mask_file:
            # wm_mask_data already loaded above if wm_mask_file is found
            wm_bin = (wm_mask_data > 0).astype(np.uint8)
            prob_data = prob_data * wm_bin
        nib.save(nib.Nifti1Image(prob_data, truenet_output_img.affine), probmap_file)

        # Set the output files
        self._wmh_mask = wmh_mask_file
        self._wmh_prob_map = probmap_file

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['wmh_mask'] = os.path.abspath(self._wmh_mask)
        outputs['wmh_prob_map'] = os.path.abspath(self._wmh_prob_map)
        return outputs