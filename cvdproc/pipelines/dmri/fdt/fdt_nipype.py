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
from traits.api import Bool, Int, Str

###########################################
# DTI preprocessing, dtifit, and bedpostx #
###########################################
class DTIpreprocessingInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(exists=True, desc='BIDS root directory')
    t1w_file = File(exists=True, desc='T1w file')
    dwi_file = File(exists=True, desc='DWI file')
    bval_file = File(exists=True, desc='bval file')
    bvec_file = File(exists=True, desc='bvec file')
    json_file = File(exists=True, desc='json file')
    use_synb0 = Bool(False, desc='Use synb0 for topup')
    output_path_synb0 = Directory(exists=False, desc='Output path for synb0')
    output_path = Directory(exists=True, desc='Output path for fdt')
    phase_encoding_number = Str(desc='Phase encoding number. eg. "-1 0 0" for "i"')
    total_readout_time = Str(desc='Total readout time')
    script_path_dtipreprocess = Str(desc='Path to the script') # DTI processing, fitting, and Bedpostx

class DTIpreprocessingOutputSpec(TraitedSpec):
    fa_file = File(desc='FA file')
    bedpostx_input_dir = Directory(desc='Bedpostx output directory')

class DTIpreprocessing(BaseInterface):
    input_spec = DTIpreprocessingInputSpec
    output_spec = DTIpreprocessingOutputSpec

    def _run_interface(self, runtime):
        # get the input
        bids_dir = self.inputs.bids_dir
        t1w_file = self.inputs.t1w_file
        dwi_image = self.inputs.dwi_file
        dwi_bval = self.inputs.bval_file
        dwi_bvec = self.inputs.bvec_file
        dwi_json = self.inputs.json_file
        use_synb0 = self.inputs.use_synb0
        output_path_synb0 = self.inputs.output_path_synb0
        output_path = self.inputs.output_path
        phase_encoding_number = self.inputs.phase_encoding_number
        total_readout_time = self.inputs.total_readout_time
        script_path_dtipreprocess = self.inputs.script_path_dtipreprocess
        
        b0_all = os.path.join(output_path, 'b0_all.nii.gz')
        acqparam = os.path.join(output_path, 'acqparam.txt')

        # preprocess using the script
        fa_file = os.path.join(output_path, 'dti_FA.nii.gz')

        if not os.path.exists(fa_file):
            # use synb0-disco for topup
            if use_synb0:
                synb0_input_path = os.path.join(output_path_synb0, 'INPUTS')
                os.makedirs(synb0_input_path, exist_ok=True)
                synb0_output_path = os.path.join(output_path_synb0, 'OUTPUTS')

                if not os.path.exists(os.path.join(output_path_synb0, 'b0_all.nii.gz')):
                    os.makedirs(synb0_output_path, exist_ok=True)

                    # INPUTS for synb0-disco
                    # Create a skull-stripped T1w image in INPUTS
                    subprocess.run(['mri_synthstrip', '-i', t1w_file, '-o', os.path.join(synb0_input_path, 'T1.nii.gz')], check=True)
                    # Get the b0 image from the DWI (assume the first volume in the 4th dimension is b0)
                    subprocess.run(['fslroi', dwi_image, os.path.join(synb0_input_path, 'b0.nii.gz'), '0', '1'], check=True)
                    # Create a acqparam.txt file in INPUTS
                    with open(os.path.join(synb0_input_path, 'acqparam.txt'), 'w') as f:
                        f.write(phase_encoding_number + ' ' + str(total_readout_time) + '\n')
                        f.write(phase_encoding_number + ' 0')

                    freesurfer_home = os.environ.get("FREESURFER_HOME")

                    fs_license = os.environ.get("FS_LICENSE")
                    if not fs_license:
                        raise ValueError("FS_LICENSE environment variable is not set.")
                    
                    subprocess.run([
                        "docker", "run", "--rm",
                        "-v", f"{synb0_input_path}:/INPUTS",
                        "-v", f"{synb0_output_path}:/OUTPUTS",
                        "-v", f"{fs_license}:/extra/freesurfer/license.txt",
                        "leonyichencai/synb0-disco:v3.1",
                        "--user", "1000:1000",
                        "--stripped",
                        "--notopup"
                    ], check=True)

                else:
                    print("Synb0-disco has already been run. Skip the process.")

                b0_u = os.path.join(synb0_output_path, 'b0_u.nii.gz')
                b0_d_smooth = os.path.join(synb0_output_path, 'b0_d_smooth.nii.gz')
                
                # merge the synthetic b0 image with the original dwi image
                subprocess.run(['fslmerge', '-t', os.path.join(output_path_synb0, 'b0_all.nii.gz'), b0_d_smooth, b0_u], check=True)
                # make a copy of b0_all, acqparam in self.output_path
                subprocess.run(['cp', os.path.join(output_path_synb0, 'b0_all.nii.gz'), output_path], check=True)
                subprocess.run(['cp', os.path.join(synb0_input_path, 'acqparam.txt'), output_path], check=True)

            else:
                # TODO
                print("Not implemented yet. Here should be the code for topup using the original b0 image.")

            subprocess.run([
                'bash', script_path_dtipreprocess,
                dwi_image, b0_all, acqparam, dwi_bval, dwi_bvec, output_path
            ], check=True)

        # make input directory for bedpostx
        bedpostx_input_dir = os.path.join(output_path, 'bedpostX_input')
        os.makedirs(bedpostx_input_dir, exist_ok=True)

        shutil.copy(os.path.join(output_path, 'eddy_corrected_data.nii.gz'), os.path.join(bedpostx_input_dir, 'data.nii.gz'))
        shutil.copy(os.path.join(output_path, 'dwi_b0_brain_mask.nii.gz'), os.path.join(bedpostx_input_dir, 'nodif_brain_mask.nii.gz'))
        shutil.copy(os.path.join(output_path, 'eddy_corrected_data.eddy_rotated_bvecs'), os.path.join(bedpostx_input_dir, 'bvecs'))
        shutil.copy(dwi_bval, os.path.join(bedpostx_input_dir, 'bvals'))
        
        self._fa_file = fa_file
        self._bedpostx_input_dir = bedpostx_input_dir
            
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['fa_file'] = self._fa_file
        outputs['bedpostx_input_dir'] = self._bedpostx_input_dir

        return outputs

############
# bedpostx #
############
class BedpostxInputSpec(CommandLineInputSpec):
    input_dir = Str(argstr='%s', position=0, desc='Input directory', mandatory=True)

class BedpostxOutputSpec(TraitedSpec):
    output_dir = Str(desc='Output directory')

class Bedpostx(CommandLine):
    _cmd = 'bedpostx_gpu'
    input_spec = BedpostxInputSpec
    output_spec = BedpostxOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_dir'] = self.inputs.input_dir + '.bedpostX'

        return outputs

################
# Tractography #
################
class TractographyInputSpec(BaseInterfaceInputSpec):
    fs_output = Directory(exists=True, desc='Freesurfer output directory')
    seed_mask_dtispace = File(exists=True, desc='Seed mask') # Currently only in DTI space
    fs_processing_dir = Directory(desc='Freesurfer processing directory')
    t1w_file = File(exists=True, desc='T1w file')
    fa_file = File(exists=True, desc='FA file')
    script_path_fspreprocess = Str(desc='Path to the script') # Freesurfer post-processing
    skip_fs_preprocess = Bool(False, desc='Skip freesurfer preprocessing')
    seed_mask_fsspace = File(desc='Seed mask in Freesurfer space')
    bedpostx_output_dir = Directory(desc='Bedpostx output directory')
    probtrackx_output_dir1 = Directory(desc='Probtrackx output directory 1 (Default)')
    probtrackx_output_dir2 = Directory(desc='Probtrackx output directory 2 (Corrected for path length)')

class TractographyOutputSpec(TraitedSpec):
    seed_mask_fsspace = File(desc='Seed mask in Freesurfer space')
    cortex_mask = File(desc='Cortex mask')
    fs_processing_dir = Directory(desc='Freesurfer processing directory')
    fs_orig = Directory(desc='orig.nii.gz')
    # BELOW: The masks are in fsaverage space
    lh_unconn_mask = File(desc='Unconnected mask')
    rh_unconn_mask = File(desc='Unconnected mask')
    lh_unconn_corrected_mask = File(desc='Unconnected mask corrected for path length')
    rh_unconn_corrected_mask = File(desc='Unconnected mask corrected for path length')
    lh_low_conn_mask = File(desc='Low connectivity mask')
    rh_low_conn_mask = File(desc='Low connectivity mask')
    lh_low_conn_corrected_mask = File(desc='Low connectivity mask corrected for path length')
    rh_low_conn_corrected_mask = File(desc='Low connectivity mask corrected for path length')
    lh_medium_conn_mask = File(desc='Medium connectivity mask')
    rh_medium_conn_mask = File(desc='Medium connectivity mask')
    lh_medium_conn_corrected_mask = File(desc='Medium connectivity mask corrected for path length')
    rh_medium_conn_corrected_mask = File(desc='Medium connectivity mask corrected for path length')
    lh_high_conn_mask = File(desc='High connectivity mask')
    rh_high_conn_mask = File(desc='High connectivity mask')
    lh_high_conn_corrected_mask = File(desc='High connectivity mask corrected for path length')
    rh_high_conn_corrected_mask = File(desc='High connectivity mask corrected for path length')

class Tractography(BaseInterface):
    input_spec = TractographyInputSpec
    output_spec = TractographyOutputSpec

    def _merge_wmgm_boundary_gifti(self, gii_mesh_path, gii_data_path, output_gii):
        """
        Use the mesh information of gii_mesh_path
        Use the data information of gii_data_path (here we binarize the data to create a mask)
        Merge the data information with the mesh information to create a new GIFTI file
        """
        gii_mesh = nib.load(gii_mesh_path)
        gii_data = nib.load(gii_data_path)
        
        vertices = gii_mesh.darrays[0].data
        faces = gii_mesh.darrays[1].data
        values = gii_data.darrays[0].data

        num_vertices = vertices.shape[0]
        assert values.shape[0] == num_vertices, "Error: Measure and mesh vertex counts do not match!"

        vertex_values = (values > 0).astype(np.float32)

        new_gii = nib.gifti.GiftiImage()
        new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            vertices, intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
        ))
        new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            faces, intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
        ))
        new_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(
            vertex_values, intent=nib.nifti1.intent_codes['NIFTI_INTENT_SHAPE']
        ))

        nib.save(new_gii, output_gii)
        print(f"Fixed GIFTI file saved as {output_gii}")

        return output_gii
    
    def _apply_gii_mask_to_mgh(self, measure_mgh_path, mask_gii_path, output_mgh_path):
        measure_img = nib.load(measure_mgh_path)
        measure_data = measure_img.get_fdata().squeeze()

        mask_gii = nib.load(mask_gii_path)
        mask_data = mask_gii.darrays[2].data

        if not np.all(np.isin(np.unique(mask_data), [0, 1])):
            print("Warning: Mask data is not binary. Set all non-zero values to 1.")
            mask_data = np.where(mask_data > 0, 1, mask_data)
        
        masked_data = measure_data * mask_data
        masked_img = nib.MGHImage(masked_data.astype(np.float32), measure_img.affine, measure_img.header)
        nib.save(masked_img, output_mgh_path)

        return masked_data
    
    def _generate_surface_masks(self, lh_path, rh_path, output_files, divisor, ignore_value=3.8e-5):
        """
        Generate High, Medium, Low, and Other surface masks from LH/RH MGH files.

        Parameters:
        - lh_path: str, path to left hemisphere .mgh file
        - rh_path: str, path to right hemisphere .mgh file
        - output_files: dict, expected keys: 
            'lh_High', 'lh_Medium', 'lh_Low', 'lh_Other',
            'rh_High', 'rh_Medium', 'rh_Low', 'rh_Other'
        - divisor: float, normalize surface values
        - ignore_value: float, ignore values below this threshold

        Returns:
        - Tuple of 8 file paths to the generated masks (LH/RH for each category)
        """

        lh_img = nib.load(lh_path)
        rh_img = nib.load(rh_path)

        lh_data = lh_img.get_fdata().squeeze() / divisor
        rh_data = rh_img.get_fdata().squeeze() / divisor

        # Combine and threshold
        combined_data = np.concatenate([lh_data, rh_data])
        filtered_data = combined_data[combined_data > ignore_value]
        sorted_values = np.sort(filtered_data)

        # Percentile thresholds
        thresh_High = sorted_values[int(len(sorted_values) * 0.75)]
        thresh_Medium = sorted_values[int(len(sorted_values) * 0.50)]
        thresh_Low = sorted_values[int(len(sorted_values) * 0.25)]

        def create_mask(data, threshold):
            return (data >= threshold).astype(int)

        # Create masks
        lh_mask_High = create_mask(lh_data, thresh_High)
        lh_mask_Medium = create_mask(lh_data, thresh_Medium)
        lh_mask_Low = create_mask(lh_data, thresh_Low)

        rh_mask_High = create_mask(rh_data, thresh_High)
        rh_mask_Medium = create_mask(rh_data, thresh_Medium)
        rh_mask_Low = create_mask(rh_data, thresh_Low)

        # Other masks = not in Low
        lh_mask_Unconn = (lh_mask_Low == 0).astype(int)
        rh_mask_Unconn = (rh_mask_Low == 0).astype(int)

        # Save function
        def save_mgh(data, ref_img, filename):
            img = nib.MGHImage(data.astype(np.float32), ref_img.affine, ref_img.header)
            nib.save(img, filename)

        # Save all masks
        save_mgh(lh_mask_High, lh_img, output_files["lh_High"])
        save_mgh(lh_mask_Medium, lh_img, output_files["lh_Medium"])
        save_mgh(lh_mask_Low, lh_img, output_files["lh_Low"])
        save_mgh(lh_mask_Unconn, lh_img, output_files["lh_Unconn"])

        save_mgh(rh_mask_High, rh_img, output_files["rh_High"])
        save_mgh(rh_mask_Medium, rh_img, output_files["rh_Medium"])
        save_mgh(rh_mask_Low, rh_img, output_files["rh_Low"])
        save_mgh(rh_mask_Unconn, rh_img, output_files["rh_Unconn"])

        return (
            output_files["lh_High"], output_files["rh_High"],
            output_files["lh_Medium"], output_files["rh_Medium"],
            output_files["lh_Low"], output_files["rh_Low"],
            output_files["lh_Unconn"], output_files["rh_Unconn"]
        )

    def _run_interface(self, runtime):
        fs_output = self.inputs.fs_output
        seed_mask_dtispace = self.inputs.seed_mask_dtispace
        fs_processing_dir = self.inputs.fs_processing_dir
        t1w_file = self.inputs.t1w_file
        fa_file = self.inputs.fa_file
        script_path_fspreprocess = self.inputs.script_path_fspreprocess
        bedpostx_output_dir = self.inputs.bedpostx_output_dir
        probtrackx_output_dir1 = self.inputs.probtrackx_output_dir1
        probtrackx_output_dir2 = self.inputs.probtrackx_output_dir2

        # Part I: Freesurfer processing
        if self.inputs.skip_fs_preprocess:
            # must provide seed_mask_fsspace
            self._seed_mask_fsspace = self.inputs.seed_mask_fsspace
            self._cortex_mask = os.path.join(fs_processing_dir, 'cortical_GM.nii.gz')
            if not os.path.exists(self._seed_mask_fsspace):
                raise ValueError("Seed mask in Freesurfer space is not provided. Please provide a valid seed mask.")
            
            lh_cortex_gii = os.path.join(fs_processing_dir, 'lh.cortex.gii')
            rh_cortex_gii = os.path.join(fs_processing_dir, 'rh.cortex.gii')

            lh_cortex_fsaverage_gii = os.path.join(fs_processing_dir, 'lh.cortex_fsaverage.gii')
            rh_cortex_fsaverage_gii = os.path.join(fs_processing_dir, 'rh.cortex_fsaverage.gii')
        else:
            subprocess.run([
                'bash', script_path_fspreprocess,
                fs_output, t1w_file,
                fa_file, seed_mask_dtispace, fs_processing_dir
            ])

            subprocess.run(['mris_convert', os.path.join(fs_output, 'surf', 'lh.white'), os.path.join(fs_processing_dir, 'lh.white.gii')])
            subprocess.run(['mris_convert', os.path.join(fs_output, 'surf', 'rh.white'), os.path.join(fs_processing_dir, 'rh.white.gii')])
            subprocess.run(['mris_convert', '--annot', os.path.join(fs_output, 'label', 'lh.aparc.annot'), 
                            os.path.join(fs_output, 'surf', 'lh.white'), os.path.join(fs_processing_dir, 'lh.aparc.label.gii')])
            subprocess.run(['mris_convert', '--annot', os.path.join(fs_output, 'label', 'rh.aparc.annot'),
                            os.path.join(fs_output, 'surf', 'rh.white'), os.path.join(fs_processing_dir, 'rh.aparc.label.gii')])
                        
            lh_cortex_gii = self._merge_wmgm_boundary_gifti(os.path.join(fs_processing_dir, 'lh.white.gii'), os.path.join(fs_processing_dir, 'lh.aparc.label.gii'), os.path.join(fs_processing_dir, 'lh.cortex.gii'))
            rh_cortex_gii = self._merge_wmgm_boundary_gifti(os.path.join(fs_processing_dir, 'rh.white.gii'), os.path.join(fs_processing_dir, 'rh.aparc.label.gii'), os.path.join(fs_processing_dir, 'rh.cortex.gii'))

            # delete the intermediate files
            files_to_delete = [os.path.join(fs_processing_dir, 'lh.white.gii'), os.path.join(fs_processing_dir, 'rh.white.gii'),
                            os.path.join(fs_processing_dir, 'lh.aparc.label.gii'), os.path.join(fs_processing_dir, 'rh.aparc.label.gii')]
            for file in files_to_delete:
                os.remove(file)

            # apply the same to generate rh.white_fsaverage.gii and lh.white_fsaverage.gii
            fsaverage_dir = os.path.join(os.environ.get("FREESURFER_HOME"), 'subjects', 'fsaverage')
            subprocess.run(['mris_convert', os.path.join(fsaverage_dir, 'surf', 'lh.white'), os.path.join(fs_processing_dir, 'lh.white_fsaverage.gii')])
            subprocess.run(['mris_convert', os.path.join(fsaverage_dir, 'surf', 'rh.white'), os.path.join(fs_processing_dir, 'rh.white_fsaverage.gii')])
            subprocess.run(['mris_convert', '--annot', os.path.join(fsaverage_dir, 'label', 'lh.aparc.annot'),
                            os.path.join(fsaverage_dir, 'surf', 'lh.white'), os.path.join(fs_processing_dir, 'lh.aparc_fsaverage.label.gii')])
            subprocess.run(['mris_convert', '--annot', os.path.join(fsaverage_dir, 'label', 'rh.aparc.annot'),
                            os.path.join(fsaverage_dir, 'surf', 'rh.white'), os.path.join(fs_processing_dir, 'rh.aparc_fsaverage.label.gii')])
            
            lh_cortex_fsaverage_gii = self._merge_wmgm_boundary_gifti(os.path.join(fs_processing_dir, 'lh.white_fsaverage.gii'), os.path.join(fs_processing_dir, 'lh.aparc_fsaverage.label.gii'), os.path.join(fs_processing_dir, 'lh.cortex_fsaverage.gii'))
            rh_cortex_fsaverage_gii = self._merge_wmgm_boundary_gifti(os.path.join(fs_processing_dir, 'rh.white_fsaverage.gii'), os.path.join(fs_processing_dir, 'rh.aparc_fsaverage.label.gii'), os.path.join(fs_processing_dir, 'rh.cortex_fsaverage.gii'))

            # delete the intermediate files
            files_to_delete = [os.path.join(fs_processing_dir, 'lh.white_fsaverage.gii'), os.path.join(fs_processing_dir, 'rh.white_fsaverage.gii'),
                                os.path.join(fs_processing_dir, 'lh.aparc_fsaverage.label.gii'), os.path.join(fs_processing_dir, 'rh.aparc_fsaverage.label.gii')]
            for file in files_to_delete:
                os.remove(file)

            # the txt file is used for FDT probtrackx2
            # Currently not used!
            stop_txt_path = os.path.join(fs_processing_dir, "stop.txt")

            with open(stop_txt_path, "w") as f:
                f.write(lh_cortex_gii + "\n")

            with open(stop_txt_path, "a") as f:
                f.write(rh_cortex_gii + "\n")
            
            self._seed_mask_fsspace = os.path.join(fs_processing_dir, 'seed_mask_in_fs.nii.gz')
            self._cortex_mask = os.path.join(fs_processing_dir, 'cortical_GM.nii.gz')

        # Part II: Probtrackx
        if not os.path.exists(os.path.join(probtrackx_output_dir1, 'fdt_paths.nii.gz')):
            subprocess.run(['probtrackx2_gpu',
                                '-s', os.path.join(bedpostx_output_dir, 'merged'),
                                '-m', os.path.join(bedpostx_output_dir, 'nodif_brain_mask.nii.gz'),
                                '-x', self._seed_mask_fsspace,
                                f"--xfm={os.path.join(fs_processing_dir, 'freesurfer2fa.mat')}",
                                f"--seedref={os.path.join(fs_processing_dir, 'orig.nii.gz')}",
                                '-l', '--forcedir', '--opd', '--nsamples=10000', '--ompl',
                                #'--pd',
                                f"--dir={probtrackx_output_dir1}",
                                f"--waypoints={os.path.join(fs_processing_dir, 'cortical_GM.nii.gz')}"], check=True)
        else:
            print("Probtrackx has already been run. Skip the process.")

        if not os.path.exists(os.path.join(probtrackx_output_dir2, 'fdt_paths.nii.gz')):
            subprocess.run(['probtrackx2_gpu',
                                '-s', os.path.join(bedpostx_output_dir, 'merged'),
                                '-m', os.path.join(bedpostx_output_dir, 'nodif_brain_mask.nii.gz'),
                                '-x', self._seed_mask_fsspace,
                                f"--xfm={os.path.join(fs_processing_dir, 'freesurfer2fa.mat')}",
                                f"--seedref={os.path.join(fs_processing_dir, 'orig.nii.gz')}",
                                '-l', '--forcedir', '--opd', '--nsamples=10000', '--ompl',
                                '--pd', # Correct for path length
                                f"--dir={probtrackx_output_dir2}",
                                f"--waypoints={os.path.join(fs_processing_dir, 'cortical_GM.nii.gz')}"], check=True)
        else:
            print("Probtrackx has already been run. Skip the process.")
        
        subjects_dir_temp = os.path.dirname(fs_output)
        subject_id = os.path.basename(fs_output)

        seed_mask_img = nib.load(self._seed_mask_fsspace)
        seed_mask_data = seed_mask_img.get_fdata()
        seed_number = np.count_nonzero(seed_mask_data)
        divisor = 10000 * seed_number

        # set the environment variable SUBJECTS_DIR
        os.environ["SUBJECTS_DIR"] = subjects_dir_temp
        
        probtrackx_output_dirs = [probtrackx_output_dir1, probtrackx_output_dir2]
        for probtrackx_output_dir in probtrackx_output_dirs:

            subprocess.run([
                'mri_vol2surf',
                '--mov', os.path.join(probtrackx_output_dir, 'fdt_paths.nii.gz'),
                '--regheader', subject_id,
                '--hemi', 'lh',
                '--o', os.path.join(probtrackx_output_dir, 'lh.fdt_paths.mgh'),
                '--projfrac', '0.5'
            ], check=True)

            self._apply_gii_mask_to_mgh(os.path.join(probtrackx_output_dir, 'lh.fdt_paths.mgh'), lh_cortex_gii, os.path.join(probtrackx_output_dir, 'lh.fdt_paths.mgh'))

            subprocess.run([
                'mri_vol2surf',
                '--mov', os.path.join(probtrackx_output_dir, 'fdt_paths.nii.gz'),
                '--regheader', subject_id,
                '--hemi', 'rh',
                '--o', os.path.join(probtrackx_output_dir, 'rh.fdt_paths.mgh'),
                '--projfrac', '0.5'
            ], check=True)

            self._apply_gii_mask_to_mgh(os.path.join(probtrackx_output_dir, 'rh.fdt_paths.mgh'), rh_cortex_gii, os.path.join(probtrackx_output_dir, 'rh.fdt_paths.mgh'))

            output_files_subject = {
                "lh_High": os.path.join(probtrackx_output_dir, 'lh.HighConn.mgh'),
                "rh_High": os.path.join(probtrackx_output_dir, 'rh.HighConn.mgh'),
                "lh_Medium": os.path.join(probtrackx_output_dir, 'lh.MediumConn.mgh'),
                "rh_Medium": os.path.join(probtrackx_output_dir, 'rh.MediumConn.mgh'),
                "lh_Low": os.path.join(probtrackx_output_dir, 'lh.LowConn.mgh'),
                "rh_Low": os.path.join(probtrackx_output_dir, 'rh.LowConn.mgh'),
                "lh_Unconn": os.path.join(probtrackx_output_dir, 'lh.UnConn.mgh'),
                "rh_Unconn": os.path.join(probtrackx_output_dir, 'rh.UnConn.mgh')
            }

            self._generate_surface_masks(
                os.path.join(probtrackx_output_dir, 'lh.fdt_paths.mgh'),
                os.path.join(probtrackx_output_dir, 'rh.fdt_paths.mgh'),
                output_files_subject,
                divisor,
                ignore_value=3.8e-5
            )

            # apply mask to unconn mask
            self._apply_gii_mask_to_mgh(os.path.join(probtrackx_output_dir, 'lh.UnConn.mgh'), lh_cortex_gii, os.path.join(probtrackx_output_dir, 'lh.UnConn.mgh'))
            self._apply_gii_mask_to_mgh(os.path.join(probtrackx_output_dir, 'rh.UnConn.mgh'), rh_cortex_gii, os.path.join(probtrackx_output_dir, 'rh.UnConn.mgh'))

            subprocess.run([
                'mri_vol2surf',
                '--mov', os.path.join(probtrackx_output_dir, 'fdt_paths.nii.gz'),
                '--regheader', subject_id,
                '--hemi', 'lh',
                '--o', os.path.join(probtrackx_output_dir, 'lh.fdt_paths_fsaverage.mgh'),
                '--projfrac', '0.5',
                '--trgsubject', 'fsaverage'
            ], check=True)

            self._apply_gii_mask_to_mgh(os.path.join(probtrackx_output_dir, 'lh.fdt_paths_fsaverage.mgh'), lh_cortex_fsaverage_gii, os.path.join(probtrackx_output_dir, 'lh.fdt_paths_fsaverage.mgh'))

            subprocess.run([
                'mri_vol2surf',
                '--mov', os.path.join(probtrackx_output_dir, 'fdt_paths.nii.gz'),
                '--regheader', subject_id,
                '--hemi', 'rh',
                '--o', os.path.join(probtrackx_output_dir, 'rh.fdt_paths_fsaverage.mgh'),
                '--projfrac', '0.5',
                '--trgsubject', 'fsaverage'
            ], check=True)

            self._apply_gii_mask_to_mgh(os.path.join(probtrackx_output_dir, 'rh.fdt_paths_fsaverage.mgh'), rh_cortex_fsaverage_gii, os.path.join(probtrackx_output_dir, 'rh.fdt_paths_fsaverage.mgh'))

            output_files_fsaverage = {
                "lh_High": os.path.join(probtrackx_output_dir, 'lh.HighConn_fsaverage.mgh'),
                "rh_High": os.path.join(probtrackx_output_dir, 'rh.HighConn_fsaverage.mgh'),
                "lh_Medium": os.path.join(probtrackx_output_dir, 'lh.MediumConn_fsaverage.mgh'),
                "rh_Medium": os.path.join(probtrackx_output_dir, 'rh.MediumConn_fsaverage.mgh'),
                "lh_Low": os.path.join(probtrackx_output_dir, 'lh.LowConn_fsaverage.mgh'),
                "rh_Low": os.path.join(probtrackx_output_dir, 'rh.LowConn_fsaverage.mgh'),
                "lh_Unconn": os.path.join(probtrackx_output_dir, 'lh.UnConn_fsaverage.mgh'),
                "rh_Unconn": os.path.join(probtrackx_output_dir, 'rh.UnConn_fsaverage.mgh')
            }

            self._generate_surface_masks(
                os.path.join(probtrackx_output_dir, 'lh.fdt_paths_fsaverage.mgh'),
                os.path.join(probtrackx_output_dir, 'rh.fdt_paths_fsaverage.mgh'),
                output_files_fsaverage,
                divisor,
                ignore_value=3.8e-5
            )

            # apply mask to unconn mask
            self._apply_gii_mask_to_mgh(os.path.join(probtrackx_output_dir, 'lh.UnConn_fsaverage.mgh'), lh_cortex_fsaverage_gii, os.path.join(probtrackx_output_dir, 'lh.UnConn_fsaverage.mgh'))
            self._apply_gii_mask_to_mgh(os.path.join(probtrackx_output_dir, 'rh.UnConn_fsaverage.mgh'), rh_cortex_fsaverage_gii, os.path.join(probtrackx_output_dir, 'rh.UnConn_fsaverage.mgh'))

        self._fs_processing_dir = fs_processing_dir
        self._fs_orig = os.path.join(fs_processing_dir, 'orig.nii.gz')

        self._lh_low_conn_mask = os.path.join(probtrackx_output_dir1, 'lh.LowConn_fsaverage.mgh')
        self._rh_low_conn_mask = os.path.join(probtrackx_output_dir1, 'rh.LowConn_fsaverage.mgh')
        self._lh_low_conn_corrected_mask = os.path.join(probtrackx_output_dir2, 'lh.LowConn_fsaverage.mgh')
        self._rh_low_conn_corrected_mask = os.path.join(probtrackx_output_dir2, 'rh.LowConn_fsaverage.mgh')
        self._lh_medium_conn_mask = os.path.join(probtrackx_output_dir1, 'lh.MediumConn_fsaverage.mgh')
        self._rh_medium_conn_mask = os.path.join(probtrackx_output_dir1, 'rh.MediumConn_fsaverage.mgh')
        self._lh_medium_conn_corrected_mask = os.path.join(probtrackx_output_dir2, 'lh.MediumConn_fsaverage.mgh')
        self._rh_medium_conn_corrected_mask = os.path.join(probtrackx_output_dir2, 'rh.MediumConn_fsaverage.mgh')
        self._lh_high_conn_mask = os.path.join(probtrackx_output_dir1, 'lh.HighConn_fsaverage.mgh')
        self._rh_high_conn_mask = os.path.join(probtrackx_output_dir1, 'rh.HighConn_fsaverage.mgh')
        self._lh_high_conn_corrected_mask = os.path.join(probtrackx_output_dir2, 'lh.HighConn_fsaverage.mgh')
        self._rh_high_conn_corrected_mask = os.path.join(probtrackx_output_dir2, 'rh.HighConn_fsaverage.mgh')
        self._lh_unconn_mask = os.path.join(probtrackx_output_dir1, 'lh.UnConn_fsaverage.mgh')
        self._rh_unconn_mask = os.path.join(probtrackx_output_dir1, 'rh.UnConn_fsaverage.mgh')
        self._lh_unconn_corrected_mask = os.path.join(probtrackx_output_dir2, 'lh.UnConn_fsaverage.mgh')
        self._rh_unconn_corrected_mask = os.path.join(probtrackx_output_dir2, 'rh.UnConn_fsaverage.mgh')

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['seed_mask_fsspace'] = self._seed_mask_fsspace
        outputs['cortex_mask'] = self._cortex_mask
        outputs['fs_processing_dir'] = self._fs_processing_dir
        outputs['fs_orig'] = self._fs_orig
        outputs['lh_low_conn_mask'] = self._lh_low_conn_mask
        outputs['rh_low_conn_mask'] = self._rh_low_conn_mask
        outputs['lh_low_conn_corrected_mask'] = self._lh_low_conn_corrected_mask
        outputs['rh_low_conn_corrected_mask'] = self._rh_low_conn_corrected_mask
        outputs['lh_medium_conn_mask'] = self._lh_medium_conn_mask
        outputs['rh_medium_conn_mask'] = self._rh_medium_conn_mask
        outputs['lh_medium_conn_corrected_mask'] = self._lh_medium_conn_corrected_mask
        outputs['rh_medium_conn_corrected_mask'] = self._rh_medium_conn_corrected_mask
        outputs['lh_high_conn_mask'] = self._lh_high_conn_mask
        outputs['rh_high_conn_mask'] = self._rh_high_conn_mask
        outputs['lh_high_conn_corrected_mask'] = self._lh_high_conn_corrected_mask
        outputs['rh_high_conn_corrected_mask'] = self._rh_high_conn_corrected_mask
        outputs['lh_unconn_mask'] = self._lh_unconn_mask
        outputs['rh_unconn_mask'] = self._rh_unconn_mask
        outputs['lh_unconn_corrected_mask'] = self._lh_unconn_corrected_mask
        outputs['rh_unconn_corrected_mask'] = self._rh_unconn_corrected_mask

        return outputs

##############################
# Extract Surface Parameters #
##############################
# nipype interface of mri_segstats
class MRIsegstatsInputSpec(CommandLineInputSpec):
    segvol = Str(argstr='--seg %s', position=0, desc='Segmentation volume', mandatory=True)
    invol = Str(argstr='--in %s', position=1, desc='Input volume', mandatory=True)
    sum = Str(argstr='--sum %s', position=2, desc='Output summary file', mandatory=True)

class MRIsegstatsOutputSpec(TraitedSpec):
    sum = Str(desc='Output summary file')

class MRIsegstats(CommandLine):
    _cmd = 'mri_segstats'
    input_spec = MRIsegstatsInputSpec
    output_spec = MRIsegstatsOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['sum'] = self.inputs.sum

        return outputs

####################################################################
# Extracrt Surface Parameters (specifically for Tractography Node) #
####################################################################
class ExtractSurfaceParametersInputSpec(BaseInterfaceInputSpec):
    fs_subjects_dir = Directory(exists=True, desc='Freesurfer subjects directory')
    sessions = traits.List(desc='List of sessions')
    lh_unconn_mask = File(exists=True, desc='Unconnected mask (left hemisphere)')
    rh_unconn_mask = File(exists=True, desc='Unconnected mask (right hemisphere)')
    lh_unconn_corrected_mask = File(exists=True, desc='Unconnected mask corrected for path length (left hemisphere)')
    rh_unconn_corrected_mask = File(exists=True, desc='Unconnected mask corrected for path length (right hemisphere)')
    lh_low_conn_mask = File(exists=True, desc='Low connectivity mask (left hemisphere)')
    rh_low_conn_mask = File(exists=True, desc='Low connectivity mask (right hemisphere)')
    lh_low_conn_corrected_mask = File(exists=True, desc='Low connectivity mask corrected for path length (left hemisphere)')
    rh_low_conn_corrected_mask = File(exists=True, desc='Low connectivity mask corrected for path length (right hemisphere)')
    lh_medium_conn_mask = File(exists=True, desc='Medium connectivity mask (left hemisphere)')
    rh_medium_conn_mask = File(exists=True, desc='Medium connectivity mask (right hemisphere)')
    lh_medium_conn_corrected_mask = File(exists=True, desc='Medium connectivity mask corrected for path length (left hemisphere)')
    rh_medium_conn_corrected_mask = File(exists=True, desc='Medium connectivity mask corrected for path length (right hemisphere)')
    lh_high_conn_mask = File(exists=True, desc='High connectivity mask (left hemisphere)')
    rh_high_conn_mask = File(exists=True, desc='High connectivity mask (right hemisphere)')
    lh_high_conn_corrected_mask = File(exists=True, desc='High connectivity mask corrected for path length (left hemisphere)')
    rh_high_conn_corrected_mask = File(exists=True, desc='High connectivity mask corrected for path length (right hemisphere)')
    output_dir = Str(desc='Output directory')
    csv_file_name = Str(desc='Output CSV file name', default_value='surface_parameters.csv')

class ExtractSurfaceParametersOutputSpec(TraitedSpec):
    csv_file = File(desc='Output CSV file')

class ExtractSurfaceParameters(BaseInterface):
    input_spec = ExtractSurfaceParametersInputSpec
    output_spec = ExtractSurfaceParametersOutputSpec

    def _run_interface(self, runtime):
        fs_subjects_dir = self.inputs.fs_subjects_dir
        sessions = self.inputs.sessions
        output_dir = self.inputs.output_dir

        paired_roi_masks = {
            "unconn": [self.inputs.lh_unconn_mask, self.inputs.rh_unconn_mask],
            "unconn_corrected": [self.inputs.lh_unconn_corrected_mask, self.inputs.rh_unconn_corrected_mask],
            "low_conn": [self.inputs.lh_low_conn_mask, self.inputs.rh_low_conn_mask],
            "low_conn_corrected": [self.inputs.lh_low_conn_corrected_mask, self.inputs.rh_low_conn_corrected_mask],
            "medium_conn": [self.inputs.lh_medium_conn_mask, self.inputs.rh_medium_conn_mask],
            "medium_conn_corrected": [self.inputs.lh_medium_conn_corrected_mask, self.inputs.rh_medium_conn_corrected_mask],
            "high_conn": [self.inputs.lh_high_conn_mask, self.inputs.rh_high_conn_mask],
            "high_conn_corrected": [self.inputs.lh_high_conn_corrected_mask, self.inputs.rh_high_conn_corrected_mask]
        }

        fwhm_values = [0, 5, 10, 15, 20, 25]

        measure_files = [
            "area", "area.pial", "curv", "jacobian_white", "sulc",
            "thickness", "volume", "w-g.pct.mgh", "white.H", "white.K"
        ]

        results = []

        # walk through sessions
        for session in sessions:
            session_surf_dir = os.path.join(fs_subjects_dir, f"ses-{session}", 'surf')
            session_data = {"session": f'ses-{session}'}

            if not os.path.exists(session_surf_dir):
                continue
            
            for measure in measure_files:
                for fwhm in fwhm_values:
                    lh_measure_path = os.path.join(session_surf_dir, f"lh.{measure}.fwhm{fwhm}.fsaverage.mgh")
                    rh_measure_path = os.path.join(session_surf_dir, f"rh.{measure}.fwhm{fwhm}.fsaverage.mgh")

                    for roi_name, roi_masks in paired_roi_masks.items():
                        lh_roi_mask = roi_masks[0]
                        rh_roi_mask = roi_masks[1]

                        lh_measure_img = nib.load(lh_measure_path)
                        rh_measure_img = nib.load(rh_measure_path)
                        lh_roi_img = nib.load(lh_roi_mask)
                        rh_roi_img = nib.load(rh_roi_mask)

                        lh_measure_data = lh_measure_img.get_fdata().squeeze()  # (N,)
                        rh_measure_data = rh_measure_img.get_fdata().squeeze()  # (N,)
                        lh_roi_data = lh_roi_img.get_fdata().squeeze()          # (N,)
                        rh_roi_data = rh_roi_img.get_fdata().squeeze()          # (N,)

                        if not np.all(np.equal(np.mod(lh_roi_data, 1), 0)) or not np.all(np.equal(np.mod(rh_roi_data, 1), 0)):
                            raise ValueError("ROI MGH files contain non-integer values. Please check the input files.")
                        
                        combined_measure_data = np.concatenate([lh_measure_data, rh_measure_data])
                        combined_roi_data = np.concatenate([lh_roi_data, rh_roi_data])

                        unique_rois = np.unique(combined_roi_data)
                        roi_avg_values = {}

                        for roi in unique_rois:
                            roi_mask = combined_roi_data == roi
                            roi_avg_values[int(roi)] = np.mean(combined_measure_data[roi_mask])
                        
                        mean_roi_1 = roi_avg_values.get(1, np.nan)
                        colname = f"{roi_name}_{measure}.fwhm{fwhm}"
                        session_data[colname] = mean_roi_1

                        #print(f"Session {session}, {colname}: {mean_roi_1}")

            results.append(session_data)

        df = pd.DataFrame(results)
        self._csv_file = os.path.join(output_dir, self.inputs.csv_file_name)
        df.to_csv(self._csv_file, index=False)

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['csv_file'] = self._csv_file

        return outputs

############
# DTI-ALPS # (Deprecated)
############
class DTIALPSInputSpec(CommandLineInputSpec):
    # -a 4d nifti file
    input_dwi = Str(argstr='-a %s', position=0, desc='Input 4D nifti file', mandatory=False)
    # -b bval
    bval = Str(argstr='-b %s', position=1, desc='bval', mandatory=False)
    # -c bvec
    bvec = Str(argstr='-c %s', position=2, desc='bvec', mandatory=False)
    # -m json
    json = Str(argstr='-m %s', position=3, desc='json', mandatory=False)
    # -i a second dwi file
    input_dwi2 = Str(argstr='-i %s', position=4, desc='A second dwi file', mandatory=False)
    # -j bval
    bval2 = Str(argstr='-j %s', position=5, desc='bval', mandatory=False)
    # -k bvec
    bvec2 = Str(argstr='-k %s', position=6, desc='bvec', mandatory=False)
    # -m json
    json2 = Str(argstr='-m %s', position=7, desc='json', mandatory=False)
    # -d 0=skip, 1=both denoising and unringing, 3=only denoising, 4=only unringing
    preprocessing_steps = Str(argstr='-d %s', position=8, desc='Preprocessing steps', mandatory=False)
    # -e 0=skip eddy, 1=eddy_cpu, 2=eddy, 3=eddy_correct
    eddy = Str(argstr='-e %s', position=9, desc='Eddy', mandatory=False)
    # -r 0=skip ROI analysis, 1=do ROI analysis
    roi_analysis = Str(argstr='-r %s', position=10, desc='ROI analysis', mandatory=False)
    # -t 0=native, 1=JHU-ICBM
    template = Str(argstr='-t %s', position=11, desc='Template', mandatory=False)
    # -v volume structure file
    volume_file = Str(argstr='-v %s', position=12, desc='Volume structure file', mandatory=False)
    # -h 1=t1w, 2=t2w
    struc_type = Str(argstr='-h %s', position=13, desc='Structural file modality', mandatory=False)
    # -w 0=linear, 1=nonlinear, 2=both
    warp = Str(argstr='-w %s', position=14, desc='Warp', mandatory=False)
    # -f 1=flirt or applywrap, 2=vecreg
    tensor_transform = Str(argstr='-f %s', position=15, desc='Tensor transform', mandatory=False)
    # -o output directory
    output_dir = Directory(exists=True, argstr='-o %s', position=16, desc='Output directory', mandatory=True)
    # -s 0=do, 1=skip preprocessing
    skip_preprocessing = Str(argstr='-s %s', position=17, desc='Skip preprocessing', mandatory=False)

class DTIALPSOutputSpec(TraitedSpec):
    # will generate a folder 'alps.stat' in the output directory
    output_dir = Directory(exists=True, desc='Output directory')
    alps_stat = Str(desc='ALPS statistics')

class DTIALPS(CommandLine):
    _cmd = 'alps.sh'
    input_spec = DTIALPSInputSpec
    output_spec = DTIALPSOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['alps_stat'] = os.path.join(self.inputs.output_dir, 'alps.stat')
        outputs['output_dir'] = self.inputs.output_dir
        return outputs

def input_for_alps(dti_fa, dti_md, dti_tensor, input_dir, alps_script_path):
    import os
    import subprocess
    os.makedirs(input_dir, exist_ok=True)

    alps_dir = os.path.dirname(alps_script_path)
    os.environ["PATH"] = os.environ.get("PATH", "") + ":" + alps_dir

    # copy the files to the input directory
    fa_file = subprocess.run(['cp', dti_fa, input_dir], check=True)
    md_file = subprocess.run(['cp', dti_md, input_dir], check=True)
    tensor_file = subprocess.run(['cp', dti_tensor, input_dir], check=True)

    return fa_file, md_file, tensor_file, input_dir

def delete_alps_inputs(input_dir):
    import os
    import subprocess
    files_to_delete = [os.path.join(input_dir, 'dti_FA.nii.gz'), os.path.join(input_dir, 'dti_MD.nii.gz'), os.path.join(input_dir, 'dti_tensor.nii.gz')]
    for file in files_to_delete:
        os.remove(file)
    
    return files_to_delete

####################################
# DTI-ALPS simple python interface #
####################################
class DTIALPSsimpleInputSpec(BaseInterfaceInputSpec):
    perform_roi_analysis = Str(desc='Perform ROI analysis', default_value='1')
    use_templete = Str(desc='Use template', default_value='1')
    dtifit_output_dir = Directory(exists=True, desc='DTIFIT output directory')
    alps_input_dir = Directory(desc='ALPS input directory')
    skip_preprocessing = Str(desc='Skip preprocessing', default_value='0')
    alps_script_path = Str(desc='Path to the alps script')

class DTIALPSsimpleOutputSpec(TraitedSpec):
    alps_stat = Str(desc='ALPS statistics')

class DTIALPSsimple(BaseInterface):
    input_spec = DTIALPSsimpleInputSpec
    output_spec = DTIALPSsimpleOutputSpec

    def _run_interface(self, runtime):
        perform_roi_analysis = self.inputs.perform_roi_analysis
        use_templete = self.inputs.use_templete
        dtifit_output_dir = self.inputs.dtifit_output_dir
        alps_input_dir = self.inputs.alps_input_dir
        skip_preprocessing = self.inputs.skip_preprocessing
        alps_script_path = self.inputs.alps_script_path

        os.makedirs(alps_input_dir, exist_ok=True)
        # copy the files to the input directory
        files_to_copy = [
            os.path.join(dtifit_output_dir, 'dti_FA.nii.gz'),
            os.path.join(dtifit_output_dir, 'dti_MD.nii.gz'),
            os.path.join(dtifit_output_dir, 'dti_tensor.nii.gz')
        ]

        for file in files_to_copy:
            subprocess.run(['cp', file, alps_input_dir], check=True)

        subprocess.run([
            'bash', alps_script_path,
            '-s', skip_preprocessing,
            '-r', perform_roi_analysis,
            '-t', use_templete,
            '-o', alps_input_dir
        ])

        # delete the input files
        files_to_delete = [
            os.path.join(alps_input_dir, 'dti_FA.nii.gz'),
            os.path.join(alps_input_dir, 'dti_MD.nii.gz'),
            os.path.join(alps_input_dir, 'dti_tensor.nii.gz')
        ]

        for file in files_to_delete:
            os.remove(file)

        self._alps_stat = os.path.join(dtifit_output_dir, 'alps.stat')

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['alps_stat'] = self._alps_stat

        return outputs