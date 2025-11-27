import os
import subprocess
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, Directory, traits, CommandLineInputSpec, File, TraitedSpec, CommandLine, Directory
from nipype.interfaces.utility import IdentityInterface
from traits.api import Bool, Int, Str
import gzip

from cvdproc.bids_data.rename_bids_file import rename_bids_file

class PVeDInputSpec(BaseInterfaceInputSpec):
    qsdr_fib_file = File(exists=True, desc='Path to the QSDR fib file')
    output_dir = Str(desc='Output directory')
    script_path = Str(desc='Path to the MATLAB script for PVeD estimation', mandatory=True)
    spm_path = Str(desc='Path to SPM installation', mandatory=True)
    pved_path = Str(desc='Path to the PVeD MATLAB functions', mandatory=True)

class PVeDOutputSpec(TraitedSpec):
    # % output:
    # % 1. [fib_name].md.nii: extracted mean diffusivity maps from fib files
    # % 2. csf_mask_[fib_name].md.nii: segmented CSF masks
    # % 3. lv_mask_[fib_name].md.nii: estimated lateral ventricle masks
    # % 4. final_pvs_[fib_name].md.nii: estimated periventricular area masks
    # % 5. ttr_[fib_name].md.nii: estimated TTR maps
    # % 6. PVeD_metrics.csv: output table with the estimated metrics
    md_map = File(desc='Mean diffusivity map')
    pvs_mask = File(desc='Periventricular area mask')
    ttr_map = File(desc='TTR map')
    metrics_csv = File(desc='CSV file with the estimated metrics')

class PVeD(BaseInterface):
    input_spec = PVeDInputSpec
    output_spec = PVeDOutputSpec

    def _run_interface(self, runtime):
        qsdr_fib_file = self.inputs.qsdr_fib_file
        output_dir = self.inputs.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # copy the fib file to output_dir
        fib_dir = os.path.join(output_dir, 'fib_dir')
        os.makedirs(fib_dir, exist_ok=True)
        shutil.copy(qsdr_fib_file, fib_dir)
        
        with open(self.inputs.script_path) as script_file:
            script_content = script_file.read()
        
        subject_matlab_script = os.path.join(self.inputs.output_dir, 'pved_script.m')
        with open(subject_matlab_script, 'w') as f:
            new_script_content = script_content
            new_script_content = new_script_content.replace('/this/is/for/nipype/spm_path', self.inputs.spm_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/pved_path', self.inputs.pved_path)
            new_script_content = new_script_content.replace('/this/is/for/nipype/fib_dir', fib_dir)
            f.write(new_script_content)
        
        cmd_str = f"run('{subject_matlab_script}'); exit;"
        mlab = CommandLine('matlab', args=f"-nodisplay -nosplash -nodesktop -r \"{cmd_str}\"", terminal_output='stream')
        result = mlab.run()

        # outputs will be in the fib_dir, move them to output_dir
        md_map = os.path.join(fib_dir, os.path.basename(qsdr_fib_file)+'.md.nii')
        csf_mask = os.path.join(fib_dir, f"csf_mask_{os.path.basename(qsdr_fib_file)+'.md.nii'}")
        lv_mask = os.path.join(fib_dir, f"lv_mask_{os.path.basename(qsdr_fib_file)+'.md.nii'}")
        pvs_mask = os.path.join(fib_dir, f"final_pvs_{os.path.basename(qsdr_fib_file)+'.md.nii'}")
        ttr_map = os.path.join(fib_dir, f"ttr_{os.path.basename(qsdr_fib_file)+'.md.nii'}")
        seg8_mat = os.path.join(fib_dir, os.path.basename(qsdr_fib_file)+'.md_seg8.mat')
        metrics_csv = os.path.join(fib_dir, "PVeD_metrics.csv") 

        # rename and gzip the outputs (.nii to .nii.gz)
        if os.path.exists(md_map):
            new_md_map = os.path.join(output_dir, rename_bids_file(qsdr_fib_file, {'model': 'tensor', 'param': 'md', 'space': 'MNI'}, 'dwimap', '.nii.gz'))
            # gzip the file
            subprocess.run(['gzip', '-c', md_map], stdout=open(new_md_map, 'wb'))
            os.remove(md_map)
        if os.path.exists(csf_mask):
            new_csf_mask = os.path.join(output_dir, rename_bids_file(qsdr_fib_file, {'model': None, 'label': 'csf'}, 'mask', '.nii.gz'))
            #subprocess.run(['gzip', '-c', csf_mask], stdout=open(new_csf_mask, 'wb'))
            os.remove(csf_mask)
        if os.path.exists(lv_mask):
            new_lv_mask = os.path.join(output_dir, rename_bids_file(qsdr_fib_file, {'model': None, 'label': 'LateralVentricle'}, 'mask', '.nii.gz'))
            #subprocess.run(['gzip', '-c', lv_mask], stdout=open(new_lv_mask, 'wb'))
            os.remove(lv_mask)
        if os.path.exists(pvs_mask):
            new_pvs_mask = os.path.join(output_dir, rename_bids_file(qsdr_fib_file, {'model': None, 'label': 'PeriventricularArea', 'space': 'MNI'}, 'mask', '.nii.gz'))
            subprocess.run(['gzip', '-c', pvs_mask], stdout=open(new_pvs_mask, 'wb'))
            os.remove(pvs_mask)
        if os.path.exists(ttr_map):
            new_ttr_map = os.path.join(output_dir, rename_bids_file(qsdr_fib_file, {'model': 'tensor', 'param': 'ttr', 'space': 'MNI'}, 'dwimap', '.nii.gz'))
            subprocess.run(['gzip', '-c', ttr_map], stdout=open(new_ttr_map, 'wb'))
            os.remove(ttr_map)
        if os.path.exists(seg8_mat):
            os.remove(seg8_mat)
        if os.path.exists(metrics_csv):
            new_metrics_csv = os.path.join(output_dir, "PVeD_metrics.csv")
            shutil.move(metrics_csv, new_metrics_csv)
        
        # delete the fib_dir
        shutil.rmtree(fib_dir)
            
        return result.runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        qsdr_fib_file = self.inputs.qsdr_fib_file
        output_dir = self.inputs.output_dir

        outputs['md_map'] = os.path.join(output_dir, rename_bids_file(qsdr_fib_file, {'model': 'tensor', 'param': 'md', 'space': 'MNI'}, 'dwimap', '.nii.gz'))
        outputs['pvs_mask'] = os.path.join(output_dir, rename_bids_file(qsdr_fib_file, {'model': None, 'label': 'PeriventricularArea', 'space': 'MNI'}, 'mask', '.nii.gz'))
        outputs['ttr_map'] = os.path.join(output_dir, rename_bids_file(qsdr_fib_file, {'model': 'tensor', 'param': 'ttr', 'space': 'MNI'}, 'dwimap', '.nii.gz'))
        outputs['metrics_csv'] = os.path.join(output_dir, "PVeD_metrics.csv")

        return outputs

if __name__ == '__main__':
    # Example usage
    pved = PVeD()
    pved.inputs.qsdr_fib_file = '/mnt/f/BIDS/ALL/derivatives/dwi_pipeline/sub-WZCU002/ses-01/dsistudio/sub-WZCU002_ses-01_acq-DTIb1000_space-preprocdwi_model-qsdr_dwimap.fib.gz'
    pved.inputs.output_dir = '/mnt/f/BIDS/ALL/derivatives/dwi_pipeline/sub-WZCU002/ses-01/PVeD'
    pved.inputs.script_path = '/mnt/e/codes/cvdproc/cvdproc/pipelines/matlab/pved/pved_single.m'
    pved.inputs.spm_path = '/mnt/e/codes/cvdproc/cvdproc/data/matlab_toolbox/spm12'
    pved.inputs.pved_path = '/mnt/e/codes/cvdproc/cvdproc/data/matlab_toolbox/EstPVeD/MATLAB/pkg_pved_est'
    pved.run()
    print(pved._list_outputs())
