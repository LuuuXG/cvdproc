import os
import re
import nibabel as nib
import numpy as np
import pandas as pd
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, Directory, File, InputMultiPath
from neuromaps import transforms
from traits.api import Either

from cvdproc.config.paths import LH_MEDIAL_WALL_TXT, RH_MEDIAL_WALL_TXT, find_qcache_metric_pairs

class NemoPostprocessInputSpec(BaseInterfaceInputSpec):
    nemo_output_dir = Directory(exists=True, mandatory=True, desc='Directory containing Nemo output files')
    nemo_postprocessed_dir = Directory(mandatory=True, desc='Directory to save postprocessed Nemo output files')
    freesurfer_output_dirs = InputMultiPath(Directory(exists=True), mandatory=False, desc='List of Freesurfer output directories')
    output_csv_dir = Directory(mandatory=False, desc='Directory to save one CSV per chacovol file')

class NemoPostprocessOutputSpec(TraitedSpec):
    nemo_postprocessed_dir = Directory(exists=True, desc='Directory containing postprocessed Nemo output files')
    output_csv_dir = Directory(desc='Directory containing one CSV per chacovol file')

class NemoPostprocess(BaseInterface):
    input_spec = NemoPostprocessInputSpec
    output_spec = NemoPostprocessOutputSpec

    def _run_interface(self, runtime):
        nemo_output_dir = self.inputs.nemo_output_dir
        nemo_postprocessed_dir = self.inputs.nemo_postprocessed_dir

        if not os.path.exists(nemo_postprocessed_dir):
            os.makedirs(nemo_postprocessed_dir)

        lh_medial_wall = np.loadtxt(LH_MEDIAL_WALL_TXT, dtype=int)
        rh_medial_wall = np.loadtxt(RH_MEDIAL_WALL_TXT, dtype=int)

        # Step 1: transform all MNI nii.gz files to fsaverage.func.gii
        mni_images = [f for f in os.listdir(nemo_output_dir) if f.endswith('.nii.gz')]
        for mni_image in mni_images:
            mni_image_path = os.path.join(nemo_output_dir, mni_image)
            fsaverage_lh, fsaverage_rh = transforms.mni152_to_fsaverage(mni_image_path, '164k')

            lh_data = fsaverage_lh.darrays[0].data
            rh_data = fsaverage_rh.darrays[0].data
            lh_data[lh_medial_wall] = None
            rh_data[rh_medial_wall] = None
            fsaverage_lh.darrays[0].data[:] = lh_data
            fsaverage_rh.darrays[0].data[:] = rh_data

            lh_filename = mni_image.replace('.nii.gz', '_lh_fsaverage.func.gii')
            rh_filename = mni_image.replace('.nii.gz', '_rh_fsaverage.func.gii')
            fsaverage_lh.to_filename(os.path.join(nemo_postprocessed_dir, lh_filename))
            fsaverage_rh.to_filename(os.path.join(nemo_postprocessed_dir, rh_filename))
            print(f"Transformed {mni_image} to fsaverage space and saved as {lh_filename} and {rh_filename}")

        if self.inputs.freesurfer_output_dirs:
            fs_dirs = self.inputs.freesurfer_output_dirs
            fsavg_files = os.listdir(nemo_postprocessed_dir)
            lh_pattern = re.compile(r'(.+_chacovol_res\d+mm_smooth\d+mm_mean)_lh_fsaverage\.func\.gii')
            lh_prob_maps = [f for f in fsavg_files if lh_pattern.match(f)]

            for lh_file in lh_prob_maps:
                match = lh_pattern.match(lh_file)
                base_id = match.group(1)
                rh_file = f"{base_id}_rh_fsaverage.func.gii"
                lh_path = os.path.join(nemo_postprocessed_dir, lh_file)
                rh_path = os.path.join(nemo_postprocessed_dir, rh_file)
                if not os.path.exists(rh_path):
                    continue

                lh_data = nib.load(lh_path).darrays[0].data
                rh_data = nib.load(rh_path).darrays[0].data
                weight = np.concatenate([lh_data, rh_data])

                rows = []
                for fs_dir in fs_dirs:
                    metric_pairs = find_qcache_metric_pairs(fs_dir)
                    row = {'freesurfer_dir': os.path.basename(fs_dir.rstrip('/'))}
                    for metric_key, hemis in metric_pairs.items():
                        lh_metric = nib.load(hemis['lh']).get_fdata().squeeze()
                        rh_metric = nib.load(hemis['rh']).get_fdata().squeeze()
                        metric_data = np.concatenate([lh_metric, rh_metric])

                        valid = (~np.isnan(weight)) & (~np.isnan(metric_data))
                        wsum = np.nansum(weight[valid])
                        wmean = np.nan if wsum == 0 else np.nansum(weight[valid] * metric_data[valid]) / wsum
                        row[metric_key] = wmean
                    rows.append(row)

                if self.inputs.output_csv_dir:
                    os.makedirs(self.inputs.output_csv_dir, exist_ok=True)
                    output_path = os.path.join(self.inputs.output_csv_dir, f"{base_id}_cortical_metrics.csv")
                    df = pd.DataFrame(rows)
                    df.to_csv(output_path, index=False)
                    print(f"Saved {base_id} results to {output_path}")

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['nemo_postprocessed_dir'] = self.inputs.nemo_postprocessed_dir
        outputs['output_csv_dir'] = self.inputs.output_csv_dir if self.inputs.output_csv_dir else None
        return outputs

if __name__ == "__main__":
    lh_medial_wall = np.loadtxt(LH_MEDIAL_WALL_TXT, dtype=int)
    rh_medial_wall = np.loadtxt(RH_MEDIAL_WALL_TXT, dtype=int)

    print("LH Medial Wall:", lh_medial_wall)
    print("RH Medial Wall:", rh_medial_wall)