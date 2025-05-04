import os
import shutil
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, CommandLine, BaseInterfaceInputSpec, BaseInterface
from traits.api import Str, Int, Directory
import subprocess
from ....bids_data.rename_bids_file import rename_bids_file

#########################
# recon-all-clinical.sh #
#########################
# https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all-clinical

class ReconAllClinicalInputSpec(CommandLineInputSpec):
    input_scan = File(exists=True, mandatory=True, argstr="%s", position=0, desc="Input scan (NIfTI or MGZ)")
    subject_id = Str(mandatory=True, argstr="%s", position=1, desc="Subject ID")
    threads = Int(1, usedefault=True, argstr="%d", position=2, desc="Number of threads")
    subject_dir = Directory(argstr="%s", position=3, desc="Freesurfer SUBJECTS_DIR (optional)")

class ReconAllClinicalOutputSpec(TraitedSpec):
    output_dir = Directory(desc='Freesurfer output directory')
    synthsr_raw = Str(desc='raw output of SynthSR')
    synthsr_norm = Str(desc='cleaned up version of synthSR.raw.mgz, scaled such that the white matter has intensity of 110')

class ReconAllClinical(CommandLine):
    _cmd = 'recon-all-clinical.sh'
    input_spec = ReconAllClinicalInputSpec
    output_spec = ReconAllClinicalOutputSpec
    terminal_output = 'allatonce'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.subject_dir:
            outputs['output_dir'] = os.path.join(self.inputs.subject_dir, self.inputs.subject_id)
            outputs['synthsr_raw'] = os.path.join(outputs['output_dir'], 'mri', 'synthSR.raw.mgz')
            outputs['synthsr_norm'] = os.path.join(outputs['output_dir'], 'mri', 'synthSR.norm.mgz')
        else:
            outputs['output_dir'] = os.path.join(os.environ.get('SUBJECTS_DIR', ''), self.inputs.subject_id)
            outputs['synthsr_raw'] = os.path.join(outputs['output_dir'], 'mri', 'synthSR.raw.mgz')
            outputs['synthsr_norm'] = os.path.join(outputs['output_dir'], 'mri', 'synthSR.norm.mgz')
        return outputs

############################################
# make a copy of SynthSR output to rawdata #
############################################

class CopySynthSRInputSpec(BaseInterfaceInputSpec):
    synthsr_raw = File(mandatory=True, desc='raw output of SynthSR')
    synthsr_norm = File(mandatory=True, desc='cleaned up version of synthSR.raw.mgz, scaled such that the white matter has intensity of 110')
    t1w = File(mandatory=True, desc='T1w image')

class CopySynthSROutputSpec(TraitedSpec):
    synthsr_raw = File(desc='raw output of SynthSR')
    synthsr_norm = File(desc='cleaned up version of synthSR.raw.mgz, scaled such that the white matter has intensity of 110')

class CopySynthSR(BaseInterface):
    input_spec = CopySynthSRInputSpec
    output_spec = CopySynthSROutputSpec

    def _run_interface(self, runtime):
        anat_dir = os.path.dirname(self.inputs.t1w)

        synthsr_raw_ent = {
            'space': 'T1w',
            'desc': 'SynthSRraw',
        }

        synthsr_norm_ent = {
            'space': 'T1w',
            'desc': 'SynthSRnorm',
        }

        suffix = "T1w"
        extension = ".nii.gz"

        new_filename_raw = rename_bids_file(self.inputs.t1w, synthsr_raw_ent, suffix, extension)
        new_filename_norm = rename_bids_file(self.inputs.t1w, synthsr_norm_ent, suffix, extension)

        subprocess.run([
            'mri_convert',
            self.inputs.synthsr_raw,
            os.path.join(anat_dir, new_filename_raw)
        ])

        subprocess.run([
            'mri_convert',
            self.inputs.synthsr_norm,
            os.path.join(anat_dir, new_filename_norm)
        ])

        self._synthsr_raw = os.path.join(anat_dir, new_filename_raw)
        self._synthsr_norm = os.path.join(anat_dir, new_filename_norm)

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['synthsr_raw'] = self._synthsr_raw
        outputs['synthsr_norm'] = self._synthsr_norm
        
        return outputs

##################################################
# Post processing (so can run recon-all -qcache) #
##################################################
class PostProcessInputSpec(BaseInterfaceInputSpec):
    fs_output_dir = Directory(mandatory=True, desc='Freesurfer output directory')

class PostProcessOutputSpec(TraitedSpec):
    fs_output_dir = Directory(desc='Freesurfer output directory')

class PostProcess(BaseInterface):
    input_spec = PostProcessInputSpec
    output_spec = PostProcessOutputSpec

    def _run_interface(self, runtime):
        fs_output_dir = self.inputs.fs_output_dir
        
        fs_subjects_dir = os.path.dirname(fs_output_dir)
        fs_subject_id = os.path.basename(fs_output_dir)

        os.environ['SUBJECTS_DIR'] = fs_subjects_dir

        # make sure the 'fsaverage' soft link is created and pointing to the correct directory ($FREESURFER_HOME/subjects/fsaverage)
        subprocess.run([
            'ln', '-sf',
            os.path.join(os.environ.get('FREESURFER_HOME'), 'subjects', 'fsaverage'),
            os.path.join(fs_subjects_dir, 'fsaverage')
        ], check=True)

        if os.path.exists(os.path.join(fs_output_dir, 'scripts', 'IsRunning.lh+rh')):
            # remove the IsRunning.lh+rh file
            subprocess.run([
                'rm', os.path.join(fs_output_dir, 'scripts', 'IsRunning.lh+rh')
            ])

        # create link
        subprocess.run([
            'ln', '-sf',
            os.path.join(fs_output_dir, 'surf', 'lh.white.preaparc.H'),
            os.path.join(fs_output_dir, 'surf', 'lh.white.H')
        ], check=True)

        subprocess.run([
            'ln', '-sf',
            os.path.join(fs_output_dir, 'surf', 'rh.white.preaparc.H'),
            os.path.join(fs_output_dir, 'surf', 'rh.white.H')
        ], check=True)

        subprocess.run([
            'ln', '-sf',
            os.path.join(fs_output_dir, 'surf', 'lh.white.preaparc.K'),
            os.path.join(fs_output_dir, 'surf', 'lh.white.K')
        ], check=True)

        subprocess.run([
            'ln', '-sf',
            os.path.join(fs_output_dir, 'surf', 'rh.white.preaparc.K'),
            os.path.join(fs_output_dir, 'surf', 'rh.white.K')
        ], check=True)

        # copy synthSR.raw.mgz -> rawavg.mgz
        shutil.copyfile(
            os.path.join(fs_output_dir, 'mri', 'synthSR.raw.mgz'),
            os.path.join(fs_output_dir, 'mri', 'rawavg.mgz')
        )

        # copy synthSR.raw.mgz -> ./orig/001.mgz
        os.makedirs(os.path.join(fs_output_dir, 'mri', 'orig'), exist_ok=True)
        shutil.copyfile(
            os.path.join(fs_output_dir, 'mri', 'synthSR.raw.mgz'),
            os.path.join(fs_output_dir, 'mri', 'orig', '001.mgz')
        )

        subprocess.run([
            "mri_convert",
            "--conform",
            os.path.join(fs_output_dir, 'mri', 'rawavg.mgz'),
            os.path.join(fs_output_dir, 'mri', 'orig.mgz')
        ], check=True)

        subprocess.run([
            "mri_add_xform_to_header",
            "-c",
            os.path.join(fs_output_dir, 'mri', 'transforms', 'talairach.xfm'),
            os.path.join(fs_output_dir, 'mri', 'orig.mgz'),
            os.path.join(fs_output_dir, 'mri', 'orig.mgz')
        ], check=True)

        subprocess.run([
            "pctsurfcon",
            "--s", fs_subject_id
        ], check=True)

        # run recon-all -qcache
        subprocess.run([
            "recon-all",
            "-qcache",
            "-s", fs_subject_id
        ], check=True)

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['fs_output_dir'] = self.inputs.fs_output_dir
        
        return outputs