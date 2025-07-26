# ExploreASL nipype interfaces

import os
import shutil
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File)
from traits.api import Directory, Str, List, Bool, Enum, Int, Float, TraitError

class CreateRawdataFolderInputSpec(BaseInterfaceInputSpec):
    bids_root_dir = Directory(exists=True, mandatory=True, desc="BIDS root directory")
    subject_id = Str(mandatory=True, desc="Subject ID for the raw data")
    session_id = Str(mandatory=True, desc="Session ID for the raw data")

class CreateRawdataFolderOutputSpec(TraitedSpec):
    bids_root_dir = Directory(exists=True, desc="BIDS root directory after creating rawdata folder")

class CreateRawdataFolder(BaseInterface):
    input_spec = CreateRawdataFolderInputSpec
    output_spec = CreateRawdataFolderOutputSpec

    def _run_interface(self, runtime):
        bids_root = self.inputs.bids_root_dir
        subject_id = self.inputs.subject_id
        session_id = self.inputs.session_id

        anat_dir = os.path.join(bids_root, 'sub-' + subject_id, 'ses-' + session_id, 'anat')
        perf_dir = os.path.join(bids_root, 'sub-' + subject_id, 'ses-' + session_id, 'perf')

        rawdata_dir = os.path.join(bids_root, 'rawdata')
        os.makedirs(rawdata_dir, exist_ok=True)

        # copy the anatomical and perfusion directories to the rawdata directory
        target_anat_dir = os.path.join(rawdata_dir, 'sub-' + subject_id, 'ses-' + session_id, 'anat')
        target_perf_dir = os.path.join(rawdata_dir, 'sub-' + subject_id, 'ses-' + session_id, 'perf')

        os.makedirs(target_anat_dir, exist_ok=True)
        os.makedirs(target_perf_dir, exist_ok=True)

        if os.path.exists(anat_dir):
            shutil.copytree(anat_dir, target_anat_dir, dirs_exist_ok=True)
        if os.path.exists(perf_dir):
            shutil.copytree(perf_dir, target_perf_dir, dirs_exist_ok=True)

        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['bids_root_dir'] = self.inputs.bids_root_dir
        return outputs