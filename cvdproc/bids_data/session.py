import os
import glob
import re

class BIDSSession:
    def __init__(self, bids_dir, subject_id, session_id):
        """
        Initialize a BIDSSession instance.
        :param bids_dir: str, BIDS root directory
        :param subject_id: str, subject ID (e.g. "01")
        :param session_id: str, session ID (e.g. "01")
        """
        self.subject_id = subject_id
        self.session_id = session_id
        self.bids_dir = os.path.abspath(bids_dir)
        self.session_dir = os.path.join(bids_dir, f"sub-{subject_id}", f"ses-{session_id}")

        if not os.path.exists(self.session_dir):
            raise FileNotFoundError(f"Session directory {self.session_dir} does not exist.")

        # Modalities: sub-<subject_id>/ses-<session_id>/<modality>
        #### BIDS standard ####
        self.anat_files = self._find_files('anat')
        self.dwi_files = self._find_files('dwi')
        self.func_files = self._find_files('func')
        self.fmap_files = self._find_files('fmap')
        self.perf_files = self._find_files('perf')
        #### Non-standard ####
        self.swi_files = self._find_files('swi')
        self.qsm_files = self._find_files('qsm')
        self.pwi_files = self._find_files('pwi')

        # Derivatives: derivatives/<output_name>/sub-<subject_id>/ses-<session_id>
        self.freesurfer_dir = self._find_output('freesurfer')
        self.freesurfer_clinical_dir = self._find_output('freesurfer_clinical')
        self.lesion_mask_dir = self._find_output('lesion_mask')
        self.fsl_anat_dir = self._find_output('fsl_anat')
        self.xfm_dir = self._find_output('xfm')

    def _find_files(self, modality):
        """
        Locate all NIfTI files within the specified modality directory.
        :param modality: str, modality name (e.g. "anat", "dwi", "func")
        :return: list, matching file paths
        """
        modality_dir = os.path.join(self.session_dir, modality)
        if not os.path.exists(modality_dir):
            return []
        if modality == 'dwi':
            return sorted(glob.glob(os.path.join(modality_dir, '*.nii*')) + glob.glob(os.path.join(modality_dir, '*.bval')) + glob.glob(os.path.join(modality_dir, '*.bvec')) + glob.glob(os.path.join(modality_dir, '*.json')))
        elif modality == 'fmap':
            # return all files
            return sorted(glob.glob(os.path.join(modality_dir, '*')))
        elif modality == 'perf':
            # return nifti and tsv files
            return sorted(glob.glob(os.path.join(modality_dir, '*.nii*')) + glob.glob(os.path.join(modality_dir, '*.tsv')))
        else:
            return sorted(glob.glob(os.path.join(modality_dir, '*.nii*')))
    
    def _find_output(self, output_name):
        """
        Locate a derivative output directory.
        :param output_name: str, output directory name (e.g. "freesurfer")
        :return: str, output directory path; returns None if it does not exist
        """
        if self.session_id is not None:
            output_dir = os.path.join(self.bids_dir, 'derivatives', output_name, f"sub-{self.subject_id}", f"ses-{self.session_id}")
        else:
            output_dir = os.path.join(self.bids_dir, 'derivatives', output_name, f"sub-{self.subject_id}")
        return output_dir if os.path.exists(output_dir) else None

    def list_all_files(self):
        """
        List all modality files available in this session.
        :return: dict, modalities and their file paths
        """
        return {
            'anat': self.anat_files,
            'dwi': self.dwi_files,
            'func': self.func_files,
            'fmap': self.fmap_files,
            'swi': self.swi_files,
            'qsm': self.qsm_files
        }

    def get_flair_files(self):
        """
        Return all FLAIR file paths under anat.
        :return: list, FLAIR file paths; returns an empty list if none exist
        """
        return [file for file in self.anat_files if 'FLAIR.nii' in file]

    def get_t1w_files(self):
        """
        Return all T1w file paths under anat.
        :return: list, T1w file paths; returns an empty list if none exist
        """
        return [file for file in self.anat_files if 'T1w.nii' in file]
    
    def get_dwi_files(self):
        """
        Return DWI file paths.
        :return: list, file paths including dwi, bval, bvec, and json files
        """
        dwi_list = []
        for file in self.dwi_files:
            if file.endswith('.nii.gz'):
                dwi_list.append({'type': 'dwi', 'path': file})
            elif file.endswith('.bval'):
                dwi_list.append({'type': 'bval', 'path': file})
            elif file.endswith('.bvec'):
                dwi_list.append({'type': 'bvec', 'path': file})
            elif file.endswith('.json'):
                dwi_list.append({'type': 'json', 'path': file})
        return dwi_list

    def get_fmap_files(self):
        """
        Return fieldmap file paths.
        :return: list, file paths including reverse b0 files
        """
        fmap_list = []
        for file in self.fmap_files:
            if 'dir-PA' in file:
                fmap_list.append({'type': 'reverse_b0', 'path': file})
        return fmap_list
    
    def get_swi_files(self):
        """
        Return SWI file paths.
        :return: list, SWI file paths
        """
        return self.swi_files
    
    def get_qsm_files(self):
        """
        Return QSM file paths.
        :return: list, QSM file paths
        """
        return self.qsm_files
    
    def get_pwi_files(self):
        """
        Return PWI file paths.
        :return: list, PWI file paths
        """
        return self.pwi_files

    def get_perf_files(self):
        """
        Return Perfusion file paths.
        :return: list, Perfusion file paths
        """
        return self.perf_files