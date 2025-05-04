import os
import glob
import re

class BIDSSession:
    def __init__(self, bids_dir, subject_id, session_id):
        """
        初始化一个 BIDSSession 实例
        :param bids_dir: str, BIDS 根目录
        :param subject_id: str, 受试者 ID (如 '01')
        :param session_id: str, 会话 ID (如 '01')
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
        #### Non-standard ####
        self.swi_files = self._find_files('swi')
        self.qsm_files = self._find_files('qsm')

        # Derivatives: derivatives/<output_name>/sub-<subject_id>/ses-<session_id>/<pipeline>
        self.freesurfer_dir = self._find_output('freesurfer')
        self.freesurfer_clinical_dir = self._find_output('freesurfer_clinical')
        self.lesion_mask_dir = self._find_output('lesion_mask')
        self.fsl_anat_dir = self._find_output('fsl_anat')
        self.xfm_dir = self._find_output('xfm')

    def _find_files(self, modality):
        """
        查找指定模态目录下的所有 NIfTI 文件
        :param modality: str, 模态名称 (如 'anat', 'dwi', 'func')
        :return: list, 匹配的文件路径列表
        """
        modality_dir = os.path.join(self.session_dir, modality)
        if not os.path.exists(modality_dir):
            return []
        if modality == 'dwi':
            return sorted(glob.glob(os.path.join(modality_dir, '*.nii*')) + glob.glob(os.path.join(modality_dir, '*.bval')) + glob.glob(os.path.join(modality_dir, '*.bvec')) + glob.glob(os.path.join(modality_dir, '*.json')))
        else:
            return sorted(glob.glob(os.path.join(modality_dir, '*.nii*')))
    
    def _find_output(self, output_name):
        """
        查找指定输出目录
        :param output_name: str, 输出目录名称 (如 'freesurfer')
        :return: str, 输出目录路径；如果不存在，返回 None
        """
        if self.session_id is not None:
            output_dir = os.path.join(self.bids_dir, 'derivatives', output_name, f"sub-{self.subject_id}", f"ses-{self.session_id}")
        else:
            output_dir = os.path.join(self.bids_dir, 'derivatives', output_name, f"sub-{self.subject_id}")
        return output_dir if os.path.exists(output_dir) else None

    def list_all_files(self):
        """
        列出该 session 下所有模态文件
        :return: dict, 各模态及其文件路径
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
        返回 anat 下的所有 FLAIR 文件路径
        :return: list, FLAIR 文件路径列表；如果不存在，返回空列表
        """
        return [file for file in self.anat_files if 'FLAIR.nii' in file]

    def get_t1w_files(self):
        """
        返回 anat 下的所有 T1w 文件路径
        :return: list, T1w 文件路径列表；如果不存在，返回空列表
        """
        return [file for file in self.anat_files if 'T1w.nii' in file]
    
    def get_dwi_files(self):
        """
        返回 DWI 文件路径
        :return: list, 包括dwi, bval, bvec, json 的文件路径列表
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
        返回 fieldmap 文件路径
        :return: list, 包括反向 b0 的文件路径
        """
        fmap_list = []
        for file in self.fmap_files:
            if 'dir-PA' in file:
                fmap_list.append({'type': 'reverse_b0', 'path': file})
        return fmap_list
    
    def get_swi_files(self):
        """
        返回 SWI 文件路径
        :return: list, SWI 文件路径列表
        """
        return self.swi_files
    
    def get_qsm_files(self):
        """
        返回 QSM 文件路径
        :return: list, QSM 文件路径列表
        """
        return self.qsm_files
