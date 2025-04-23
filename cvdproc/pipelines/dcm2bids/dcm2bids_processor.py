import subprocess
import os
import nibabel as nib
import numpy as np
import json
import pandas as pd
from bids.cli import layout
from openpyxl import Workbook
from openpyxl.styles import Font
from bids.layout import BIDSLayout
from requests import session


class Dcm2BidsProcessor:
    def __init__(self, BIDS_root_folder):
        self.BIDS_root_folder = BIDS_root_folder

    def initialize(self):
        # check if the output directory exists
        if not os.path.exists(self.BIDS_root_folder):
            print('Creating output directory...')
            os.makedirs(self.BIDS_root_folder)
        else:
            print('Output directory already exists.')

        subprocess.run(['dcm2bids_scaffold', '-o', self.BIDS_root_folder])

        # Experimental: create a 'population' folder in derivatives folder
        derivatives_folder = os.path.join(self.BIDS_root_folder, 'derivatives')
        population_folder = os.path.join(derivatives_folder, 'population')
        if not os.path.exists(population_folder):
            print('Creating population folder...')
            os.makedirs(population_folder)
        else:
            print('Population folder already exists.')

        print('Initialization completed.')

    def convert(self, config_file, dicom_directory, subject_id, session_id):
        subprocess.run([
            'dcm2bids',
            '-d', dicom_directory,
            '-p', subject_id,
            '-s', session_id,
            '-c', config_file,
            '-o', self.BIDS_root_folder,
            #'--force_dcm2bids',
            '--auto_extract_entities'
        ])

        ## use croped 3d image if exists
        # check if the 'tmp_dcm2bids' folder exists
        tmp_dcm2bids_folder = os.path.join(self.BIDS_root_folder, 'tmp_dcm2bids')
        if os.path.exists(tmp_dcm2bids_folder):
            print('Checking for cropped 3D images...')
            cropped_image_count = 0
            
            # find temp folder for the subject and session
            if session_id:
                subject_temp_folder = os.path.join(tmp_dcm2bids_folder, f'sub-{subject_id}_ses-{session_id}')
                subject_bids_folder = os.path.join(self.BIDS_root_folder, f'sub-{subject_id}', f'ses-{session_id}')
            else:
                subject_temp_folder = os.path.join(tmp_dcm2bids_folder, f'sub-{subject_id}')
                subject_bids_folder = os.path.join(self.BIDS_root_folder, f'sub-{subject_id}')
            
            # search for the cropped 3D image (has 'Crop' in the filename)
            # subject_temp_folder has no subfolders, so we can use os.listdir
            for file in os.listdir(subject_temp_folder):
                if 'Crop' in file and '.nii' in file:
                    # get the paired original nifti file
                    # for example, we found '2021-11-4_T1-1_20211104131746_Crop_1.nii.gz'
                    # we need to find '2021-11-4_T1-1_20211104131746.nii.gz'
                    original_file = file.replace('_Crop_1', '')
                    original_file_path = os.path.join(subject_temp_folder, original_file)

                    # if the original file not exits, it means it has been moved to BIDS folder by the dcm2bids
                    # so, we need to move the cropped file to the BIDS folder and replace the original file
                    if not os.path.exists(original_file_path):
                        print('Found cropped 3D image, moving to BIDS folder...')
                        # we need to find the original file in the BIDS folder
                        # a possible strategy is to find nifti files with the same transform matrix (because we only
                        # cropped the z axis). subject_bids_folder has subfolders, so we can use os.walk
                        # first we can check the x and y dimensions of the cropped image
                        cropped_image = os.path.join(subject_temp_folder, file)
                        cropped_nii = nib.load(cropped_image)
                        cropped_shape = cropped_nii.shape
                        #cropped_affine = cropped_nii.affine
                        cropped_voxel_size = cropped_nii.header.get_zooms()
                        
                        files_matched = 0
                        for root, dirs, files in os.walk(subject_bids_folder):
                            for file in files:
                                if '.nii' in file:
                                    nii_file = os.path.join(root, file)
                                    nii = nib.load(nii_file)
                                    if nii.shape[0] == cropped_shape[0] and nii.shape[1] == cropped_shape[1] and nii.header.get_zooms() == cropped_voxel_size:
                                        bids_converted_file_path = nii_file
                                        files_matched += 1
                                        cropped_image_count += 1

                        if files_matched == 1:
                            print(f'{cropped_image} -> {bids_converted_file_path}')
                            os.rename(cropped_image, bids_converted_file_path)
                        elif files_matched > 1:
                            print('More than one file matched, please check the files manually.')
                        else:
                            print('No file matched, please check the files manually.')
            
            if cropped_image_count == 0:
                print('No cropped 3D image found.')
                    
        else:
            print('No temporary dcm2bids folder found, or it is not in the BIDS root folder.')
    
    def check_data(self, check_data_config):
        """检查BIDS文件夹下的影像数据是否满足多个序列配置，并保存结果为check_data.xlsx"""
        print("Generating the BIDS Layout...")
        layout = BIDSLayout(self.BIDS_root_folder)
        results = []
        column_titles = ["Subject_id", "Session_id"]

        # 为每个序列配置添加列标题，同时增加 "_number" 列
        for check in check_data_config:
            custom_name = check["custom_name"]
            column_titles.append(custom_name)  # 原有列
            column_titles.append(f"{custom_name}_number")  # 新增数量列

        # 遍历每个配置，检查BIDS数据中的文件
        for check in check_data_config:
            custom_name = check["custom_name"]
            suffix = check["suffix"]
            criteria = check["criteria"]
            method = check["method"]

            print(f"Checking for {custom_name} in suffix {suffix} with criteria {criteria}...")

            # 查找所有被试
            for subject in layout.get_subjects():
                sessions = layout.get_sessions(subject=subject)
                for session in sessions:
                    match_found = False
                    match_count = 0  # 用于记录符合条件的文件数量

                    # 使用BIDSLayout获取符合标准的文件，或使用os.listdir()获取非标准文件
                    if suffix in ['T1w', 'T2w', 'dwi', 'bold', 'FLAIR', 'asl', 'epi']: 
                        files = layout.get(subject=subject, session=session, suffix=suffix, extension=["nii.gz"])
                    else:
                        session_dir = os.path.join(self.BIDS_root_folder, f"sub-{subject}", f"ses-{session}")
                        modality_folders = [f for f in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, f))]

                        files = []
                        for folder in modality_folders:
                            modality_folder_path = os.path.join(session_dir, folder)
                            # 对于每个文件夹，我们遍历其中的所有文件，检查是否符合后缀
                            for file in os.listdir(modality_folder_path):
                                if file.endswith(".nii.gz") and (suffix in file):
                                    files.append(os.path.join(modality_folder_path, file))

                    # 遍历文件，匹配指定的criteria
                    for file in files:
                        if method == "json":
                            # 使用json文件中的字段匹配
                            json_file_path = file.replace(".nii.gz", ".json")
                            if os.path.exists(json_file_path):
                                with open(json_file_path, "r") as json_file:
                                    json_data = json.load(json_file)

                                    # 如果 criteria 是字典，表示可能有多个字段需要匹配
                                    for key, value in criteria.items():
                                        if key in json_data:
                                            if json_data[key] == value:
                                                match_found = True
                                                match_count += 1  # 增加符合条件的文件计数
                                        else:
                                            print(f"Warning: Key '{key}' not found in JSON for {file}.")
                                            match_found = False
                                            break
                                    
                                    if match_found:
                                        break
                        elif method == "filename":
                            # 使用文件名匹配
                            if criteria in os.path.basename(file):
                                match_found = True
                                match_count += 1  # 增加符合条件的文件计数

                    # 将结果添加到列表
                    if not any(d[0] == subject and d[1] == session for d in results):
                        # 如果是新的一行（新被试/新会话），添加
                        results.append([subject, session] + [0] * (len(check_data_config) * 2))  # 为每个序列添加两个列
                        row_idx = len(results) - 1
                    else:
                        # 如果已经有这行（同一个被试/会话），找到该行
                        row_idx = next(idx for idx, row in enumerate(results) if row[0] == subject and row[1] == session)

                    # 更新对应的列值
                    custom_name_idx = column_titles.index(custom_name)
                    custom_name_number_idx = column_titles.index(f"{custom_name}_number")

                    results[row_idx][custom_name_idx] = 1 if match_found else 0  # 更新是否符合的列
                    results[row_idx][custom_name_number_idx] = match_count  # 更新文件数量列

        # 保存结果为Excel文件
        output_dir = os.path.join(self.BIDS_root_folder, "derivatives", "population")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "check_data.xlsx")
        
        df = pd.DataFrame(results, columns=column_titles)
        
        # 创建Excel文件并设置字体为Calibri
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Check Data")
            workbook = writer.book
            sheet = workbook["Check Data"]
            
            # 设置所有单元格的字体为 Calibri
            calibri_font = Font(name="Calibri")
            for row in sheet.iter_rows():
                for cell in row:
                    cell.font = calibri_font

        print(f"Results saved to {output_file}")
    

if __name__ == 'main':
    import yaml

    def load_config(config_file):
        """加载 YAML 配置文件"""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    config_path = '/mnt/f/BIDS/SVD_BIDS/code/config.yml'
    config = load_config(config_path)
    check_data_config = config.get("check_data", [])

    layout = BIDSLayout(config["bids_dir"])
    subjects = layout.get_subjects()
    session = layout.get_sessions(subject="SVD0050")
    files = layout.get(subject="SVD0077", extension=["nii.gz"], session='02')
    files = layout.get(subject="SVD0050", session='02', suffix='T1w')