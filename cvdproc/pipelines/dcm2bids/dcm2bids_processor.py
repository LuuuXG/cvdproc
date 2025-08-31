import subprocess
import os
import shutil
import nibabel as nib
import numpy as np
import json
import pandas as pd
import glob
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

        # Experimental: create 'population', 'workflows' folder in derivatives folder
        derivatives_folder = os.path.join(self.BIDS_root_folder, 'derivatives')
        population_folder = os.path.join(derivatives_folder, 'population')
        workflows_folder = os.path.join(derivatives_folder, 'workflows')
        if not os.path.exists(population_folder):
            print('Creating population folder...')
            os.makedirs(population_folder)
        else:
            print('Population folder already exists.')
        
        if not os.path.exists(workflows_folder):
            print('Creating workflows folder...')
            os.makedirs(workflows_folder)
        else:
            print('Workflows folder already exists.')

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
        
        # Fix 'IntendedFor' fields in JSON files
        # self.fix_intendedfor_for_subject_session(subject_id, session_id)
    
    def fix_intendedfor_for_subject_session(self, subject_id, session_id):
        """
        Fix 'IntendedFor' fields only under a specific subject/session.
        """
        target_dir = os.path.join(self.BIDS_root_folder, f"sub-{subject_id}", f"ses-{session_id}")
        if not os.path.exists(target_dir):
            print(f"[WARN] Target directory not found: {target_dir}")
            return

        fixed_count = 0
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        if "IntendedFor" in data:
                            original = data["IntendedFor"]
                            modified = None

                            if isinstance(original, str):
                                if not original.startswith("ses-") and "ses-" in original:
                                    modified = "ses-" + original.split("ses-", 1)[1]
                            elif isinstance(original, list):
                                changed = False
                                new_list = []
                                for item in original:
                                    if not item.startswith("ses-") and "ses-" in item:
                                        new_list.append("ses-" + item.split("ses-", 1)[1])
                                        changed = True
                                    else:
                                        new_list.append(item)
                                if changed:
                                    modified = new_list

                            if modified is not None:
                                data["IntendedFor"] = modified
                                with open(json_path, "w", encoding="utf-8") as f:
                                    json.dump(data, f, indent=4)
                                fixed_count += 1
                                print(f"[FIXED] IntendedFor: {json_path}")

                    except Exception as e:
                        print(f"[ERROR] Failed to process {json_path}: {e}")
                        continue

        if fixed_count > 0:
            print(f"Fixed {fixed_count} JSON file(s) for sub-{subject_id} ses-{session_id}")
    
    def fix_dwi_bvec_bval(self, subject_id, session_id, fix_config):
        dwi_dir = os.path.join(
            self.BIDS_root_folder,
            f"sub-{subject_id}",
            f"ses-{session_id}",
            "dwi"
        )

        if not os.path.exists(dwi_dir):
            print(f"[WARN] No DWI directory found for sub-{subject_id}, ses-{session_id}. Skipping fix.")
            return

        for file in os.listdir(dwi_dir):
            for fix_item in fix_config:
                match_str = fix_item.get("match", "")
                if match_str in file and file.endswith(".nii.gz"):
                    base = file.replace(".nii.gz", "")
                    bvec_path = os.path.join(dwi_dir, base + ".bvec")
                    bval_path = os.path.join(dwi_dir, base + ".bval")

                    if os.path.exists(bvec_path) and os.path.exists(fix_item["bvec"]):
                        print(f"Replacing {bvec_path} with {fix_item['bvec']}")
                        shutil.copyfile(fix_item["bvec"], bvec_path)

                    if os.path.exists(bval_path) and os.path.exists(fix_item["bval"]):
                        print(f"Replacing {bval_path} with {fix_item['bval']}")
                        shutil.copyfile(fix_item["bval"], bval_path)
    
    def check_data(self, check_data_config):
        """Check BIDS rawdata and/or derivatives files based on filename or json criteria, and save results to check_data.xlsx"""

        print("Generating the BIDS Layout for rawdata...")
        layout = BIDSLayout(self.BIDS_root_folder, validate=False)
        results = []
        column_titles = ["Subject_id", "Session_id"]

        for check in check_data_config:
            custom_name = check["custom_name"]
            column_titles.append(custom_name)
            column_titles.append(f"{custom_name}_number")

        for subject in layout.get_subjects():
            sessions = layout.get_sessions(subject=subject)
            for session in sessions:
                row = [subject, session] + [0] * (len(check_data_config) * 2)

                for check_idx, check in enumerate(check_data_config):
                    custom_name = check["custom_name"]
                    criteria = check["criteria"]
                    method = check["method"]
                    derivatives_name = check.get("derivatives_name", None)
                    suffix = check.get("suffix", None)

                    match_found = False
                    match_count = 0

                    # ---------- rawdata 逻辑 ----------
                    if derivatives_name is None:
                        if suffix in ['T1w', 'T2w', 'dwi', 'bold', 'FLAIR', 'asl', 'epi']:
                            try:
                                files = layout.get(subject=subject, session=session, suffix=suffix, extension=["nii.gz"])
                                filepaths = [f.path for f in files]
                            except:
                                filepaths = []
                        else:
                            session_dir = os.path.join(self.BIDS_root_folder, f"sub-{subject}", f"ses-{session}")
                            filepaths = []
                            if os.path.exists(session_dir):
                                for root, _, files in os.walk(session_dir):
                                    for f in files:
                                        if f.endswith(".nii.gz") and suffix in f:
                                            filepaths.append(os.path.join(root, f))

                    # ---------- derivatives 逻辑 ----------
                    else:
                        derivatives_dir = os.path.join(self.BIDS_root_folder, "derivatives", derivatives_name,
                                                    f"sub-{subject}", f"ses-{session}")
                        filepaths = []
                        if os.path.exists(derivatives_dir):
                            pattern = os.path.join(derivatives_dir, criteria)
                            filepaths = glob.glob(pattern, recursive=True)

                    # ---------- 匹配逻辑 ----------
                    for file in filepaths:
                        if method == "filename":
                            match_found = True
                            match_count += 1

                        elif method == "json" and derivatives_name is None:
                            json_file_path = file.replace(".nii.gz", ".json")
                            if os.path.exists(json_file_path):
                                with open(json_file_path, "r") as json_file:
                                    json_data = json.load(json_file)
                                    for key, value in criteria.items():
                                        if key in json_data and json_data[key] == value:
                                            match_found = True
                                            match_count += 1
                                        else:
                                            match_found = False
                                            break
                            if match_found:
                                break  # stop after first match

                    # 更新该项结果
                    custom_name_idx = 2 + check_idx * 2
                    custom_name_number_idx = custom_name_idx + 1
                    row[custom_name_idx] = 1 if match_found else 0
                    row[custom_name_number_idx] = match_count

                results.append(row)

        # 保存为 Excel
        output_dir = os.path.join(self.BIDS_root_folder, "derivatives", "population")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "check_data.xlsx")

        df = pd.DataFrame(results, columns=column_titles)
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Check Data")
            sheet = writer.book["Check Data"]
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

    config_path = '/mnt/f/BIDS/UKB_AFproject/code/config.yml'
    config = load_config(config_path)
    check_data_config = config.get("check_data", [])

    layout = BIDSLayout(config["bids_dir"], validate=False)

    subjects = layout.get_subjects()
    session = layout.get_sessions(subject="SVD0050")
    files = layout.get(subject="SVD0077", extension=["nii.gz"], session='02')
    files = layout.get(subject="SVD0050", session='02', suffix='T1w')