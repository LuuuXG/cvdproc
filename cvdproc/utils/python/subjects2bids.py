import os
import shutil


def convert_to_bids(input_dir, output_dir, subject_id, session_id, subject_prefix="AF"):
    # 添加前缀
    new_subject_id = f"{subject_prefix}{subject_id}"

    # Create the BIDS directory structure
    bids_subject_dir = os.path.join(output_dir, f"sub-{new_subject_id}", f"ses-{session_id}")
    os.makedirs(bids_subject_dir, exist_ok=True)

    # Create modality folders
    anat_dir = os.path.join(bids_subject_dir, "anat")
    swi_dir = os.path.join(bids_subject_dir, "swi")
    os.makedirs(anat_dir, exist_ok=True)
    os.makedirs(swi_dir, exist_ok=True)

    # Define the files to rename and copy
    file_map = {
        "T1.nii.gz": f"sub-{new_subject_id}_ses-{session_id}_T1w.nii.gz",
        "T2_FLAIR.nii.gz": f"sub-{new_subject_id}_ses-{session_id}_FLAIR.nii.gz",
        "SWI.nii.gz": f"sub-{new_subject_id}_ses-{session_id}_swi.nii.gz",
    }

    # Loop through the files and rename/copy them to the new BIDS structure
    for original_name, new_name in file_map.items():
        original_file = os.path.join(input_dir, original_name)
        if os.path.exists(original_file):
            if "T1" in original_name or "FLAIR" in original_name:
                shutil.copy2(original_file, os.path.join(anat_dir, new_name))
            elif "SWI" in original_name:
                shutil.copy2(original_file, os.path.join(swi_dir, new_name))


def process_all_subjects(root_input_dir, output_dir, session_id="test", subject_prefix="AF"):
    """ 遍历 root_input_dir 下所有 'sub-' 开头的文件夹，并进行 BIDS 格式转换 """
    for subject_folder in os.listdir(root_input_dir):
        subject_path = os.path.join(root_input_dir, subject_folder)
        if os.path.isdir(subject_path) and subject_folder.startswith("sub-"):
            subject_id = subject_folder.split("-")[1]  # 提取编号部分
            print(f"Processing subject: {subject_id}")
            convert_to_bids(subject_path, output_dir, subject_id, session_id, subject_prefix)


if __name__ == "__main__":
    root_input_dir = "/mnt/f/UKBdata/data_NC"  # 需要遍历的根目录
    output_dir = "/mnt/f/BIDS/UKB_AFproject"  # BIDS 目标文件夹
    session_id = "01"  # 统一 session ID
    subject_prefix = "NC"  # 前缀，例如 "AF"

    process_all_subjects(root_input_dir, output_dir, session_id, subject_prefix)
