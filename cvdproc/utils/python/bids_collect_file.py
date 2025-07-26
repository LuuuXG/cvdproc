import os
import shutil

def collect_files_by_pattern(bids_deriv_root, subfolder_path, target_filename, have_id, output_dir):
    """
    Traverse BIDS derivatives to collect specified files across all subjects and sessions.

    Parameters:
    - bids_deriv_root (str): Path to the BIDS derivatives root directory.
    - subfolder_path (str): Relative subfolder under each subject/session (e.g., 'fs_processing').
    - target_filename (str): Filename to search for in each subject/session (e.g., 'seed_mask_in_fs.nii.gz').
    - output_dir (str): Directory where matched files will be copied.
    """
    os.makedirs(output_dir, exist_ok=True)

    for subject_folder in os.listdir(bids_deriv_root):
        if not subject_folder.startswith("sub-"):
            continue
        subject_id = subject_folder.split('-')[1]
        subject_path = os.path.join(bids_deriv_root, subject_folder)

        if not os.path.isdir(subject_path):
            continue

        for session_folder in os.listdir(subject_path):
            if not session_folder.startswith("ses-"):
                continue
            session_id = session_folder.split('-')[1]
            session_path = os.path.join(subject_path, session_folder)

            if have_id is False:
                full_file_path = os.path.join(session_path, subfolder_path.strip('/'), target_filename)
            elif have_id is True:
                full_file_path = os.path.join(session_path, subfolder_path.strip('/'), f"sub-{subject_id}_ses-{session_id}_{target_filename}")

            if os.path.exists(full_file_path):
                new_filename = f"sub-{subject_id}_ses-{session_id}_{target_filename}"
                output_path = os.path.join(output_dir, new_filename)
                shutil.copy2(full_file_path, output_path)
                print(f"Copied: {full_file_path} -> {output_path}")
            else:
                print(f"Missing: {full_file_path}")

if __name__ == '__main__':
    collect_files_by_pattern(
        bids_deriv_root="/mnt/f/BIDS/SVD_BIDS/derivatives/nemo",
        subfolder_path="",
        target_filename="seed_mask_in_mni_nemo_output_ifod2act_chacovol_res1mm_smooth6mm_mean.nii.gz",
        have_id=True,
        output_dir="/mnt/f/BIDS/SVD_BIDS/derivatives/nemo_mean_mni"
    )