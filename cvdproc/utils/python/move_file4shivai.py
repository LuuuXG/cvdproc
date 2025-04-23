import os
import subprocess
import shutil

sepia_output_path = '/mnt/f/BIDS/YC+YT_QSM_BIDS/derivatives/SEPIA'
bids_root_path = '/mnt/f/BIDS/YC+YT_QSM_BIDS'

shivai_input_path = '/mnt/f/CSVD_revision/shiva_input'

# 获取第一层文件夹
first_level_dirs = [d for d in os.listdir(sepia_output_path) if os.path.isdir(os.path.join(sepia_output_path, d))]

# 遍历所有第一层文件夹，筛选出以'sub'开头的文件夹
for subject_dir in first_level_dirs:
    if subject_dir.startswith('sub-'):  # 确保是以'sub-'开头的文件夹
        subject_id = subject_dir
        print(f'Processing subject: {subject_id}')

        # 构建对应的swi文件路径
        swi_path = os.path.join(sepia_output_path, subject_dir, 'Sepia_clearswi.nii.gz')

        # 寻找对应的t1w文件路径
        t1w_path = os.path.join(bids_root_path, subject_dir, 'anat', f'{subject_id}_T1w.nii.gz')

        # 在shivai_input_path中创建对应的subject文件夹
        shivai_subject_path = os.path.join(shivai_input_path, subject_id)
        if not os.path.exists(shivai_subject_path):
            os.makedirs(shivai_subject_path)

        # 在其中分别创建t1和swi文件夹，复制对应的文件（保留源文件），重命名为swi.nii.gz和t1.nii.gz
        shivai_t1_path = os.path.join(shivai_subject_path, 't1')
        shivai_swi_path = os.path.join(shivai_subject_path, 'swi')
        if not os.path.exists(shivai_t1_path):
            os.makedirs(shivai_t1_path)
        if not os.path.exists(shivai_swi_path):
            os.makedirs(shivai_swi_path)

        shutil.copy2(t1w_path, os.path.join(shivai_t1_path, 't1.nii.gz'))
        shutil.copy2(swi_path, os.path.join(shivai_swi_path, 'swi.nii.gz'))

        swi_file_path = os.path.join(shivai_swi_path, 'swi.nii.gz')
        t1_file_path = os.path.join(shivai_t1_path, 't1.nii.gz')

        # 预处理swi和t1文件
        subprocess.run(['mri_synthstrip', '-i', swi_file_path, '-o', os.path.join(shivai_swi_path, 'swi_brain.nii.gz')])
        subprocess.run(['mri_synthstrip', '-i', t1_file_path, '-o', os.path.join(shivai_t1_path, 't1_brain.nii.gz')])
        subprocess.run(['flirt', '-ref', os.path.join(shivai_t1_path, 't1_brain.nii.gz'), '-in',
                        os.path.join(shivai_swi_path, 'swi_brain.nii.gz'), '-out',
                        os.path.join(shivai_swi_path, 'swi_brain_t1.nii.gz')])

        # 删除原始文件，只保留预处理后的文件
        os.remove(os.path.join(shivai_swi_path, 'swi.nii.gz'))
        os.remove(os.path.join(shivai_t1_path, 't1.nii.gz'))
        os.remove(os.path.join(shivai_swi_path, 'swi_brain.nii.gz'))
