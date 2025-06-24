# import os
# import subprocess
# import shutil

# bids_root_path = '/mnt/f/UKBdata/data_NC'

# shivai_input_path = '/mnt/f/UKBdata/NC_SHIVA_INPUT'

# # 遍历所有第一层文件夹，筛选出以'sub'开头的文件夹
# for subject_dir in os.listdir(bids_root_path):
#     #print(f'Processing directory: {subject_dir}')
#     if subject_dir.startswith('sub-'):  # 确保是以'sub-'开头的文件夹
#         subject_id = subject_dir
#         print(f'Processing subject: {subject_id}')

#         # 构建对应的swi文件路径
#         swi_path = os.path.join(bids_root_path, subject_dir, 'SWI.nii.gz')

#         # 寻找对应的t1w文件路径
#         t1w_path = os.path.join(bids_root_path, subject_dir, 'T1.nii.gz')

#         # 在shivai_input_path中创建对应的subject文件夹
#         if os.path.exists(t1w_path) or os.path.exists(swi_path):
#             shivai_subject_path = os.path.join(shivai_input_path, subject_id)
#             if not os.path.exists(shivai_subject_path):
#                 os.makedirs(shivai_subject_path)

#         # 在其中分别创建t1和swi文件夹，复制对应的文件（保留源文件），重命名为swi.nii.gz和t1.nii.gz

#         if os.path.exists(t1w_path):
#             shivai_t1_path = os.path.join(shivai_subject_path, 't1')

#             if not os.path.exists(shivai_t1_path):
#                 os.makedirs(shivai_t1_path)

#                 shutil.copy2(t1w_path, os.path.join(shivai_t1_path, 't1.nii.gz'))
        
#         if os.path.exists(swi_path):
#             shivai_swi_path = os.path.join(shivai_subject_path, 'swi')

#             if not os.path.exists(shivai_swi_path):
#                 os.makedirs(shivai_swi_path)

#                 shutil.copy2(swi_path, os.path.join(shivai_swi_path, 'swi.nii.gz'))
import os
import shutil

bids_root_path = 'f:/UKBdata/data_NC'
shivai_input_path = 'f:/UKBdata/NC_TRUENET_INPUT'

# 遍历所有以'sub-'开头的被试文件夹
for subject_dir in os.listdir(bids_root_path):
    if subject_dir.startswith('sub-'):
        subject_id = subject_dir
        print(f'Processing subject: {subject_id}')

        # 构建SWI路径
        swi_path = os.path.join(bids_root_path, subject_dir, 'T2_FLAIR.nii.gz')

        # 如果SWI文件存在，创建目标文件夹并复制
        if os.path.exists(swi_path):
            shivai_subject_path = os.path.join(shivai_input_path, subject_id)

            if os.path.exists(os.path.join(shivai_subject_path, 'T2_FLAIR.nii.gz')):
                print(f'File already exists for {subject_id}, skipping copy.')
                continue

            os.makedirs(shivai_subject_path, exist_ok=True)
            shutil.copy2(swi_path, os.path.join(shivai_subject_path, 'T2_FLAIR.nii.gz'))
