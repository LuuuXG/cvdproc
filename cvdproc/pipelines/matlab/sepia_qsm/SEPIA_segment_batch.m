% BIDS �ļ���·��
BIDS_folder = '/mnt/e/Neuroimage/Projects/QSM_process/YT_QSM_BIDS';

% ���������ļ���·��
derivatives_folder = fullfile(BIDS_folder, 'derivatives');

% ��������� derivatives �ļ��У��򴴽�
if ~exist(derivatives_folder, 'dir')
    mkdir(derivatives_folder);
end

% SEPIA ����ļ���·��
SEPIA_output_folder = fullfile(derivatives_folder, 'SEPIA');

% ��������� SEPIA ����ļ��У��򴴽�
if ~exist(SEPIA_output_folder, 'dir')
    mkdir(SEPIA_output_folder);
end

% ���� BIDS �ļ����������� 'sub' ��ͷ���ļ���
subject_raw_folders = dir(fullfile(BIDS_folder, 'sub*'));

%% SEPIA Analysis (segment)
failed_subjects_file = fullfile(SEPIA_output_folder, 'failed_subjects_segment.txt');
fid = fopen(failed_subjects_file, 'w');
fclose(fid);

% General algorithm parameters

algorParam = struct();
algorParam.mode = 1;
algorParam.isBiasFieldCorr = 0 ;

subject_SEPIA_output_folders = dir(fullfile(SEPIA_output_folder, 'sub*'));

% ����ÿ�������ļ���
for i = 1:length(subject_SEPIA_output_folders)
    subject_folder_name = subject_SEPIA_output_folders(i).name;
    subject_output_folder = fullfile(SEPIA_output_folder, subject_folder_name);
    subject_raw_folder = fullfile(BIDS_folder, subject_folder_name);
    
    input = struct();
    input.gre      = fullfile(subject_raw_folder, 'anat', [subject_folder_name '_echo-1_part-mag_QSM.nii.gz']);
    input.greMask  = fullfile(subject_output_folder, 'Sepia_Chimap_reg_intermediate_files', 'reg_brain_mask.nii.gz');
    input.t1w      = fullfile(subject_raw_folder, 'anat', [subject_folder_name '_T1w.nii.gz']);
    input.t1wMask  = fullfile(subject_output_folder, 'Sepia_Chimap_reg_intermediate_files', 'structural_brain_mask.nii.gz');
    output_dir     = subject_output_folder;
    
    try
        % ���� sepiaIO ����
        get_CIT168_reinf_learn_labels(input,output_dir,algorParam);
    catch ME
        % ����������󣬼�¼��������
        fid = fopen(failed_subjects_file, 'a');
        fprintf(fid, '%s\n', subject_folder_name);
        fclose(fid);
        % ��ʾ������Ϣ�����жϳ���ִ��
        fprintf('Error processing subject %s: %s\n', subject_folder_name, ME.message);
    end
end

fprintf('Processing completed.\n');