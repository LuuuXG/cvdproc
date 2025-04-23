% Function for WMH segmentation using LST-LPA
% Youjie Wang. 2024-12-07

% Dependencies:
% - SPM12
% - LST (https://www.applied-statistics.de/lst.html)

% Input:
% - FLAIR_path: path to FLAIR image (.nii.gz)
% - thresholds: list of thresholds for binary thresholding
% - output_path: path to save the output

% Output:
% - mFLAIR.nii.gz
% - ples_lpa_mFLAIR.nii.gz (probability map)
% - LST_lpa_mFLAIR.mat (probability map)
% - bles_<threshold>_lpa_mFLAIR.nii.gz (binary mask for each threshold)

function WMH_seg_LST(FLAIR_path, thresholds, output_path)
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end

    spm12_path = spm('Dir');
    
    % unzip the FLAIR image to output folder and rename to 'FLAIR.nii
    FLAIR_img = gunzip(FLAIR_path, output_path);
    movefile(FLAIR_img{1}, fullfile(output_path, 'FLAIR.nii'));

    FLAIR_img_input = [fullfile(output_path, 'FLAIR.nii'), ',1'];
    
    % 0. Initialize SPM
    addpath(spm12_path);

    spm_get_defaults;
    global defaults;
    spm_jobman('initcfg');

    % 1. Run LST-LPA
    spm('defaults', 'fmri');
    matlabbatch{1}.spm.tools.LST.lpa.data_F2 = {FLAIR_img_input};
    matlabbatch{1}.spm.tools.LST.lpa.data_coreg = {''};
    matlabbatch{1}.spm.tools.LST.lpa.html_report = 0;

    try
        spm_jobman('run', matlabbatch);
    catch ME
        fprintf('Error processing %s: %s\n', FLAIR_img, ME.message);
    end

    clear matlabbatch

    % 2. Binary thresholding
    ples_img = [fullfile(output_path, 'ples_lpa_mFLAIR.nii'), ',1'];

    for thr = thresholds
        spm('defaults', 'fmri');
        matlabbatch{1}.spm.tools.LST.thresholding.data_plm = {ples_img};
        matlabbatch{1}.spm.tools.LST.thresholding.bin_thresh = thr;
        spm_jobman('run', matlabbatch);
        clear matlabbatch

        binary_WMH = fullfile(output_path, ['bles_', num2str(thr), '_lpa_mFLAIR.nii']);

        gzip(binary_WMH);
        delete(binary_WMH);

    end

    % gzip the output files and clean up
    %gzip(fullfile(output_path, 'mFLAIR.nii'));
    delete(fullfile(output_path, 'mFLAIR.nii'));
    delete(fullfile(output_path, 'LST_lpa_mFLAIR.mat'))
    gzip(fullfile(output_path, 'ples_lpa_mFLAIR.nii'));
    delete(fullfile(output_path, 'ples_lpa_mFLAIR.nii'));
    delete(fullfile(output_path, 'FLAIR.nii'));
    
    fprintf('WMH segmentation using LST-LPA is done for %s\n', FLAIR_path);
end