% Script for WMH segmentation using LST toolbox in SPM
% Youjie Wang, 2025-08-31.

% Dependencies:
% - SPM12
% - LST (https://www.applied-statistics.de/lst.html)

clear;

FLAIR_path = '/this/is/for/nipype/flair';
T1w_path = '/this/is/for/nipype/t1w'; % can be empty ''
threshold = '/this/is/for/nipype/threshold';
output_path = '/this/is/for/nipype/output_path';

spm_path = '/this/is/for/nipype/spm_path';
lst_path = '/this/is/for/nipype/lst_path';

% output
WMH_mask_flair_filename = '/this/is/for/nipype/WMH_mask_flair';
WMH_prob_flair_filename = '/this/is/for/nipype/WMH_prob_flair';
WMH_mask_t1w_filename = '/this/is/for/nipype/WMH_mask_t1w';
WMH_prob_t1w_filename = '/this/is/for/nipype/WMH_prob_t1w';
FLAIR_in_T1w_filepath = '/this/is/for/nipype/FLAIR_in_T1w_path';

addpath(spm_path);
addpath(lst_path);

% change threshold to a float
threshold = str2num(threshold);

if ~exist(output_path, 'dir')
	mkdir(output_path);
end

FLAIR_img = gunzip(FLAIR_path, output_path);
movefile(FLAIR_img{1}, fullfile(output_path, 'FLAIR.nii'));

FLAIR_img_input = [fullfile(output_path, 'FLAIR.nii'), ',1'];

if exist(T1w_path, 'file')
	disp('T1w exists, outputs will be in T1w space');
    T1w_img = gunzip(T1w_path, output_path);
    movefile(T1w_img{1}, fullfile(output_path, 'T1w.nii'));
    
    T1w_img_input = [fullfile(output_path, 'T1w.nii'), ',1'];
    
    spm_get_defaults;
    global defaults;
    spm_jobman('initcfg');
    
    % Segmentation
    spm('defaults', 'fmri');
    matlabbatch{1}.spm.tools.LST.lpa.data_F2 = {FLAIR_img_input};
    matlabbatch{1}.spm.tools.LST.lpa.data_coreg = {T1w_img_input};
    matlabbatch{1}.spm.tools.LST.lpa.html_report = 0;

    try
        spm_jobman('run', matlabbatch);
    catch ME
        fprintf('Error processing %s: %s\n', FLAIR_img, ME.message);
    end

    clear matlabbatch
    
    % Binary thresholding
    ples_img = [fullfile(output_path, 'ples_lpa_mrFLAIR.nii'), ',1'];

    spm('defaults', 'fmri');
    matlabbatch{1}.spm.tools.LST.thresholding.data_plm = {ples_img};
    matlabbatch{1}.spm.tools.LST.thresholding.bin_thresh = threshold;
    spm_jobman('run', matlabbatch);
    clear matlabbatch
    
    prob_WMH = fullfile(output_path, 'ples_lpa_mrFLAIR.nii');
    binary_WMH = fullfile(output_path, ['bles_', num2str(threshold), '_lpa_mrFLAIR.nii']);
    outmat = fullfile(output_path, 'LST_lpa_mrFLAIR.mat');
    
    system(sprintf('gzip -c %s > %s', binary_WMH, WMH_mask_t1w_filename));
    system(sprintf('gzip -c %s > %s', prob_WMH, WMH_prob_t1w_filename));
else
    disp('Only FLAIR image set, outputs will be in FLAIR space');
    
    spm_get_defaults;
    global defaults;
    spm_jobman('initcfg');
    
    % Segmentation
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
    
    % Binary thresholding
    ples_img = [fullfile(output_path, 'ples_lpa_mFLAIR.nii'), ',1'];

    spm('defaults', 'fmri');
    matlabbatch{1}.spm.tools.LST.thresholding.data_plm = {ples_img};
    matlabbatch{1}.spm.tools.LST.thresholding.bin_thresh = threshold;
    spm_jobman('run', matlabbatch);
    clear matlabbatch
    
    prob_WMH = fullfile(output_path, 'ples_lpa_mFLAIR.nii');
    binary_WMH = fullfile(output_path, ['bles_', num2str(threshold), '_lpa_mFLAIR.nii']);
    outmat = fullfile(output_path, 'LST_lpa_mFLAIR.mat');
    
    system(sprintf('gzip -c %s > %s', binary_WMH, WMH_mask_flair_filename));
    system(sprintf('gzip -c %s > %s', prob_WMH, WMH_prob_flair_filename));
end

% delete temp files
delete(prob_WMH);
delete(binary_WMH);
delete(fullfile(output_path, 'FLAIR.nii'));
delete(outmat);

if exist(T1w_path, 'file')
    delete(fullfile(output_path, 'T1w.nii'));
    mrFLAIR_path = fullfile(output_path, 'mrFLAIR.nii');
    % gzip the registered FLAIR in T1w space
    system(sprintf('gzip -c %s > %s', mrFLAIR_path, FLAIR_in_T1w_filepath));
    delete(fullfile(output_path, 'mrFLAIR.nii'));
else
    delete(fullfile(output_path, 'mFLAIR.nii'));
end

disp('All done for LST WMH segmentation');
