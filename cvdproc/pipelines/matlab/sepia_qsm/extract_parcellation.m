%% 0. Define Parcellation
% folder contains .nii or .nii.gz files
folderPath = 'F:\QSM_process\YC+YT_QSM_BIDS\derivatives\QSM_MNI_res\Filter3_PT';

% mask info
MaskFile = 'F:\QSM_process\YC+YT_QSM_BIDS\derivatives\VBA\VBA_v240929\all_clusters_PT-HC_TFCE_FWE5e-2.nii';
MaskInfoFile = 'F:\QSM_process\YC+YT_QSM_BIDS\derivatives\VBA\VBA_v240929\all_clusters_PT-HC_TFCE_FWE5e-2.csv';
ExtractType = 2;
OutputCsv = 'F:\QSM_process\YC+YT_QSM_BIDS\derivatives\VBA\VBA_v240929\all_clusters_PT-HC_TFCE_FWE5e-2_PT.csv';

%% 1. Parcellate image

niiFiles = dir(fullfile(folderPath, '*.nii'));
niiGzFiles = dir(fullfile(folderPath, '*.nii.gz'));

allFiles = [niiFiles; niiGzFiles];
ImgFiles = fullfile({allFiles.folder}, {allFiles.name});

func_Extract_ROI(ImgFiles, MaskFile, MaskInfoFile, ExtractType, OutputCsv);