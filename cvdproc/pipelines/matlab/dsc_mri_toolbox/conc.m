% For the process of DSC-MRI
% Youjie Wang, 2025-07-26. wangyoujie2002@163.com

% params
toolbox_dir = '/this/is/for/nipype/dsc_mri_toolbox'; % path to the DSC-MRI toolbox

DSC_info   = niftiinfo('/this/is/for/nipype/pwi_path');
mask_path = '/this/is/for/nipype/mask_path'; % path to the mask file
TE = '/this/is/for/nipype/echo_time'; % echo time in seconds, e.g., 0.032 for 25ms\
TR = '/this/is/for/nipype/repetition_time'; % repetition time in seconds, e.g., 2.0 for 2000ms

output_conc_path = '/this/is/for/nipype/conc_path';

% process
addpath(genpath(toolbox_dir)); % add the toolbox to the MATLAB path

mask_volume = niftiread(mask_path);
DSC_volume = niftiread(DSC_info);
mask_volume = double(mask_volume);
TE = str2double(TE); % Convert TE to double if it's a string
TR = str2double(TR); % Convert TR to double if it's a string

opt_dsc_MRI=DSC_mri_getOptions();

DSC_volume = double(DSC_volume);
volumes = double(DSC_volume);
[nR,nC,nS,nT]=size(volumes);
opt_dsc_MRI.nR=nR;
opt_dsc_MRI.nC=nC;
opt_dsc_MRI.nS=nS;
opt_dsc_MRI.nT=nT;
opt_dsc_MRI.te=TE;
opt_dsc_MRI.tr=TR;
opt_dsc_MRI.time=0:TR:(nT-1)*TR;

[conc_result,S0map,bolus]=DSC_mri_conc(DSC_volume,mask_volume,opt_dsc_MRI);

conc_single = int16(conc_result);
niftiwrite(conc_single, output_conc_path, DSC_info, 'Compressed', true);