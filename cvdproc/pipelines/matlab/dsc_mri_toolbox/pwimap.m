% For the process of DSC-MRI
% Youjie Wang, 2025-07-26. wangyoujie2002@163.com

% params
toolbox_dir = '/this/is/for/nipype/dsc_mri_toolbox'; % path to the DSC-MRI toolbox

DSC_info   = niftiinfo('/this/is/for/nipype/pwi_path');
load('/this/is/for/nipype/aif_conc.mat', 'aif_conc'); % aif_conc should be a vector of AIF concentration values
mask_path = '/this/is/for/nipype/mask_path'; % path to the mask file
TE = '/this/is/for/nipype/echo_time'; % echo time in seconds, e.g., 0.032 for 25ms\
TR = '/this/is/for/nipype/repetition_time'; % repetition time in seconds, e.g., 2.0 for 2000ms

output_cbv_path = '/this/is/for/nipype/cbv_path'; % path to save the CBV output
output_cbv_lc_path = '/this/is/for/nipype/cbv_lc_path'; % path to save the CBV_LC output
output_k2_path = '/this/is/for/nipype/k2_path'; % path to save the K2 output
output_cbf_svd_path = '/this/is/for/nipype/cbf_svd_path'; % path to save the CBF_SVD output
output_cbf_csvd_path = '/this/is/for/nipype/cbf_csvd_path'; % path to save the CBF_CSV output
output_cbf_osvd_path = '/this/is/for/nipype/cbf_osvd_path'; % path to save the CBF_OSVD output
output_mtt_svd_path = '/this/is/for/nipype/mtt_svd_path'; % path to save the MTT_SVD output
output_mtt_csvd_path = '/this/is/for/nipype/mtt_csvd_path'; % path to save the MTT_CSV output
output_mtt_osvd_path = '/this/is/for/nipype/mtt_osvd_path'; % path to save the MTT_OSVD output
output_ttp_path = '/this/is/for/nipype/ttp_path'; % path to save the TTP output
output_s0_path = '/this/is/for/nipype/s0_path'; % path to save the S0 output

% process
addpath(genpath(toolbox_dir)); % add the toolbox to the MATLAB path

mask_volume = niftiread(mask_path);
DSC_volume = niftiread(DSC_info);
mask_volume = double(mask_volume);
TE = str2double(TE); % Convert TE to double if it's a string
TR = str2double(TR); % Convert TR to double if it's a string

opt_dsc_MRI=DSC_mri_getOptions();
opt_dsc_MRI.aif.enable= 0; 
aif.fit.gv = aif_conc;
aif.conc = aif_conc;

[cbv,cbf,mtt,cbv_lc,ttp,mask,aif,conc,s0,fwhm,K2_map]=DSC_mri_core(DSC_volume,TE,TR,opt_dsc_MRI,aif, mask_volume);

% save to .nii.gz
ref_nii_path = mask_path;

ref_nii_info = niftiinfo(ref_nii_path);

ref_nii_info.Datatype = 'double';
niftiwrite(cbv,     output_cbv_path,               ref_nii_info, 'Compressed', true);
niftiwrite(cbv_lc,  output_cbv_lc_path,            ref_nii_info, 'Compressed', true);
niftiwrite(K2_map,  output_k2_path,                ref_nii_info, 'Compressed', true);
niftiwrite(cbf.svd.map,  output_cbf_svd_path,      ref_nii_info, 'Compressed', true);
niftiwrite(cbf.csvd.map,  output_cbf_csvd_path,    ref_nii_info, 'Compressed', true);
niftiwrite(cbf.osvd.map,  output_cbf_osvd_path,    ref_nii_info, 'Compressed', true);
niftiwrite(mtt.svd,  output_mtt_svd_path,          ref_nii_info, 'Compressed', true);
niftiwrite(mtt.csvd,  output_mtt_csvd_path,        ref_nii_info, 'Compressed', true);
niftiwrite(mtt.osvd,  output_mtt_osvd_path,        ref_nii_info, 'Compressed', true);
niftiwrite(ttp,  output_ttp_path,                  ref_nii_info, 'Compressed', true);
niftiwrite(s0,  output_s0_path,                    ref_nii_info, 'Compressed', true);