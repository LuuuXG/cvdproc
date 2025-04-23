%% Define the inputs
input_qsm_bids_dir = '/this/is/prepared/for/nipype/input_qsm_bids_dir';
% QSM data are in the `input_qsm_bids_dir`
phase_image_correction = false; % For GE acquisition
reverse_phase = 0; % For GE acquisition
subject_output_folder = '/this/is/prepared/for/nipype/subject_output_folder';
sepia_toolbox_path = '/this/is/prepared/for/nipype/sepia_toolbox_path';

%% Add the SEPIA toolbox to the path (with subfolders)
%addpath(genpath(sepia_toolbox_path));

%% 01 - Prepare the input for SEPIA from raw BIDS data,
%       and correct the inter-slice polarity differences (For GE acquisition)
fprintf('Prepare the input for SEPIA from raw BIDS data\n');
outputPrefix = fullfile(subject_output_folder, 'Sepia_');
% use `read_bids_to_filelist` in SEPIA
read_bids_to_filelist(input_qsm_bids_dir, outputPrefix);

if phase_image_correction
    temp_dir = fullfile(subject_output_folder, 'uncorrected_phase');
    if ~exist(temp_dir,'dir')
        mkdir(temp_dir);
    end

    movefile(fullfile(subject_output_folder, 'Sepia_part-phase.nii.gz'), fullfile(temp_dir,'Sepia_part-phase.nii.gz'));

    mag = load_nii_img_only(fullfile(subject_output_folder, 'Sepia_part-mag.nii.gz'));
    pha = load_nii_img_only(fullfile(temp_dir,'Sepia_part-phase.nii.gz'));

    % create complex-valued data
    img = mag .* exp(1i*pha);

    % separate the real and imaginary parts
    img_real = real(img);
    img_imag = imag(img);

    % correct inter-slice polarity differences
    img_real(:,:,2:2:end,:) = -img_real(:,:,2:2:end,:);
    img_imag(:,:,2:2:end,:) = -img_imag(:,:,2:2:end,:);

    % compute the correct phase from compex-valued data
    img_phase = angle(complex(img_real, img_imag));

    % export the data into disk
    save_nii_img_only(fullfile(temp_dir,'Sepia_part-phase.nii.gz'), fullfile(subject_output_folder, 'Sepia_part-phase_corr.nii.gz'), img_phase);
end

%% 02 - Calculate the Chimap (QSM value/susceptibility map)
% General algorithm parameters
algorParam = struct();
algorParam.general.isBET = 1 ;
algorParam.general.fractional_threshold = 0.3 ;
algorParam.general.gradient_threshold = 0 ;
algorParam.general.isInvert = reverse_phase ;
algorParam.general.isRefineBrainMask = 0 ;
% Total field recovery algorithm parameters
algorParam.unwrap.echoCombMethod = 'ROMEO total field calculation' ;
algorParam.unwrap.offsetCorrect = 'On' ;
algorParam.unwrap.mask = 'SEPIA mask' ;
algorParam.unwrap.qualitymaskThreshold = 0.5 ;
algorParam.unwrap.useRomeoMask = 0 ;
algorParam.unwrap.isEddyCorrect = 0 ;
algorParam.unwrap.isSaveUnwrappedEcho = 0 ;
% Background field removal algorithm parameters
algorParam.bfr.refine_method = 'None' ;
algorParam.bfr.refine_order = 2 ;
algorParam.bfr.erode_radius = 0 ;
algorParam.bfr.erode_before_radius = 1 ;
algorParam.bfr.method = 'VSHARP' ;
algorParam.bfr.radius = [12:-1:1] ;
% QSM algorithm parameters
algorParam.qsm.reference_tissue = 'Brain mask' ;
algorParam.qsm.method = 'FANSI' ;
algorParam.qsm.tol = 0.1 ;
algorParam.qsm.maxiter = 400 ;
algorParam.qsm.lambda = 0.0005 ;
algorParam.qsm.mu1 = 0.05 ;
algorParam.qsm.mu2 = 1 ;
algorParam.qsm.solver = 'Non-linear' ;
algorParam.qsm.constraint = 'TV' ;
algorParam.qsm.gradient_mode = 'Vector field' ;
algorParam.qsm.isGPU = 1 ;
algorParam.qsm.isWeakHarmonic = 1 ;
algorParam.qsm.beta = 150 ;
algorParam.qsm.muh = 3 ;

sepiaIO(subject_output_folder, fullfile(subject_output_folder, 'Sepia'), '', algorParam);

%% 03 - Calculate the R2*, S0, and T2* maps
% General algorithm parameters
algorParam = struct();
algorParam.general.isBET = 0 ;
algorParam.general.isInvert = 0 ;
algorParam.general.isRefineBrainMask = 0 ;
% R2* algorithm parameters
algorParam.r2s.method = 'Trapezoidal' ;
algorParam.r2s.s0mode = 'Weighted sum' ;

sepiaIO(subject_output_folder, fullfile(subject_output_folder, 'Sepia'), [fullfile(subject_output_folder, 'Sepia_mask_brain.nii.gz')], algorParam);

%% 04 - Calculate the SWI map (use CLEAR-SWI)
setenv('LD_LIBRARY_PATH', '');
% SWI/SMWI algorithm parameters
algorParam = struct();
algorParam.swismwi.method = 'CLEAR-SWI' ;
algorParam.swismwi.phaseScalingStrength = 4 ;
algorParam.swismwi.echoCombineMethodAdd = 1 ;
algorParam.swismwi.slice_mIP = 4 ;
algorParam.swismwi.filterSize = '[4,4,0]' ;
algorParam.swismwi.echoes = ':' ;
algorParam.swismwi.softplusScaling = 1 ;
algorParam.swismwi.sensitivityCorrection = 1 ;
algorParam.swismwi.ismIP = 1 ;
algorParam.swismwi.phaseScalingType = 'tanh' ;
algorParam.swismwi.unwrappingAlgorithm = 'laplacian' ;
algorParam.swismwi.echoCombineMethod = 'SNR' ;

input = struct();

if reverse_phase == 1
    input(1).name = fullfile(subject_output_folder, 'Sepia_part-phase_reverse.nii.gz');
elseif reverse_phase == 0
    input(1).name = fullfile(subject_output_folder, 'Sepia_part-phase.nii.gz');
end

input(2).name = fullfile(subject_output_folder, 'Sepia_part-mag.nii.gz');
input(3).name = fullfile(subject_output_folder, 'Sepia_header.mat');
output_basename = fullfile(subject_output_folder, 'Sepia');

SWISMWIIOWrapper(input, output_basename, algorParam);

exit;