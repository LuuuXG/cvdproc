%% 00_Folder creation
% Define BIDS (BIDS-like) root folder path
BIDS_folder = 'F:\BIDS\YC+YT_QSM_BIDS';

% Define derivatives root folder path
derivatives_folder = fullfile(BIDS_folder, 'derivatives');

if ~exist(derivatives_folder, 'dir')
    mkdir(derivatives_folder);
end

% Define SEPIA derivatives folder path?
SEPIA_output_folder = fullfile(derivatives_folder, 'SEPIA');

if ~exist(SEPIA_output_folder, 'dir')
    mkdir(SEPIA_output_folder);
end

%% %% 00_Data preparation
% Search for each subject
subject_raw_folders = dir(fullfile(BIDS_folder, 'sub*'));

% Record failed subjects
failed_subjects_file = fullfile(SEPIA_output_folder, 'failed_subjects_BIDS2SEPIA.txt');
fid = fopen(failed_subjects_file, 'w');
fclose(fid);

% Generate SEPIA input file for each subject
for i = 1:length(subject_raw_folders)
    subject_folder_name = subject_raw_folders(i).name;
    subject_output_folder = fullfile(SEPIA_output_folder, subject_folder_name);

    sub_BIDS_fmap_folder = fullfile(BIDS_folder, subject_raw_folders(i).name, 'fmap');
    
    if ~exist(subject_output_folder, 'dir')
        mkdir(subject_output_folder);
        
        % correct phase image
        try
            temp_dir = fullfile(subject_output_folder,'uncorrected_phase');
            if ~exist(temp_dir,'dir')
                mkdir(temp_dir);
            end

            outputPrefix = fullfile(subject_output_folder, 'Sepia_');
            read_bids_to_filelist(sub_BIDS_fmap_folder, outputPrefix);
            
            movefile(fullfile(subject_output_folder, 'Sepia_part-phase.nii.gz'), fullfile(temp_dir,'Sepia_part-phase.nii.gz'));
            
            % load data
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
        catch ME
            fid = fopen(failed_subjects_file, 'a');
            fprintf(fid, '%s\n', subject_folder_name);
            fclose(fid);
            
            fprintf('Error processing subject %s: %s\n', subject_folder_name, ME.message);
        end
    end
end

fprintf('Processing completed.\n');

%% 02_Chimap
% Record failed subjects
failed_subjects_file = fullfile(SEPIA_output_folder, 'failed_subjects_chimap.txt');
fid = fopen(failed_subjects_file, 'w');
fclose(fid);

% Search for each subject
subject_raw_folders = dir(fullfile(BIDS_folder, 'sub*'));

% General algorithm parameters
algorParam = struct();
algorParam.general.isBET = 1 ;
algorParam.general.fractional_threshold = 0.3 ;
algorParam.general.gradient_threshold = 0 ;
algorParam.general.isInvert = 1 ;
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

% Generate QSM Chimap (Susceptibility map) for each subject
for i = 1:length(subject_raw_folders)
    subject_folder_name = subject_raw_folders(i).name;
    subject_output_folder = fullfile(SEPIA_output_folder, subject_folder_name);
    
    input_folder = subject_output_folder;
    output_basename = fullfile(subject_output_folder, 'Sepia');
    mask_filename = '';
    
    try
        sepiaIO(input_folder, output_basename, mask_filename, algorParam);
    catch ME
        fid = fopen(failed_subjects_file, 'a');
        fprintf(fid, '%s\n', subject_folder_name);
        fclose(fid);
        
        fprintf('Error processing subject %s: %s\n', subject_folder_name, ME.message);
    end
end

fprintf('Processing completed.\n');

%% 03_R2star+S0+T2star
failed_subjects_file = fullfile(SEPIA_output_folder, 'failed_subjects_r2ss0t2s.txt');
fid = fopen(failed_subjects_file, 'w');
fclose(fid);

% Search for each subject
subject_raw_folders = dir(fullfile(BIDS_folder, 'sub*'));

% General algorithm parameters
algorParam = struct();
algorParam.general.isBET = 0 ;
algorParam.general.isInvert = 0 ;
algorParam.general.isRefineBrainMask = 0 ;
% R2* algorithm parameters
algorParam.r2s.method = 'Trapezoidal' ;
algorParam.r2s.s0mode = 'Weighted sum' ;

subject_SEPIA_output_folders = dir(fullfile(SEPIA_output_folder, 'sub*'));

% Generate R2star, S0, and T2star map for each subject
for i = 1:length(subject_SEPIA_output_folders)
    subject_folder_name = subject_SEPIA_output_folders(i).name;
    subject_output_folder = fullfile(SEPIA_output_folder, subject_folder_name);
    
    input = struct();
    input = subject_output_folder;
    output_basename = fullfile(subject_output_folder, 'Sepia');
    mask_filename = [fullfile(subject_output_folder, 'Sepia_mask_brain.nii.gz')];
    
    try
        sepiaIO(input,output_basename,mask_filename,algorParam);
    catch ME
        fid = fopen(failed_subjects_file, 'a');
        fprintf(fid, '%s\n', subject_folder_name);
        fclose(fid);
        
        fprintf('Error processing subject %s: %s\n', subject_folder_name, ME.message);
    end
end

fprintf('Processing completed.\n');

%% 04_SWI
failed_subjects_file = fullfile(SEPIA_output_folder, 'failed_subjects_swi.txt');
fid = fopen(failed_subjects_file, 'w');
fclose(fid);

% SWI/SMWI algorithm parameters
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

subject_SEPIA_output_folders = dir(fullfile(SEPIA_output_folder, 'sub*'));

% Generate SWI and mIP for each subject
for i = 1:length(subject_SEPIA_output_folders)
    subject_folder_name = subject_SEPIA_output_folders(i).name;
    subject_output_folder = fullfile(SEPIA_output_folder, subject_folder_name);
    
    % Input/Output filenames
    input = struct();
    input(1).name = fullfile(subject_output_folder, 'Sepia_part-phase_reverse.nii.gz');
    input(2).name = fullfile(subject_output_folder, 'Sepia_part-mag.nii.gz');
    input(3).name = fullfile(subject_output_folder, 'Sepia_header.mat');
    output_basename = fullfile(subject_output_folder, 'Sepia');
    
    try
        SWISMWIIOWrapper(input,output_basename,algorParam);
    catch ME
        fid = fopen(failed_subjects_file, 'a');
        fprintf(fid, '%s\n', subject_folder_name);
        fclose(fid);

        fprintf('Error processing subject %s: %s\n', subject_folder_name, ME.message);
    end
end

fprintf('Processing completed.\n');