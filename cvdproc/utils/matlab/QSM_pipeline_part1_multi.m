%% =========================================
% Batch wrapper for running multiple subjects/sessions
% Put this block ABOVE your existing pipeline code.
% Only code comments are in English as requested.
% =========================================

clear;
cd(fileparts(matlab.desktop.editor.getActiveFilename));

%% ------- User controls (safe to edit) -------
bids_root_dir = 'F:/BIDS/WCH_AF_Project';
cvdproc_dir   = 'E:/Codes/cvdproc';
FS_HOME       = '/usr/local/freesurfer/7-dev';
ants_path     = ""; % currently not used

% How to pick subjects/sessions:
%  - subjects = 'auto' will scan BIDS for all sub-*
%  - or provide a cell array, e.g., {'HC0261','HC0262'}
subjects = { ...
    'HC0062','HC0063','HC0064','HC0065', ...
    'HC0066','HC0067','HC0068','HC0069','HC0070', ...
    'HC0071','HC0072','HC0073','HC0074','HC0075', ...
    'HC0076','HC0077','HC0078','HC0079','HC0080', ...
    'HC0082','HC0083','HC0084','HC0085', ...
    'HC0086','HC0087','HC0088','HC0089', ...
    'HC0090','HC0091','HC0092','HC0093','HC0094', ...
    'HC0095','HC0096','HC0097','HC0098','HC0099', ...
    'HC0100','HC0101','HC0102','HC0104', ...
    'HC0105','HC0106','HC0107','HC0108' ...
};

% sessions_mode:
%  - 'auto'  : scan ses-* under each subject
%  - 'single': force a single session id for all subjects (set forced_session_id)
%  - 'list'  : provide sessions_list_map for per-subject sessions (containers.Map)
%  - 'none'  : for datasets without ses-* level (rare in your data)
sessions_mode = 'single';
forced_session_id = 'baseline';  % used only when sessions_mode == 'single'

% GPU and environment checks you already had:
parallel.gpu.enableCUDAForwardCompatibility(true);
gpuDevice(); % must see the GPU info by gpuDevice()

% process settings
phase_image_correction = false; % set to true For GE phase data
reverse_phase = 0; % set to 1 For GE phase data

%% ------- Discover subjects -------
if ischar(subjects) && strcmpi(subjects, 'auto')
    subjects = find_bids_subjects(bids_root_dir);
    if isempty(subjects)
        error('No sub-* found under: %s', bids_root_dir);
    end
end
assert(iscellstr(subjects), 'subjects must be ''auto'' or a cellstr of subject IDs.');

%% ------- Loop over subjects & sessions -------
for si = 1:numel(subjects)
    subject_id = subjects{si};

    % Determine session list for this subject
    switch lower(sessions_mode)
        case 'auto'
            ses_list = find_bids_sessions(bids_root_dir, subject_id);
            if isempty(ses_list)
                % If no ses-* folders found, try a common default you use:
                cand = fullfile(bids_root_dir, ['sub-' subject_id], 'ses-baseline');
                if isfolder(cand)
                    ses_list = {'baseline'};
                else
                    error('No ses-* found for sub-%s. Please set sessions_mode or check data.', subject_id);
                end
            end

        case 'single'
            ses_list = {forced_session_id};

        case 'none'
            % If your data really has no 'ses-*' level, put a placeholder
            ses_list = {''};

        case 'list'
            error('Implement your own per-subject sessions list via a containers.Map if needed.');

        otherwise
            error('Unknown sessions_mode: %s', sessions_mode);
    end

    for ti = 1:numel(ses_list)
        session_id = ses_list{ti};
        fprintf('\n=============================================\n');
        fprintf('Running subject: %s | session: %s\n', subject_id, session_id);
        fprintf('=============================================\n');

        %% ----------------------00 Load Funtions---------------------------
        chisep_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'Chisep_Toolbox_v1.2');
        vesselseg_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'Chisep_Toolbox_v1.2', 'models');
        sepia_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'sepia');
        sti_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'STISuite_V3.0');
        medi_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'MEDI_toolbox');
        mritools_win_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'mritools_Windows_3.5.5');
        mritools_linux_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'mritools_Linux_3.5.5');
        segue_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'SEGUE_28012021');
        mrisc_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'MRI_susceptibility_calculation_12072021');
        fansi_path = fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'FANSI-toolbox');
        
        addpath(genpath(fullfile(cvdproc_dir, 'cvdproc', 'utils', 'matlab', 'QSM_pipeline_part1_func')));
        addpath(genpath(fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'xiangruili-dicm2nii-3fe1a27')));
        addpath(genpath(fullfile(cvdproc_dir, 'cvdproc', 'data', 'matlab_toolbox', 'nifti_toolbox')));
        addpath(genpath(sepia_path));
        addpath(genpath(sti_path));
        addpath(genpath(medi_path));
        addpath(genpath(segue_path));
        addpath(genpath(mrisc_path));
        addpath(genpath(fansi_path));
        addpath(genpath(chisep_path));
        addpath(genpath(vesselseg_path));
        
        if ispc
            mritools_path = mritools_win_path;
        elseif isunix || ismac
            mritools_path = mritools_linux_path;
        else
            error('Unsupported platform.');
        end
        addpath(genpath(mritools_path));
        
        sepia_toolbox_config = fullfile(sepia_path, 'SpecifyToolboxesDirectory.m');
        % Collect resolved paths (ensure trailing filesep)
        ensure_trailing = @(p) string(p) + filesep;
        
        MEDI_HOME     = ensure_trailing(medi_path);
        STISuite_HOME = ensure_trailing(sti_path);
        FANSI_HOME    = ensure_trailing(fansi_path);
        SEGUE_HOME    = ensure_trailing(segue_path);
        MRITOOLS_HOME = ensure_trailing(mritools_path);
        MRISC_HOME    = ensure_trailing(mrisc_path);
        
        % If you have ANTs, set its bin folder here; otherwise keep empty:
        ANTS_HOME = ants_path;  % keep as empty string if not installed
        
        % Make a backup before overwrite
        backup_file = sepia_toolbox_config + ".bak";
        if ~isfile(backup_file)
            try, copyfile(sepia_toolbox_config, backup_file); end
        end
        
        % Build new file content
        hdr = [
        "%% SpecifyToolboxesDirectory.m (auto-generated by script)"
        "% This file sets toolbox root paths used by SEPIA."
        "% Modify the variables below if your installation paths change."
        ""
        "%% Toolbox roots (each ends with filesep)"
        ];
        
        % Lines to write (use single quotes in MATLAB source)
        lines = [
        "MEDI_HOME      = '" + MEDI_HOME     + "';"
        "STISuite_HOME  = '" + STISuite_HOME + "';"
        "FANSI_HOME     = '" + FANSI_HOME    + "';"
        "SEGUE_HOME     = '" + SEGUE_HOME    + "';"
        "MRITOOLS_HOME  = '" + MRITOOLS_HOME + "';"
        "MRISC_HOME     = '" + MRISC_HOME    + "';"
        "ANTS_HOME      = '" + ANTS_HOME     + "';"
        ""
        "% Sanity check display (optional)"
        "disp('SEPIA toolbox paths set:');"
        "disp(MEDI_HOME);"
        "disp(STISuite_HOME);"
        "disp(FANSI_HOME);"
        "disp(SEGUE_HOME);"
        "disp(MRITOOLS_HOME);"
        "disp(MRISC_HOME);"
        "disp(ANTS_HOME);"
        ];
        
        % Write file
        fid = fopen(sepia_toolbox_config, 'w');
        assert(fid>0, 'Cannot open %s for writing.', sepia_toolbox_config);
        fprintf(fid, '%s\n', hdr);
        fprintf(fid, '%s\n', lines);
        fclose(fid);
        
        disp('Updated SEPIA SpecifyToolboxesDirectory.m successfully.');
        %% ------------------------01 Load Data-----------------------------
        sub_dir = sprintf('sub-%s', subject_id);
        ses_dir = sprintf('ses-%s', session_id);
        
        qsm_bids_rawdata_dir = char(fullfile(bids_root_dir, ...
            sub_dir, ...
            ses_dir, ...
            'qsm'));
        
        assert(isfolder(qsm_bids_rawdata_dir), 'Folder does not exist: %s', qsm_bids_rawdata_dir);
        
        % 1) Require at least one part-mag
        mag_files = dir(fullfile(qsm_bids_rawdata_dir, '*_part-mag_*GRE.nii.gz'));
        if isempty(mag_files)
            error('No part-mag files found in: %s', qsm_bids_rawdata_dir);
        end
        
        % 2) Already have part-phase?
        phase_files = dir(fullfile(qsm_bids_rawdata_dir, '*_part-phase_*GRE.nii.gz'));
        if ~isempty(phase_files)
            fprintf('[Info] Found %d part-phase files. No synthesis is needed.\n', numel(phase_files));
        else
            % No phase at all -> try to synthesize from real/imag
            real_files = dir(fullfile(qsm_bids_rawdata_dir, '*_part-real_*GRE.nii.gz'));
            imag_files = dir(fullfile(qsm_bids_rawdata_dir, '*_part-imag_*GRE.nii.gz'));
            if isempty(real_files) || isempty(imag_files)
                error('No part-phase found, and real/imag pairs are incomplete (real=%d, imag=%d).', ...
                      numel(real_files), numel(imag_files));
            end
        
            % Build echo maps from filenames
            real_map = containers.Map('KeyType','char','ValueType','any');
            for i = 1:numel(real_files)
                f = fullfile(real_files(i).folder, real_files(i).name);
                ek = parse_echo_from_name(f);
                if ~isempty(ek)
                    template_json = replace_ext(f, '.json');
                    if ~isfile(template_json)
                        error('Missing JSON sidecar for: %s', f);
                    end
                    real_map(ek) = struct('nii', f, 'json', template_json);
                else
                    warning('Cannot parse echo index from: %s', f);
                end
            end
        
            imag_map = containers.Map('KeyType','char','ValueType','any');
            for i = 1:numel(imag_files)
                f = fullfile(imag_files(i).folder, imag_files(i).name);
                ek = parse_echo_from_name(f);
                if ~isempty(ek)
                    imag_map(ek) = struct('nii', f);
                else
                    warning('Cannot parse echo index from: %s', f);
                end
            end
        
            % Intersect echoes present in both maps
            echoes = intersect(real_map.keys, imag_map.keys);
            if isempty(echoes)
                error('No matching echo indices between real and imag files.');
            end
        
            fprintf('[Info] Synthesizing phase for %d echoes...\n', numel(echoes));
        
            for k = 1:numel(echoes)
                ek = echoes{k};
                real_file = real_map(ek).nii;
                imag_file = imag_map(ek).nii;
                real_json = real_map(ek).json;
        
                % Output names by replacing part-real -> part-phase
                out_nii  = replace_part_token(real_file, 'part-real', 'part-phase');
                out_json = replace_part_token(real_json, 'part-real', 'part-phase');
        
                % Read real/imag and compute phase
                infoR = niftiinfo(real_file);
                R = niftiread(infoR);
                I = niftiread(imag_file);
        
                if ~isequal(size(R), size(I))
                    error('Shape mismatch for echo %s: real %s vs imag %s', ek, mat2str(size(R)), mat2str(size(I)));
                end
        
                phaseData = angle(complex(double(R), double(I)));
        
                % Prepare header and write NIfTI (prefer single)
                infoOut = infoR;
                infoOut.Datatype = 'single';
                infoOut.BitsPerPixel = 32;
        
                % Always write first as .nii (without extension)
                out_base = regexprep(out_nii, '\.nii(\.gz)?$', ''); % strip any .nii/.nii.gz
                tmpNii = [out_base '.nii'];
        
                niftiwrite(single(phaseData), out_base, infoOut);  % produces tmpNii
        
                % Now gzip it
                if isfile([tmpNii '.gz']), delete([tmpNii '.gz']); end
                gzip(tmpNii);
                delete(tmpNii);
        
                % Move to final .nii.gz (ensures consistent name)
                finalNiiGz = [out_base '.nii.gz'];
                if ~strcmp(finalNiiGz, out_nii)
                    movefile(finalNiiGz, out_nii);
                end
        
                % Write JSON sidecar: copy from real.json and override ImageType
                T = jsondecode(fileread(real_json));
                T.ImageType = {'ORIGINAL','PRIMARY','OTHER','PHASE'};
                fid = fopen(out_json, 'w'); assert(fid>0, 'Cannot write: %s', out_json);
                fwrite(fid, jsonencode(T));
                fclose(fid);
        
                fprintf('[OK] echo-%s -> %s\n', ek, out_nii);
            end
        
            fprintf('[Done] All missing phase echoes have been synthesized.\n');
        end
        
        fprintf('[Info] Creating the output directory...\n');
        qsm_bids_derivatives_dir = fullfile(bids_root_dir, ...
            'derivatives', ...
            'qsm_pipeline', ...
            sub_dir, ...
            ses_dir);
        
        if ~exist(qsm_bids_derivatives_dir,'dir')
                mkdir(qsm_bids_derivatives_dir);
        end
        fprintf('[Done] The output directory has been created\n');
        
        sub_and_ses_str = sprintf('sub-%s_ses-%s_', subject_id, session_id);
        raw_qsm_prefix = char(fullfile(qsm_bids_derivatives_dir, sub_and_ses_str));
        read_bids_to_filelist(qsm_bids_rawdata_dir, raw_qsm_prefix);
        
        mag_old = sprintf('%spart-mag.nii.gz', raw_qsm_prefix);
        mag_raw = sprintf('%spart-mag_GRE.nii.gz', raw_qsm_prefix);
        movefile(mag_old, mag_raw);
        phase_old = sprintf('%spart-phase.nii.gz', raw_qsm_prefix);
        phase_raw = sprintf('%spart-phase_GRE.nii.gz', raw_qsm_prefix);
        movefile(phase_old, phase_raw);
        header_old = sprintf('%sheader.mat', raw_qsm_prefix);
        sepia_header = sprintf('%sdesc-sepia_header.mat', raw_qsm_prefix);
        movefile(header_old, sepia_header);
        
        fprintf('[Info] Preprocessing for magnitude and phase image\n');
        if phase_image_correction
            mag = load_nii_img_only(mag_raw);
            pha = load_nii_img_only(phase_raw);
        
            img = mag .* exp(1i*pha);
        
            img_real = real(img);
            img_imag = imag(img);
        
            img_real(:,:,2:2:end,:) = -img_real(:,:,2:2:end,:);
            img_imag(:,:,2:2:end,:) = -img_imag(:,:,2:2:end,:);
        
            img_phase = angle(complex(img_real, img_imag));
            
            phase_raw_corrected = sprintf('%spart-phase_desc-corrected_GRE.nii.gz', raw_qsm_prefix);
            save_nii_img_only(phase_raw, phase_raw_corrected, img_phase);
            delete(phase_raw);
            phase_raw = phase_raw_corrected;
        end
        
        mag = load_nii_img_only(mag_raw);
        pha = load_nii_img_only(phase_raw);
        maxval = max(pha(:));
        minval = min(pha(:));
        pha = (pha-(minval+maxval)/2)/(maxval-minval)*2*pi;
        Tukey_smooth = double(0.4);
        imgc = mag .* exp(1i*pha * (-1)^(reverse_phase));
        imgc = tukey_windowing(imgc,Tukey_smooth);
        Mag_Tukey_data = abs(imgc);
        Phs_Tukey_data = angle(imgc);
        mag_smooth = sprintf('%spart-mag_desc-smoothed_GRE.nii.gz', raw_qsm_prefix);
        phase_smooth = sprintf('%spart-phase_desc-smoothed_GRE.nii.gz', raw_qsm_prefix);
        save_nii_img_only(mag_raw, mag_smooth, Mag_Tukey_data);
        save_nii_img_only(phase_raw, phase_smooth, Phs_Tukey_data);
        
        fprintf('[Done] Finished Loading data\n');
        
        %% adds: create brain mask by synthstrip
        search_pattern = fullfile(qsm_bids_rawdata_dir, '*echo-1*part-mag*.nii.gz');
        
        mag_files = dir(search_pattern);
        
        if isempty(mag_files)
            error('No magnitude echo-1 file found in %s', qsm_bids_rawdata_dir);
        else
            first_mag_echo1 = fullfile(qsm_bids_rawdata_dir, mag_files(1).name);
            fprintf('Found first magnitude echo-1 file: %s\n', first_mag_echo1);
        end
        
        brain_mask_out = sprintf('%slabel-brain_mask.nii.gz', raw_qsm_prefix);
        
        in_wsl  = win2wsl(first_mag_echo1);
        mask_wsl = win2wsl(brain_mask_out);
        
        cmd = sprintf(['wsl bash -lc ''export FREESURFER_HOME=%s; ' ...
                           'source "$FREESURFER_HOME/SetUpFreeSurfer.sh"; ' ...
                           'mri_synthstrip -i "%s" -m "%s"'''], ...
                           FS_HOME, in_wsl, mask_wsl);
        fprintf('Running: %s\n', cmd);
        [status, result] = system(cmd);
        
        if status == 0
            fprintf('mri_synthstrip finished successfully.\n');
        else
            fprintf('Error running mri_synthstrip:\n%s\n', result);
        end
        
        %% -------------02 SEPIA: SWI + R2star, T2star, S0------------------
        sub_and_ses_str_sepia = sprintf('sub-%s_ses-%s', subject_id, session_id);
        sepia_output_dir = fullfile(qsm_bids_derivatives_dir, 'sepia_output');
        sepia_output_basename = fullfile(qsm_bids_derivatives_dir, 'sepia_output', sub_and_ses_str_sepia);
        
        % -----------------------------------SWI-----------------------------------
        % Input/Output filenames
        input(1).name = phase_smooth;
        input(2).name = mag_smooth;
        input(3).name = sepia_header;
        output_basename = sepia_output_basename;
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
        
        SWISMWIIOWrapper(input,output_basename,algorParam);
        
        % ----------------------------R2star, T2star, S0---------------------------
        % Input/Output filenames
        input = struct();
        input(1).name = '' ;
        input(2).name = mag_smooth;
        input(3).name = '' ;
        input(4).name = sepia_header;
        output_basename = sepia_output_basename;
        mask_filename = [brain_mask_out] ;
        
        % General algorithm parameters
        algorParam = struct();
        algorParam.general.isBET = 0 ;
        algorParam.general.isInvert = 0 ;
        algorParam.general.isRefineBrainMask = 0 ;
        % R2* algorithm parameters
        algorParam.r2s.method = 'ARLO' ;
        algorParam.r2s.s0mode = 'Weighted sum' ;
        
        sepiaIO(input,output_basename,mask_filename,algorParam);
        
        % ----------------------------Post-process---------------------------------
        r2s = fullfile(sepia_output_dir, sprintf('sub-%s_ses-%s_R2starmap.nii.gz', subject_id, session_id));
        swi = fullfile(sepia_output_dir, sprintf('sub-%s_ses-%s_clearswi.nii.gz', subject_id, session_id));
        mip = fullfile(sepia_output_dir, sprintf('sub-%s_ses-%s_clearswi-minIP.nii.gz', subject_id, session_id));
        
        mask = niftiread(brain_mask_out);
        mask = logical(mask);
        
        files_in = {swi, mip};
        
        for i = 1:numel(files_in)
            data = niftiread(files_in{i});
            info = niftiinfo(files_in{i});
        
            % 应用 mask
            data_masked = data .* cast(mask, class(data));
        
            % 去掉后缀，得到 prefix
            [folder, name, ~] = fileparts(files_in{i});
            if endsWith(name, '.nii')  % 处理 .nii.gz 的情况
                [~, name, ~] = fileparts(name);
            end
            out_prefix = fullfile(folder, name);
        
            % 覆盖写回（保持压缩）
            niftiwrite(data_masked, out_prefix, info, 'Compressed', true);
        
            fprintf('Masked and overwritten: %s\n', files_in{i});
        end
        
        %% -------------03 ChiSep & SEPIA: QSM, ChiDia, ChiPara-------------
        sub_and_ses_str_sepia = sprintf('sub-%s_ses-%s', subject_id, session_id);
        qsm_output_dir = fullfile(qsm_bids_derivatives_dir, 'QSM_reconstruction');
        qsm_output_basename = fullfile(qsm_bids_derivatives_dir, 'QSM_reconstruction', sub_and_ses_str_sepia);
        
        % -------------------------01 SEPIA: phase unwrap--------------------------
        input = struct();
        input(1).name = phase_smooth ;
        input(2).name = mag_smooth ;
        input(3).name = '' ;
        input(4).name = sepia_header ;
        output_basename = qsm_output_basename ;
        mask_filename = [brain_mask_out] ;
        
        % General algorithm parameters
        algorParam = struct();
        algorParam.general.isBET = 0 ;
        algorParam.general.fractional_threshold = 0.5 ;
        algorParam.general.gradient_threshold = 0 ;
        algorParam.general.isInvert = 0 ;
        algorParam.general.isRefineBrainMask = 0 ;
        % Total field recovery algorithm parameters
        algorParam.unwrap.echoCombMethod = 'ROMEO total field calculation' ;
        algorParam.unwrap.offsetCorrect = 'On' ;
        algorParam.unwrap.mask = 'SEPIA mask' ;
        algorParam.unwrap.qualitymaskThreshold = 0.5 ;
        algorParam.unwrap.useRomeoMask = 0 ;
        algorParam.unwrap.isEddyCorrect = 0 ;
        algorParam.unwrap.isSaveUnwrappedEcho = 1 ;
        
        sepiaIO(input,output_basename,mask_filename,algorParam);
        
        % calculate weighted average of unwrapped phase (use it rather than fieldmap)
        load(sepia_header);
        unwrapped_phase_raw = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_part-phase_unwrapped.nii.gz', subject_id, session_id));
        unwrapped_phase_raw_data = load_nii_img_only(unwrapped_phase_raw);
        
        % copy brain mask
        brain_mask = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_mask_brain.nii.gz', subject_id, session_id));
        copyfile(brain_mask_out, brain_mask);
        
        brain_mask_data = load_nii_img_only(brain_mask);
        
        localfied_mask = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_mask_localfield.nii.gz', subject_id, session_id));
        localfied_mask_data = load_nii_img_only(localfied_mask);
        
        TE_s = TE;
        t2s_roi = 0.04;
        W = (TE_s).*exp(-(TE_s)/t2s_roi);
        weightedSum = 0;
        TE_eff = 0;
        
        for echo = 1:size(unwrapped_phase_raw_data,4)
            weightedSum = weightedSum + W(echo)*unwrapped_phase_raw_data(:,:,:,echo)./sum(W);
            TE_eff = TE_eff + W(echo)*TE_s(echo)./sum(W);
        end
        
        unwrapped_phase_average_data = weightedSum / TE_eff * (TE_s(2)-TE_s(1)) .* brain_mask_data;
        unwrapped_phase_average = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_part-phase_unwrapped_weighted_average.nii.gz', subject_id, session_id));
        save_nii_img_only(unwrapped_phase_raw, unwrapped_phase_average, unwrapped_phase_average_data);
        
        % ------------------------02 backgroud field remove------------------------
        info = niftiinfo(unwrapped_phase_average);
        voxelSize = info.PixelDimensions;
        [local_field_data, brain_mask_new_data]=V_SHARP(unwrapped_phase_average_data, localfied_mask_data,'voxelsize', voxelSize,'smvsize', 12);
        local_field_hz_data = double(local_field_data) / (2*pi*delta_TE);
        
        local_field = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_localfield.nii.gz', subject_id, session_id));
        local_field_hz = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_localfield_hz.nii.gz', subject_id, session_id));
        mask_qsm = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_mask_QSM.nii.gz', subject_id, session_id));
        save_nii_img_only(unwrapped_phase_raw, local_field, local_field_data);
        save_nii_img_only(unwrapped_phase_raw, local_field_hz, local_field_hz_data);
        save_nii_img_only(unwrapped_phase_raw, mask_qsm, brain_mask_new_data);
        
        % -----------------------------03 raw QSM----------------------------------
        pad_size = [12, 12, 12];
        voxelSize_QSM = double([voxelSize(:).']); 
        QSM_data = QSM_iLSQR( ...
            local_field_data, ...
            brain_mask_new_data, ...
            'TE',        delta_TE*1e3, ...
            'B0',        B0, ...
            'H',         [0, 0, 1], ...
            'padsize',   pad_size, ...
            'voxelsize', voxelSize_QSM ...
        );
        
        QSM = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_desc-raw_Chimap.nii.gz', subject_id, session_id));
        save_nii_img_only(unwrapped_phase_raw, QSM, QSM_data);
        
        % -----------------------------04 ChiSep-----------------------------------
        RunOptions = struct();
        RunOptions.HaveR2Prime = 0;
        RunOptions.is_scaling = 0;
        RunOptions.scaling_factor = 0.19;
        RunOptions.OutputPath = qsm_output_dir;
        RunOptions.is_scaling = 0;
        RunOptions.scaling_factor = 0.19;
        RunOptions.interp_method = 'sinc';
        RunOptions.sinc_window_size = 15;
        RunOptions.sinc_window_type = 'hann';
        RunOptions.tukey_strength = 0.5;
        RunOptions.tukey_pad = 0.1;
        RunOptions.InputType = 'nifti';
        
        Data = struct();
        Data.RunOptions = RunOptions;
        
        Data.local_field_hz = rot90(double(load_nii_img_only(local_field_hz)));
        Data.R2s = rot90(double(load_nii_img_only(r2s)));
        Data.map = Data.R2s;
        Data.map(Data.map < 0) = 0;
        Data.mask_brain_new = rot90(double(load_nii_img_only(mask_qsm)));
        Data.B0dir = [0, 0, 1];
        Data.CF = B0 * 42.58e+06;
        Data.B0_strength = B0;
        Data.TE = (TE*1000)';
        Data.VoxelSize = voxelSize_QSM([2,1,3]);
        Data.MGRE_Mag_Tukey = rot90(double(load_nii_img_only(mag_smooth)));
        Data.MGRE_Phs_Tukey = rot90(double(load_nii_img_only(phase_smooth)));
        Data.output_root = RunOptions.OutputPath;
        Data.nifti_template = load_untouch_nii(mag_smooth);
        Data.mask_CSF = extract_CSF(Data.R2s, Data.mask_brain_new, Data.VoxelSize);
        
        % Force even dimension
        input_field = {'MGRE_Mag_Tukey','MGRE_Phs_Tukey'};
        for i = 1:length(input_field)
            [Data.(cell2mat(input_field(i))),x_odd,y_odd,z_odd] = even_pad(Data.(cell2mat(input_field(i))));
        end
        RunOptions.EvenSizePadding = [x_odd,y_odd,z_odd];
        Data.MatrixSize = size(Data.MGRE_Mag_Tukey);
        
        Dr = 114;
        % Use the resolution generalization pipeline. Resolution of input data is retained in the resulting chi-separation maps
        [Data.x_para, Data.x_dia, Data.x_tot, Data.qsm_map, Data.r2p_map] = chi_sepnet_general_new_wResolGen(chisep_path, Data.local_field_hz, Data.map, Data.mask_brain_new, Dr, ...
            Data.B0dir, Data.CF, Data.VoxelSize, RunOptions.HaveR2Prime, Data.B0_strength, RunOptions.is_scaling, RunOptions.scaling_factor, RunOptions.interp_method, RunOptions.sinc_window_size, RunOptions.sinc_window_type);
        
        tukey_strength = RunOptions.tukey_strength;
        tukey_pad = RunOptions.tukey_pad;
        Data.x_para = real(tukey_windowing(Data.x_para,tukey_strength,round(size(Data.x_para).*tukey_pad))) .* Data.mask_brain_new;
        Data.x_dia = real(tukey_windowing(Data.x_dia,tukey_strength,round(size(Data.x_dia).*tukey_pad))) .* Data.mask_brain_new;
        Data.x_tot = real(tukey_windowing(Data.x_tot,tukey_strength,round(size(Data.x_tot).*tukey_pad))) .* Data.mask_brain_new;
        Data.qsm_map = real(tukey_windowing(Data.qsm_map,tukey_strength,round(size(Data.qsm_map).*tukey_pad))) .* Data.mask_brain_new;
        Data.r2p_map = real(tukey_windowing(Data.r2p_map,tukey_strength,round(size(Data.r2p_map).*tukey_pad))) .* Data.mask_brain_new;
        
        Data.x_para(Data.x_para < 0) = 0;
        Data.x_dia(Data.x_dia < 0) = 0;
        Data.r2p_map(Data.r2p_map < 0) = 0;
        
        % % vessel seg
        [Data.vesselMask_para, Data.vesselMask_dia] = vesselSegmentation_Chiseparation_DL(chisep_path, Data.x_para, Data.x_dia, Data.mask_brain_new, Data.VoxelSize);
        % % Params for vessel enhancement filter (MFAT, Default)
        % params.tau = 0.02; params.tau2 = 0.35; params.D = 0.3;
        % params.spacing = Data.VoxelSize;
        % params.scales = 4; params.sigmas = [0.25,0.5,0.75,1];
        % params.whiteondark = true;
        % 
        % % params for Seed Generation
        % params.alpha = 2; % Threshold for large vessel seeds
        % params.beta = 1; % Threshold for small vessel seeds
        % params.mipSlice = round(16 / params.spacing(3) / 2) * 2;
        % params.overlap = params.mipSlice / 2;
        %     
        % % params for Region Growing
        % params.limit = [0.5, -0.5]; %% gamma1 and gamma2
        % params.Aniso_Thresh = 0.0012;
        % params.similarity = 0.5; % see (Eq. 3)
        % 
        % seedInput.img1 = Data.R2s; seedInput.img2 = Data.x_para .* Data.x_dia;
        % baseInput.img1 = Data.x_para; baseInput.img2 = Data.x_dia;
        % 
        % [paraMask_init, diaMask_init, homogeneityMeasure_p, homogeneityMeasure_d] = ...
        %                     vesselSegmentation_Chiseparation(seedInput, baseInput, Data.mask_brain_new, min(Data.mask_brain_new, 1 - Data.mask_CSF), params);
        % Data.vesselMask_para = filterVesselsByAnisotropy(paraMask_init, homogeneityMeasure_p, params.Aniso_Thresh);
        % Data.vesselMask_dia  = filterVesselsByAnisotropy(diaMask_init, homogeneityMeasure_d, params.Aniso_Thresh);
        
        % save data
        if ~(sum(RunOptions.EvenSizePadding) == 0)
            input_field = {'x_para', 'x_dia', 'x_tot','qsm_map','R2p','UnwrappedPhase','mask_brain_new'};
            %input_field = {'x_para', 'x_dia', 'x_tot','qsm_map','R2p','UnwrappedPhase','mask_brain_new','vesselMask_para','vesselMask_dia'};
            for i = 1:length(input_field)
                if isfield(Data,cell2mat(input_field(i)))
                    [Data.(cell2mat(input_field(i)))] = even_unpad(Data.(cell2mat(input_field(i))),RunOptions.EvenSizePadding);
                end
            end
        end
        
        SaveData_Chisep(Data, RunOptions)
        
        chidia_old = fullfile(qsm_output_dir, 'ChiDia.nii');
        chipara_old = fullfile(qsm_output_dir, 'ChiPara.nii');
        chitotal_old = fullfile(qsm_output_dir, 'ChiTot.nii');
        chimap_old = fullfile(qsm_output_dir, 'QSM_map.nii');
        vesseldia_old = fullfile(qsm_output_dir, 'vesselMask_dia.nii');
        vesselpara_old = fullfile(qsm_output_dir, 'vesselMask_para.nii');
        
        chidia = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_ChiDia.nii.gz', subject_id, session_id));
        chipara = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_ChiPara.nii.gz', subject_id, session_id));
        chitotal = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_ChiTotal.nii.gz', subject_id, session_id));
        chimap = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_desc-QSMnet_Chimap.nii.gz', subject_id, session_id));
        vesseldia = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_label-VesselDia_mask.nii.gz', subject_id, session_id));
        vesselpara = fullfile(qsm_output_dir, sprintf('sub-%s_ses-%s_label-VesselPara_mask.nii.gz', subject_id, session_id));
        
        convert_nii_to_gz(chidia_old,   chidia);
        convert_nii_to_gz(chipara_old,  chipara);
        convert_nii_to_gz(chitotal_old, chitotal);
        convert_nii_to_gz(chimap_old,   chimap);
        convert_nii_to_gz(vesseldia_old,   vesseldia);
        convert_nii_to_gz(vesselpara_old,   vesselpara);
        
        delete(chidia_old);
        delete(chipara_old);
        delete(chitotal_old);
        delete(chimap_old);
        delete(vesseldia_old);
        delete(vesselpara_old);
        
        % all done !
    end
end

%% ===== Helpers =====
function subs = find_bids_subjects(bids_root_dir)
    d = dir(fullfile(bids_root_dir,'sub-*'));
    d = d([d.isdir]);
    names = {d.name};
    subs = cellfun(@(s) erase(s,'sub-'), names, 'UniformOutput', false);
end

function ses = find_bids_sessions(bids_root_dir, subject_id)
    d = dir(fullfile(bids_root_dir, ['sub-' subject_id], 'ses-*'));
    d = d([d.isdir]);
    names = {d.name};
    ses = cellfun(@(s) erase(s,'ses-'), names, 'UniformOutput', false);
end
