%% Prepare the inputs
clear;
clc;

%bids_root_dir = '/mnt/f/BIDS/SVD_demo';
%subject_id = 'SSI0310';
%session_id = 'baseline';
%output_dir = '/mnt/f/BIDS/SVD_demo/derivatives/ExploreASL/sub-SSI0310/ses-baseline';
%
%t1w_filter_filename = 'sub-SSI0310_ses-baseline_acq-highres_T1w';
%asl_filter_filename = 'sub-SSI0310_ses-baseline';
%ExploreASL_dir = '/path/to/ExploreASL';

bids_root_dir = '/this/is/for/nipype/bids_root_dir';
subject_id = '/this/is/for/nipype/subject_id';
session_id = '/this/is/for/nipype/session_id';
output_dir = '/this/is/for/nipype/output_dir';

t1w_filter_filename = '/this/is/for/nipype/t1w_filter_filename';
asl_filter_filename = '/this/is/for/nipype/asl_filter_filename';

ExploreASL_dir = '/this/is/for/nipype/exploreasl_dir';

author_name = 'Temporary Author';

anat_dir = fullfile(bids_root_dir, ['sub-' subject_id], ['ses-' session_id], 'anat');
perf_dir = fullfile(bids_root_dir, ['sub-' subject_id], ['ses-' session_id], 'perf');

rawdata_dir = fullfile(output_dir, 'rawdata');
target_anat_dir = fullfile(rawdata_dir, ['sub-' subject_id], ['ses-' session_id], 'anat');
target_perf_dir = fullfile(rawdata_dir, ['sub-' subject_id], ['ses-' session_id], 'perf');

dirs_to_make = {
    output_dir
    rawdata_dir
    target_anat_dir
    target_perf_dir
};

for i = 1:numel(dirs_to_make)
    this_dir = dirs_to_make{i};
    ensure_dir_exists(this_dir);
end

fprintf('Created directories successfully.\n');
fprintf('Output dir: %s\n', output_dir);
fprintf('Rawdata dir: %s\n', rawdata_dir);

anat_copy_count = 0;
if exist(anat_dir, 'dir')
    anat_files = dir(anat_dir);
    for i = 1:numel(anat_files)
        if anat_files(i).isdir
            continue;
        end

        fname = anat_files(i).name;
        if contains(fname, t1w_filter_filename)
            src_file = fullfile(anat_dir, fname);
            dst_file = fullfile(target_anat_dir, fname);
            safe_copy_file(src_file, dst_file, true);
            anat_copy_count = anat_copy_count + 1;
        end
    end
else
    warning('Anat directory does not exist: %s', anat_dir);
end

perf_copy_count = 0;
if exist(perf_dir, 'dir')
    perf_files = dir(perf_dir);
    for i = 1:numel(perf_files)
        if perf_files(i).isdir
            continue;
        end

        fname = perf_files(i).name;
        if contains(fname, asl_filter_filename)
            src_file = fullfile(perf_dir, fname);
            dst_file = fullfile(target_perf_dir, fname);
            safe_copy_file(src_file, dst_file, true);
            perf_copy_count = perf_copy_count + 1;
        end
    end
else
    warning('Perf directory does not exist: %s', perf_dir);
end

fprintf('Copied %d anat file(s).\n', anat_copy_count);
fprintf('Copied %d perf file(s).\n', perf_copy_count);

studypar_path = fullfile(output_dir, 'studyPar.json');
datapar_path = fullfile(output_dir, 'dataPar.json');
dataset_description_path = fullfile(rawdata_dir, 'dataset_description.json');

studyPar = struct();
studyPar.Authors = author_name;

dataPar = struct();
dataPar.x = struct();
dataPar.x.subjectRegexp = ['sub-' subject_id];

dataPar.x.S = struct();
dataPar.x.S.Atlases = { ...
    'Total', ...
    'AAL3v1', ...
    'DeepWM', ...
    'WholeBrain', ...
    'Tatu_ACA_MCA_PCA' ...
};
dataPar.x.S.TissueMasking = { ...
    'WB', ...
    'GM', ...
    'WM', ...
    'WB', ...
    'GM' ...
};

dataPar.x.settings = struct();
dataPar.x.settings.Quality = 1;
dataPar.x.settings.DELETETEMP = 0;
dataPar.x.settings.SkipIfNoASL = 1;
dataPar.x.settings.SkipIfNoM0 = 0;
dataPar.x.settings.stopAfterErrors = 5;

dataPar.x.modules = struct();
dataPar.x.modules.population = struct();
dataPar.x.modules.population.bNativeSpaceAnalysis = true;
dataPar.x.modules.asl = struct();
dataPar.x.modules.asl.bInferArtBASIL = false;

dataset_description = struct();
dataset_description.Name = '';
dataset_description.BIDSVersion = 'v1.9.0';
dataset_description.License = '';
dataset_description.Authors = {''};
dataset_description.Acknowledgments = '';
dataset_description.HowToAcknowledge = '';
dataset_description.Funding = {''};
dataset_description.ReferencesAndLinks = {''};
dataset_description.DatasetDOI = '';

write_json(studypar_path, studyPar);
write_json(datapar_path, dataPar);
write_json(dataset_description_path, dataset_description);

fprintf('JSON files written successfully.\n');
fprintf('studyPar.json: %s\n', studypar_path);
fprintf('dataPar.json: %s\n', datapar_path);
fprintf('dataset_description.json: %s\n', dataset_description_path);

cd(ExploreASL_dir);
x = ExploreASL(output_dir, 0, [1 1 1]); %#ok<NASGU>

%% Post-process
raw_output_dir = fullfile(output_dir, 'derivatives', 'ExploreASL', ['sub-' subject_id, '_', session_id]);
raw_output_dir_population = fullfile(output_dir, 'derivatives', 'ExploreASL', 'Population');

target_output_dir = output_dir;
target_subject_dir = fullfile(target_output_dir, ['sub-' subject_id, '_', session_id]);
target_output_dir_population = fullfile(output_dir, 'Population');

% 1. Copy subject/session output as a whole folder
if exist(raw_output_dir, 'dir')
    fprintf('[INFO] Copying subject output folder...\n');
    safe_copy_dir(raw_output_dir, target_subject_dir, true);

    if try_remove_dir(raw_output_dir)
        fprintf('[INFO] Removed source subject directory: %s\n', raw_output_dir);
    else
        fprintf('[WARN] Could not remove source subject directory: %s\n', raw_output_dir);
    end
else
    fprintf('[WARN] Subject output not found: %s\n', raw_output_dir);
end

% 2. Copy population output contents
if exist(raw_output_dir_population, 'dir')
    fprintf('[INFO] Copying population output contents...\n');
    ensure_dir_exists(target_output_dir_population);

    files = dir(raw_output_dir_population);
    for i = 1:length(files)
        name = files(i).name;

        if strcmp(name, '.') || strcmp(name, '..')
            continue;
        end

        src = fullfile(raw_output_dir_population, name);
        dst = fullfile(target_output_dir_population, name);

        if files(i).isdir
            safe_copy_dir(src, dst, true);
        else
            safe_copy_file(src, dst, true);
        end
    end

    if try_remove_dir(raw_output_dir_population)
        fprintf('[INFO] Removed source population directory: %s\n', raw_output_dir_population);
    else
        fprintf('[WARN] Could not remove source population directory: %s\n', raw_output_dir_population);
    end
else
    fprintf('[WARN] Population folder not found: %s\n', raw_output_dir_population);
end

% 3. Clean unnecessary folders/files
fprintf('[INFO] Cleaning unnecessary files...\n');

delete_targets = {
    fullfile(output_dir, 'derivatives')
    fullfile(output_dir, 'rawdata')
    fullfile(output_dir, 'dataPar.json')
    fullfile(output_dir, 'studyPar.json')
};

for i = 1:length(delete_targets)
    path_i = delete_targets{i};

    if exist(path_i, 'dir')
        fprintf('[INFO] Removing directory: %s\n', path_i);
        if ~try_remove_dir(path_i)
            fprintf('[WARN] Failed to remove directory: %s\n', path_i);
        end

    elseif exist(path_i, 'file')
        fprintf('[INFO] Removing file: %s\n', path_i);
        if ~try_remove_file(path_i)
            fprintf('[WARN] Failed to remove file: %s\n', path_i);
        end

    else
        fprintf('[SKIP] Not found: %s\n', path_i);
    end
end

fprintf('[DONE] ExploreASL outputs reorganized successfully.\n');

%% Local helper functions

function ensure_dir_exists(dir_path)
    if ~exist(dir_path, 'dir')
        [status, msg, msgid] = mkdir(dir_path);
        if ~status
            error('Failed to create directory: %s\n%s\n%s', dir_path, msgid, msg);
        end
    end
end

function write_json(json_path, s)
    json_text = jsonencode(s);
    fid = fopen(json_path, 'w');
    if fid == -1
        error('Cannot open file for writing: %s', json_path);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', json_text);
end

function safe_copy_dir(src_dir, dst_dir, overwrite)
    if nargin < 3
        overwrite = true;
    end

    if ~exist(src_dir, 'dir')
        error('Source directory does not exist: %s', src_dir);
    end

    parent_dir = fileparts(dst_dir);
    ensure_dir_exists(parent_dir);

    if exist(dst_dir, 'dir')
        if overwrite
            if ~try_remove_dir(dst_dir)
                error('Failed to remove existing destination directory: %s', dst_dir);
            end
        else
            fprintf('[SKIP] Destination directory already exists: %s\n', dst_dir);
            return;
        end
    end

    [status, msg, msgid] = copyfile(src_dir, dst_dir, 'f');
    if ~status
        error('Failed to copy directory.\nSource: %s\nDestination: %s\n%s\n%s', ...
            src_dir, dst_dir, msgid, msg);
    end
end

function safe_copy_file(src_file, dst_file, overwrite)
    if nargin < 3
        overwrite = true;
    end

    if ~exist(src_file, 'file')
        error('Source file does not exist: %s', src_file);
    end

    dst_parent = fileparts(dst_file);
    ensure_dir_exists(dst_parent);

    if exist(dst_file, 'file')
        if overwrite
            if ~try_remove_file(dst_file)
                error('Failed to remove existing destination file: %s', dst_file);
            end
        else
            fprintf('[SKIP] Destination file already exists: %s\n', dst_file);
            return;
        end
    end

    [status, msg, msgid] = copyfile(src_file, dst_file, 'f');
    if ~status
        error('Failed to copy file.\nSource: %s\nDestination: %s\n%s\n%s', ...
            src_file, dst_file, msgid, msg);
    end
end

function ok = try_remove_dir(dir_path)
    ok = true;
    if exist(dir_path, 'dir')
        try
            rmdir(dir_path, 's');
        catch ME
            ok = false;
            fprintf('[WARN] rmdir failed for %s\n%s\n', dir_path, ME.message);
        end
    end
end

function ok = try_remove_file(file_path)
    ok = true;
    if exist(file_path, 'file')
        try
            delete(file_path);
        catch ME
            ok = false;
            fprintf('[WARN] delete failed for %s\n%s\n', file_path, ME.message);
        end
    end
end