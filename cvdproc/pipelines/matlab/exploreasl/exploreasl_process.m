%% Prepare the inputs
input_bids_dir = '/this/is/prepared/for/nipype/input_bids_dir';
subject_id = '/this/is/prepared/for/nipype/subject_id';
session_id = '/this/is/prepared/for/nipype/session_id';
use_which_t1w = '/this/is/prepared/for/nipype/use_which_t1w';
use_which_flair = '/this/is/prepared/for/nipype/use_which_flair';

anat_dir = fullfile(input_bids_dir, ['sub-' subject_id], ['ses-' session_id], 'anat');
perf_dir = fullfile(input_bids_dir, ['sub-' subject_id], ['ses-' session_id], 'perf');

rawdata_dir = fullfile(input_bids_dir, 'rawdata');
target_anat_dir = fullfile(rawdata_dir, ['sub-' subject_id], ['ses-' session_id], 'anat');
target_perf_dir = fullfile(rawdata_dir, ['sub-' subject_id], ['ses-' session_id], 'perf');

% Create target directories
if ~exist(target_anat_dir, 'dir')
    mkdir(target_anat_dir);
end
if ~exist(target_perf_dir, 'dir')
    mkdir(target_perf_dir);
end

% Copy anat folder if exists
if exist(anat_dir, 'dir')
    copyfile(anat_dir, target_anat_dir);
end

% Copy perf folder if exists
if exist(perf_dir, 'dir')
    copyfile(perf_dir, target_perf_dir);
end

% Clean anat directory by keeping only the required T1w and FLAIR images
if exist(target_anat_dir, 'dir')
    % List all files
    anat_files = dir(fullfile(target_anat_dir, '*'));

    % Loop through files and delete those not matching the criteria
    for i = 1:length(anat_files)
        file_name = anat_files(i).name;
        file_path = fullfile(target_anat_dir, file_name);

        % Skip '.' and '..'
        if startsWith(file_name, '.')
            continue;
        end

        % If it's a T1w file but does not contain the keyword -> delete
        if contains(file_name, 'T1w') && ~contains(file_name, use_which_t1w)
            delete(file_path);
        end

        % If it's a FLAIR file but does not contain the keyword -> delete
        if contains(file_name, 'FLAIR') && ~contains(file_name, use_which_flair)
            delete(file_path);
        end
    end

    % After deletion, check if folder is now empty
    remaining_files = dir(fullfile(target_anat_dir, '*'));
    if length(remaining_files) <= 2  % Only . and .. remain
        rmdir(target_anat_dir, 's');
        fprintf('Removed empty anat directory: %s\n', target_anat_dir);
    end
end

fprintf('Rawdata folders created at: %s\n', rawdata_dir);

% create the dataPar.json file
dataPar.x.dataset.subjectRegexp = ['^sub-' subject_id '$'];
dataPar.x.dataset.exclusion = '';
dataPar.x.SESSIONS = {['ses-' session_id]};

json_path = fullfile(input_bids_dir, 'dataPar.json');

% Use savejson (requires JSONlab toolbox: https://github.com/fangq/jsonlab)
savejson('', dataPar, json_path);

fprintf('dataPar.json created at: %s\n', json_path);

% Append dataPar.json and rawdata/ to .bidsignore
bidsignore_path = fullfile(input_bids_dir, '.bidsignore');
lines_to_add = {"dataPar.json", "rawdata/"};

% Create .bidsignore if it does not exist
if ~exist(bidsignore_path, 'file')
    fid = fopen(bidsignore_path, 'w');
    fclose(fid);
end

% Read existing contents to avoid duplicates
existing_lines = {};
fid = fopen(bidsignore_path, 'r');
if fid ~= -1
    existing_lines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    existing_lines = existing_lines{1};
end

% Append the new line if it is not already present
fid = fopen(bidsignore_path, 'a');
for i = 1:length(lines_to_add)
    if ~any(strcmp(existing_lines, lines_to_add{i}))
        fprintf(fid, '%s\n', lines_to_add{i});
    end
end
fclose(fid);

fprintf('.bidsignore updated at: %s\n', bidsignore_path);

%% ExploreASL Process
% sMRI process
[x] = ExploreASL(input_bids_dir, 0, [1 0 0]);

% ASL process
[x] = ExploreASL(input_bids_dir, 0, [0 1 0]);

% Population process
[x] = ExploreASL(input_bids_dir, 0, [0 0 1]);