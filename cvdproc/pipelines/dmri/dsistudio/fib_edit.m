file_name = 'E:\Neuroimage\TestDataSet\BIDS_TestDataSet\derivatives\qsiprep\sub-Patient0238\dwi\sub-Patient0238_acq-b4000_space-T1w_desc-preproc_dwi.nii.gz.src.gz.odf.gqi.1.25.fib.gz';

gunzip(file_name);
[pathstr, name, ext] = fileparts(file_name);
movefile(fullfile(pathstr, name),strcat(fullfile(pathstr, name),'.mat'));
fib = load(strcat(fullfile(pathstr, name),'.mat'));

max_fib = 0;
for i = 1:10
    if isfield(fib,strcat('fa',int2str(i-1)))
        max_fib = i;
    else
        break;
    end
end

fa = zeros([fib.dimension max_fib]);
index = zeros([fib.dimension max_fib]);

for i = 1:max_fib
    eval(strcat('fa(:,:,:,i) = reshape(fib.fa',int2str(i-1),',fib.dimension);'));
    eval(strcat('index(:,:,:,i) = reshape(fib.index',int2str(i-1),',fib.dimension);'));
end

odf_vertices = fib.odf_vertices;
odf_faces = fib.odf_faces;

