% This is a test script for the cvdproc MATLAB pipeline.
% This script will copy a test image to the output floder and smooth it.

input_image = 'placeholder/for/nipype/input_image'
output_dir = 'placeholder/for/nipype/output_dir'

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_image = fullfile(output_dir, 'test_image.nii.gz');
copyfile(input_image, output_image);
fprintf('Test image copied to %s\n', output_image);