% Load bvecs and bvals
bvecs = load('E:\Neuroimage\demo_data\TWD_AF\derivatives\qsiprep_20241128\sub-AF0237\ses-01\dwi\sub-AF0237_ses-01_acq-b4000_space-ACPC_desc-preproc_dwi.bvec');
bvals = load('E:\Neuroimage\demo_data\TWD_AF\derivatives\qsiprep_20241128\sub-AF0237\ses-01\dwi\sub-AF0237_ses-01_acq-b4000_space-ACPC_desc-preproc_dwi.bval');

% Plot b-values distribution
figure;
histogram(bvals, 'BinMethod', 'integers', 'FaceColor', 'blue', 'EdgeColor', 'black');
xlabel('b-values');
ylabel('Frequency');
title('b-value Distribution');
grid on;

% Check if b-values are shelled
unique_bvals = unique(bvals); % Get unique b-values
bval_threshold = 50; % Define threshold to group b-values into shells

% Check if the b-values are shelled
shelled_check = true;
for i = 1:length(unique_bvals) - 1
    if abs(unique_bvals(i+1) - unique_bvals(i)) > bval_threshold
        shelled_check = false;
        break;
    end
end

% Display results
if shelled_check
    disp('The b-values are shelled.');
else
    disp('The b-values are not shelled.');
end

% Optional: Visualize shells (if shelled)
if shelled_check
    figure;
    scatter3(bvecs(1, :) .* sqrt(bvals), bvecs(2, :) .* sqrt(bvals), bvecs(3, :) .* sqrt(bvals), 50, bvals, 'filled');
    colormap(jet);
    colorbar;
    xlabel('x * sqrt(b-value)');
    ylabel('y * sqrt(b-value)');
    zlabel('z * sqrt(b-value)');
    title('b-vector Shell Visualization');
    grid on;
end
