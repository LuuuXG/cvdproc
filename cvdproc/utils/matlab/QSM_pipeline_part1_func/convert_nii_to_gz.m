function convert_nii_to_gz(srcNii, dstNiiGz)
    assert(exist(srcNii, 'file')==2, 'Source NIfTI not found: %s', srcNii);

    % Ensure destination folder exists
    [dstFolder, ~, ~] = fileparts(dstNiiGz);
    if ~isempty(dstFolder) && ~exist(dstFolder, 'dir')
        mkdir(dstFolder);
    end

    % Normalize destination: strip a trailing ".gz" so we have "...something.nii"
    dstNoGz = regexprep(dstNiiGz, '\.gz$', '');

    info = niftiinfo(srcNii);
    img  = niftiread(info);

    % Preserve original datatype if possible
    if isfield(info,'Datatype') && ~strcmp(class(img), info.Datatype)
        img = cast(img, info.Datatype);
    end

    % Preferred path: write compressed directly if supported
    try
        % niftiwrite will create "<dstNoGz>.gz" when 'Compressed', true
        niftiwrite(img, dstNoGz, info, 'Compressed', true);

        % If MATLAB created "<dstNoGz>.gz" already, ensure final name is exactly dstNiiGz
        if ~strcmp([dstNoGz '.gz'], dstNiiGz) && exist([dstNoGz '.gz'], 'file')==2
            movefile([dstNoGz '.gz'], dstNiiGz);
        end
    catch
        % Fallback: write uncompressed, then gzip, then rename to exact dstNiiGz
        niftiwrite(img, dstNoGz, info);     % writes "...something.nii"
        gzip(dstNoGz);                       % creates "...something.nii.gz"
        delete(dstNoGz);                     % remove the uncompressed .nii

        % Ensure final name matches exactly
        if ~strcmp([dstNoGz '.gz'], dstNiiGz)
            movefile([dstNoGz '.gz'], dstNiiGz);
        end
    end

    fprintf('Wrote %s\n', dstNiiGz);
end
