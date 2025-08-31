function j = replace_ext(pathIn, newExt)
    if endsWith(lower(pathIn), '.nii.gz')
        j = regexprep(pathIn, '\.nii\.gz$', newExt, 'ignorecase');
    elseif endsWith(lower(pathIn), '.nii')
        j = regexprep(pathIn, '\.nii$', newExt, 'ignorecase');
    else
        j = [pathIn newExt];
    end
end