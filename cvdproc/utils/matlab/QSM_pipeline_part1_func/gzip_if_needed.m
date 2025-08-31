function gzip_if_needed(niiPath)
    if endsWith(lower(niiPath), '.nii')
        if isfile([niiPath '.gz']), delete([niiPath '.gz']); end
        gzip(niiPath);
    end
end