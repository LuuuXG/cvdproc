function out = replace_part_token(inPath, fromTok, toTok)
    [p,n,e] = fileparts(inPath);
    if strcmpi(e, '.gz')
        [p,n2,e2] = fileparts(fullfile(p,n));
        n = n2; e = [e2 '.gz'];
    end
    n = strrep(n, fromTok, toTok);
    out = fullfile(p, [n e]);
end