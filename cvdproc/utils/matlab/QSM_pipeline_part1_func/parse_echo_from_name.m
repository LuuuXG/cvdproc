function ek = parse_echo_from_name(fp)
    [~,name,ext] = fileparts(fp);
    if strcmpi(ext, '.gz')
        [~,name] = fileparts(name); % strip .nii
    end
    tok = regexp(name, '[_-]echo-([0-9]+)(?:[_-]|$)', 'tokens', 'once');
    if isempty(tok), ek = ''; else, ek = tok{1}; end
end
