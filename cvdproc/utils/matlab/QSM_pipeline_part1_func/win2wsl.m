function wsl_path = win2wsl(win_path)
    % �� \ ���� /
    wsl_path = strrep(win_path, '\', '/');
    % ��ȡ�̷� (C:, D:, ��)
    drive_letter = lower(wsl_path(1));
    % ȥ�� "X:/"
    wsl_path = ['/mnt/' drive_letter wsl_path(3:end)];
end
