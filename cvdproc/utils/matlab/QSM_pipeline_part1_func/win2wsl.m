function wsl_path = win2wsl(win_path)
    % 把 \ 换成 /
    wsl_path = strrep(win_path, '\', '/');
    % 提取盘符 (C:, D:, …)
    drive_letter = lower(wsl_path(1));
    % 去掉 "X:/"
    wsl_path = ['/mnt/' drive_letter wsl_path(3:end)];
end
