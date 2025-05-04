% use w_ExtractROITC.m in DPABI

function func_Extract_ROI(ImgFiles, MaskFile, MaskInfoFile, ExtractType, OutputCsv)

    OutputFile = '.\ROI_temp.txt';
    w_ExtractROITC(ImgFiles, MaskFile, ExtractType, OutputFile);

    % 读取MaskInfo表格和id列
    MaskInfo = readtable(MaskInfoFile);
    idCol = MaskInfo.id;  % 假设编号列名为 'id'

    % 读取输出的ROI数据
    outputData = dlmread(OutputFile);

    % 提取ROI标签和数据
    roiLabels = outputData(1, :);  % 输出的ROI标签
    ROIData = outputData(2:end, :);  % ROI数据

    % 转置ROIData，使每列对应一个图像文件
    ROIData = ROIData';

    for i = 1:length(ImgFiles)
        [~, columnName, ~] = fileparts(ImgFiles{i});
        
        % 将 columnName 转换为有效的表变量名称
        columnName = matlab.lang.makeValidName(columnName);
        
        MaskInfo.(columnName) = nan(height(MaskInfo), 1);

        % 查找 ROI 标签在 MaskInfo.id 中的匹配位置
        for j = 1:length(roiLabels)
            labelIndex = find(idCol == roiLabels(j));
            if ~isempty(labelIndex)
                % 将ROI数据填入对应的行
                MaskInfo.(columnName)(labelIndex) = ROIData(j, i);
            end
        end
    end

    writetable(MaskInfo, OutputCsv);

    if exist(OutputFile, 'file')
        delete(OutputFile);
    end

    disp('All steps completed. The new CSV file has been saved at:');
    disp(OutputCsv);

end

