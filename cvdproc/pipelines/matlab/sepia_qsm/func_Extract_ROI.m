% use w_ExtractROITC.m in DPABI

function func_Extract_ROI(ImgFiles, MaskFile, MaskInfoFile, ExtractType, OutputCsv)

    OutputFile = '.\ROI_temp.txt';
    w_ExtractROITC(ImgFiles, MaskFile, ExtractType, OutputFile);

    % ��ȡMaskInfo����id��
    MaskInfo = readtable(MaskInfoFile);
    idCol = MaskInfo.id;  % ����������Ϊ 'id'

    % ��ȡ�����ROI����
    outputData = dlmread(OutputFile);

    % ��ȡROI��ǩ������
    roiLabels = outputData(1, :);  % �����ROI��ǩ
    ROIData = outputData(2:end, :);  % ROI����

    % ת��ROIData��ʹÿ�ж�Ӧһ��ͼ���ļ�
    ROIData = ROIData';

    for i = 1:length(ImgFiles)
        [~, columnName, ~] = fileparts(ImgFiles{i});
        
        % �� columnName ת��Ϊ��Ч�ı��������
        columnName = matlab.lang.makeValidName(columnName);
        
        MaskInfo.(columnName) = nan(height(MaskInfo), 1);

        % ���� ROI ��ǩ�� MaskInfo.id �е�ƥ��λ��
        for j = 1:length(roiLabels)
            labelIndex = find(idCol == roiLabels(j));
            if ~isempty(labelIndex)
                % ��ROI���������Ӧ����
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

