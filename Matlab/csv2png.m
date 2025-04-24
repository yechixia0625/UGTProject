% 指定顶级文件夹路径
folderPath = '/users/coa22cy/UGTProject/Dataset_IID';

% 调用函数处理顶级文件夹
processFolder(folderPath);

% 创建一个递归函数来处理文件夹
function processFolder(folderPath)
    % 获取文件夹中所有文件和子文件夹
    clientFolders = dir(folderPath);
    clientFolders = clientFolders([clientFolders.isdir]); % 只保留文件夹

    % 遍历所有客户端文件夹
    for k = 1:length(clientFolders)
        if startsWith(clientFolders(k).name, '.') % 跳过.和..等系统文件夹
            continue;
        end

        % 获取当前客户端文件夹路径
        clientFolderPath = fullfile(clientFolders(k).folder, clientFolders(k).name);
        labelFolders = dir(clientFolderPath);
        labelFolders = labelFolders([labelFolders.isdir]); % 只保留文件夹

        % 遍历所有标签文件夹
        for j = 1:length(labelFolders)
            if startsWith(labelFolders(j).name, '.')
                continue;
            end

            % 获取当前标签文件夹路径
            labelFolderPath = fullfile(labelFolders(j).folder, labelFolders(j).name);
            maxMinFile = fullfile(labelFolderPath, 'max_min_values_label.csv'); % 最大最小值文件路径

            % 读取最大最小值数据
            maxMinData = readtable(maxMinFile);
            maxValues = maxMinData.Max;
            minValues = maxMinData.Min;

            % 获取当前标签文件夹下的所有CSV文件
            csvFiles = dir(fullfile(labelFolderPath, '*.csv'));

            % 遍历当前标签文件夹下的所有CSV文件
            for i = 1:length(csvFiles)
                if strcmp(csvFiles(i).name, 'max_min_values_label.csv')
                    continue; % 跳过最大最小值文件本身
                end

                % 获取完整的文件路径
                currentPath = fullfile(csvFiles(i).folder, csvFiles(i).name);

                % 读取CSV文件
                data = readmatrix(currentPath);

                % 归一化数据
                normalizedData = 255 * bsxfun(@rdivide, bsxfun(@minus, data, minValues'), (maxValues - minValues)');

                % 构建PNG文件名
                pngFileName = fullfile(csvFiles(i).folder, [csvFiles(i).name(1:end-4) '.png']);

                % 保存为PNG文件
                imwrite(uint8(normalizedData), pngFileName);
                fprintf('Converted %s to %s\n', currentPath, pngFileName);

                % 删除原CSV文件
                delete(currentPath);
                fprintf('Deleted %s\n', currentPath);
            end
        end
    end
end
