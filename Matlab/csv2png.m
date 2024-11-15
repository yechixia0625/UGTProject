% 指定顶级文件夹路径
folderPath = '/Users/yechixia/UGTProject/Matlab/Dataset_IID/';

% 调用函数处理顶级文件夹
processFolder(folderPath);

% 创建一个递归函数来处理文件夹
function processFolder(folderPath)
    % 获取文件夹中所有文件和子文件夹
    fileList = dir(fullfile(folderPath, '**/*.csv')); % 递归获取所有CSV文件的路径

    % 遍历文件列表
    for i = 1:length(fileList)
        % 获取完整的文件路径
        currentPath = fullfile(fileList(i).folder, fileList(i).name);

        % 读取CSV文件
        data = readmatrix(currentPath);

        % 数据归一化到0-255
        data_normalized = uint8(255 * mat2gray(data));

        % 构建PNG文件名
        pngFileName = fullfile(fileList(i).folder, [fileList(i).name(1:end-4) '.png']);

        % 保存为PNG文件
        imwrite(data_normalized, pngFileName);
        fprintf('Converted %s to %s\n', currentPath, pngFileName);

        % 删除原CSV文件
        delete(currentPath);
        fprintf('Deleted %s\n', currentPath);
    end
end
