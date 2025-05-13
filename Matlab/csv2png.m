folderPath = '/users/coa22cy/UGTProject/Dataset_nonIID_simplex';

processFolder(folderPath);

function processFolder(folderPath)
    clientFolders = dir(folderPath);
    clientFolders = clientFolders([clientFolders.isdir]);

    for k = 1:length(clientFolders)
        if startsWith(clientFolders(k).name, '.')
            continue;
        end

        clientFolderPath = fullfile(clientFolders(k).folder, clientFolders(k).name);
        labelFolders = dir(clientFolderPath);
        labelFolders = labelFolders([labelFolders.isdir]);

        for j = 1:length(labelFolders)
            if startsWith(labelFolders(j).name, '.')
                continue;
            end

            labelFolderPath = fullfile(labelFolders(j).folder, labelFolders(j).name);
            maxMinFile = fullfile(labelFolderPath, 'max_min_values_label.csv');

            maxMinData = readtable(maxMinFile);
            maxValues = maxMinData.Max;
            minValues = maxMinData.Min;

            csvFiles = dir(fullfile(labelFolderPath, '*.csv'));

            for i = 1:length(csvFiles)
                if strcmp(csvFiles(i).name, 'max_min_values_label.csv')
                    continue;
                end

                currentPath = fullfile(csvFiles(i).folder, csvFiles(i).name);

                data = readmatrix(currentPath);

                normalizedData = 255 * bsxfun(@rdivide, bsxfun(@minus, data, minValues'), (maxValues - minValues)');

                pngFileName = fullfile(csvFiles(i).folder, [csvFiles(i).name(1:end-4) '.png']);

                imwrite(uint8(normalizedData), pngFileName);
                fprintf('Converted %s to %s\n', currentPath, pngFileName);

                delete(currentPath);
                fprintf('Deleted %s\n', currentPath);
            end
        end
    end
end
