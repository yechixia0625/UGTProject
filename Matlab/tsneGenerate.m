clear; clc;

% Define the root directory for data storage
data_root = '/users/coa22cy/UGTProject/Matlab/Dataset_IID2';

% Define clients and labels
clients = {'local_1', 'local_2', 'local_3', 'local_4', 'local_5', 'local_6'};
labels = {'0', '1', '2', '3'};

% Initialize arrays to store features and labels for all clients
client_features = cell(length(clients), 1);
client_labels = cell(length(clients), 1);
all_features = [];
all_labels = [];
all_clients = [];

noise_client_index = 1;

% Iterate over each client
for c = 1:length(clients)
    client = clients{c};
    client_path = fullfile(data_root, client);
    
    % Initialize features and labels for the current client
    features = [];
    labels_client = [];
    
    % Iterate over each label directory
    for l = 1:length(labels)
        label = labels{l};
        label_path = fullfile(client_path, label);
        
        features = [];
        labels_client = [];

        % Process each CSV file
        for f = 1:length(csv_files)
            label = labels{f};
            file_path = fullfile(label_path, file_name);

            % Read data from the CSV file
            data = csvread(file_path);
            
            % Check the dimensions of the data
            [num_rows, num_cols] = size(data);
            if num_rows ~= 160 || num_cols ~= 20
                warning('File %s has dimensions other than 160x20, skipped.', file_path);
                continue;
            end

            if c == noise_client_index
                % 定义噪声强度（可以根据需要调整）
                noise_level = 0.1;  % 噪声强度
                % 生成与数据相同维度的随机噪声
                noise = noise_level * randn(size(data));
                % 将噪声添加到数据中
                data = data + noise;
            end

            % Initialize an array to store features from the current file
            sample_features = [];
            % Process each column of data
            for s = 1:num_cols
                sensor_data = data(:, s);
                
                % Use FFT to find the dominant frequency
                Fs = 1;  % Sampling frequency, modify if needed
                L = length(sensor_data);
                Y = fft(sensor_data);
                P2 = abs(Y/L);
                P1 = P2(1:floor(L/2)+1);
                P1(2:end-1) = 2*P1(2:end-1);
                f_freq = Fs*(0:(L/2))/L;
                [~, idx] = max(P1);
                dominant_freq = f_freq(idx);
                
                % Find the maximum value of sensor data
                max_value = max(sensor_data);
                
                % Find the minimum value of sensor data
                min_value = min(sensor_data);
                
                % Calculate the maximum gradient between adjacent data points
                gradients = abs(diff(sensor_data));
                max_gradient = max(gradients);
                
                % Aggregate features
                sample_features = [sample_features, dominant_freq, max_value, min_value, max_gradient];
            end
            
            % Add features and corresponding labels to the current client's array
            features = [features; sample_features];
            labels_client = [labels_client; str2double(label)];
        end
    end
    
    % Store current client's features and labels in global arrays
    client_features{c} = features;
    client_labels{c} = labels_client;
    
    % Add data to the overall feature and label arrays
    all_features = [all_features; features];
    all_labels = [all_labels; labels_client];
    all_clients = [all_clients; c * ones(size(labels_client))];
end

all_features = zscore(all_features);
for c = 1:length(clients)
    client_features{c} = zscore(client_features{c});
end

% Set t-SNE parameters
perplexity = 50;

% Create a new figure window
figure;

% Calculate the number of subplots needed
num_clients = length(clients);
num_subplots = num_clients + 1;  % Including the overall t-SNE

% Determine the number of rows and columns for subplots based on the number of clients
num_rows = ceil(sqrt(num_subplots));
num_cols = ceil(num_subplots / num_rows);

% Perform t-SNE for each client and plot
for c = 1:num_clients
    features = client_features{c};
    labels = client_labels{c};
    mappedX = tsne(features, 'Perplexity', perplexity);
    
    % Plot in subplot
    subplot(num_rows, num_cols, c);
    gscatter(mappedX(:,1), mappedX(:,2), labels);
    title(['t-SNE - ' clients{c}]);
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    legend('Label 0', 'Label 1', 'Label 2', 'Label 3');
    grid on;
end

% Perform t-SNE on all data
mappedX_all = tsne(all_features, 'Perplexity', perplexity);

% Plot the overall t-SNE in subplot, color by client
subplot(num_rows, num_cols, num_subplots);
gscatter(mappedX_all(:,1), mappedX_all(:,2), all_clients);
title('Overall t-SNE Visualization (by Client)');
xlabel('Dimension 1');
ylabel('Dimension 2');
legend(clients);
grid on;
