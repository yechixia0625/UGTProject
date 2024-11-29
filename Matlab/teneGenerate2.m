clear; clc;

% Define the root directory where data is stored
data_root = '/users/coa22cy/UGTProject/Matlab/Dataset_IID';

% Define the clients and labels
clients = {'local_1', 'local_2', 'local_3', 'local_4', 'local_5', 'local_6'};
labels = {'0', '1', '2', '3'};

% Initialize arrays to store features and labels for all clients
client_features = cell(length(clients), 1);
client_labels = cell(length(clients), 1);
all_features = [];
all_labels = [];
all_clients = [];

% Specify the index of the client to add noise to (if needed)
noise_client_index = 1;  % This can be modified as needed

% Loop through each client
for c = 1:length(clients)
    client = clients{c};
    client_path = fullfile(data_root, client);
    
    % Initialize features and labels for the current client
    features = [];
    labels_client = [];
    
    % Loop through each label directory
    for l = 1:length(labels)
        label = labels{l};
        label_path = fullfile(client_path, label);
        
        % Get all PNG files in the current label directory
        png_files = dir(fullfile(label_path, '*.png'));

        % Process each PNG file
        for f = 1:length(png_files)
            file_name = png_files(f).name;
            file_path = fullfile(label_path, file_name);

            % Read image data from PNG file
            img = imread(file_path);

            % Convert image to double precision type and normalize to [0, 1] range
            img = im2double(img);

            % Check image dimensions (assuming image is grayscale)
            [num_rows, num_cols, num_channels] = size(img);
            if num_channels ~= 1
                warning('File %s is not a grayscale image, skipping this file.', file_path);
                continue;
            end

            % **Add noise here (if needed)**
            % If the current client is the designated noise client, add random noise to the image
            if c == noise_client_index
                % Define noise intensity (can be adjusted as needed)
                noise_level = 0.3;  % Noise intensity
                % Generate random noise of the same dimensions as the image
                noise = noise_level * randn(size(img));
                % Add noise to the image
                img = img + noise;
                % Clip image to [0, 1] range
                img = max(min(img, 1), 0);
            end

            % Extract image features
            % Here we can simply flatten the image pixels into a feature vector, or extract other features
            % Method 1: Flatten image pixel values into a feature vector
            sample_features = img(:)';

            % Method 2: Use image processing to extract higher-level features (such as texture, shape, etc.)
            % Additional feature extraction methods can be added as needed

            % Add features and corresponding labels to the current client's array
            features = [features; sample_features];
            labels_client = [labels_client; str2double(label)];
        end
    end

    % Store current client's features and labels in global arrays
    client_features{c} = features;
    client_labels{c} = labels_client;

    % Add data to total features and labels arrays
    all_features = [all_features; features];
    all_labels = [all_labels; labels_client];
    all_clients = [all_clients; c * ones(size(labels_client))];
end

% Normalize features (helps with the effectiveness of t-SNE)
all_features = zscore(all_features);
for c = 1:length(clients)
    client_features{c} = zscore(client_features{c});
end

% Set parameters for t-SNE
perplexity = 50;

% Create a new figure window
figure;

% Calculate the number of subplots needed
num_clients = length(clients);
num_subplots = num_clients + 1;  % Includes the overall t-SNE

% Determine the rows and columns of subplots based on the number of clients
num_rows = 3;
num_cols = 2;

% Perform t-SNE and plot for each client
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
