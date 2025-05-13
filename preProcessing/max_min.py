import os
import pandas as pd

base_dir = 'Dataset_nonIID_simplex'
clients = ['local_1', 'local_2', 'local_3', 'local_4', 'local_5', 'local_6']
labels = [0, 1, 2, 3]

for client in clients:
    client_path = os.path.join(base_dir, client)
    for label in labels:
        label_path = os.path.join(client_path, str(label))
        max_values = None
        min_values = None

        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            if file.endswith('.csv'):
                data = pd.read_csv(file_path, header=None)

                if max_values is None:
                    max_values = data.max()
                    min_values = data.min()
                else:
                    max_values = pd.DataFrame([max_values, data.max()]).max()
                    min_values = pd.DataFrame([min_values, data.min()]).min()

        max_min_values = pd.DataFrame({'Max': max_values, 'Min': min_values})
        max_min_values.to_csv(os.path.join(label_path, f'max_min_values_label.csv'), index_label='Column')
