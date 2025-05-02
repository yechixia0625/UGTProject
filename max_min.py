import os
import pandas as pd

base_dir = 'Dataset_nonIID_simplex'  # 基础目录，包含客户端文件夹
clients = ['local_1', 'local_2', 'local_3', 'local_4', 'local_5', 'local_6']
labels = [0, 1, 2, 3]

for client in clients:
    client_path = os.path.join(base_dir, client)
    for label in labels:
        label_path = os.path.join(client_path, str(label))
        max_values = None
        min_values = None

        # 遍历标签目录下的每个CSV文件
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            if file.endswith('.csv'):
                # 读取CSV文件
                data = pd.read_csv(file_path, header=None)

                # 更新每列的最大值和最小值
                if max_values is None:
                    max_values = data.max()
                    min_values = data.min()
                else:
                    max_values = pd.DataFrame([max_values, data.max()]).max()
                    min_values = pd.DataFrame([min_values, data.min()]).min()

        # 保存最大值和最小值到CSV文件
        max_min_values = pd.DataFrame({'Max': max_values, 'Min': min_values})
        max_min_values.to_csv(os.path.join(label_path, f'max_min_values_label.csv'), index_label='Column')
