import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/Users/yechixia/UGTProject/datasetsByArrivalAirport/airport_data_counts.csv')

# 假设每个客户端需要的数据量
client_data_size = 1000

# 设定每个客户端的标签比例（例如 7:1:1:1）
client_label_ratios = [
    (7, 1, 1, 1),  # 客户端1
    (1, 7, 1, 1),  # 客户端2
    (1, 1, 7, 1),  # 客户端3
    (1, 1, 1, 7),  # 客户端4
    (3, 3, 2, 2),  # 客户端5
    (2, 2, 3, 3),  # 客户端6
]

# 确保数据按机场分组
airport_groups = df.groupby('arrival_airport')

# 存储每个客户端的数据
clients_data = {i: [] for i in range(6)}

# 分配数据
for client_id, label_ratio in enumerate(client_label_ratios):
    label_0_needed, label_1_needed, label_2_needed, label_3_needed = label_ratio
    
    # 当前客户端所需的标签数量
    total_needed = sum(label_ratio)
    label_0_count = int(client_data_size * (label_0_needed / total_needed))
    label_1_count = int(client_data_size * (label_1_needed / total_needed))
    label_2_count = int(client_data_size * (label_2_needed / total_needed))
    label_3_count = client_data_size - label_0_count - label_1_count - label_2_count  # 剩余的标签数量

    # 用于标记已选择的机场
    selected_airports = set()

    # 从机场分组中选择数据
    while label_0_count > 0 or label_1_count > 0 or label_2_count > 0 or label_3_count > 0:
        # 随机选择一个机场（确保没有重复）
        available_airports = list(airport_groups.groups.keys())  # 确保使用 `.groups.keys()` 获取机场名称
        np.random.shuffle(available_airports)
        
        for airport in available_airports:
            if airport not in selected_airports:
                selected_airports.add(airport)
                
                # 获取该机场的数据
                airport_data = airport_groups.get_group(airport)

                # 获取每个标签的数据
                label_0_data = airport_data[airport_data['label_0_count'] > 0]
                label_1_data = airport_data[airport_data['label_1_count'] > 0]
                label_2_data = airport_data[airport_data['label_2_count'] > 0]
                label_3_data = airport_data[airport_data['label_3_count'] > 0]

                # 根据需要的数量从每个标签数据中抽取
                if label_0_count > 0:
                    to_take = min(label_0_count, len(label_0_data))
                    clients_data[client_id].append(label_0_data.head(to_take))
                    label_0_count -= to_take
                
                if label_1_count > 0:
                    to_take = min(label_1_count, len(label_1_data))
                    clients_data[client_id].append(label_1_data.head(to_take))
                    label_1_count -= to_take
                
                if label_2_count > 0:
                    to_take = min(label_2_count, len(label_2_data))
                    clients_data[client_id].append(label_2_data.head(to_take))
                    label_2_count -= to_take
                
                if label_3_count > 0:
                    to_take = min(label_3_count, len(label_3_data))
                    clients_data[client_id].append(label_3_data.head(to_take))
                    label_3_count -= to_take

                # 如果该客户端数据已满足需求，则停止
                if label_0_count == 0 and label_1_count == 0 and label_2_count == 0 and label_3_count == 0:
                    break

# 检查并输出每个客户端的数据
for client_id, data in clients_data.items():
    print(f"Client {client_id + 1} data count: {len(data)}")
    # 合并每个客户端的数据
    client_df = pd.concat(data, ignore_index=True)
    print(client_df[['arrival_airport', 'label_0_count', 'label_1_count', 'label_2_count', 'label_3_count']].head())
