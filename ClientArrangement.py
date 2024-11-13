import pandas as pd
import matplotlib.pyplot as plt
import os

def wrap_airports(airports_list, max_airports_per_line=6):
    # 将机场列表分成每行最多max_airports_per_line个机场
    return '\n'.join([', '.join(airports_list[i:i + max_airports_per_line]) for i in range(0, len(airports_list), max_airports_per_line)])

def distribute_airports_evenly(input_csv, num_clients):
    # 确保输出文件夹存在
    output_dir = 'client'
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件，确保使用正确的列名
    data = pd.read_csv(input_csv, usecols=['arrival_airport', 'data_count'])
    
    # 创建客户端字典，初始化数据量
    clients = {i: {'Airports': [], 'Total_Data_Count': 0} for i in range(1, num_clients + 1)}
    
    # 按数据量排序机场
    sorted_data = data.sort_values(by='data_count', ascending=False)
    
    # 贪心算法分配机场
    for _, row in sorted_data.iterrows():
        # 找到当前数据量最小的客户端
        min_client = min(clients, key=lambda x: clients[x]['Total_Data_Count'])
        # 分配机场到该客户端
        clients[min_client]['Airports'].append(row['arrival_airport'])
        clients[min_client]['Total_Data_Count'] += row['data_count']

    # 准备数据用于绘图和表格
    client_numbers = []
    airports_list = []
    data_counts = []
    for client, info in clients.items():
        client_numbers.append(client)
        airports_list.append(info['Airports'])
        data_counts.append(info['Total_Data_Count'])

    # 保存CSV文件
    final_data = pd.DataFrame({
        'Client': client_numbers,
        'Airports': [wrap_airports(aps) for aps in airports_list],
        'Total_Data_Count': data_counts
    })
    csv_filename = f'{output_dir}/{num_clients}_clients.csv'
    final_data.to_csv(csv_filename, index=False)

    # 绘制饼图和表格
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 2]})
    
    # 饼图
    ax1.pie(data_counts, labels=[f'Client {n}' for n in client_numbers], autopct='%1.1f%%', startangle=140)
    ax1.set_title(f'Percentage of Total Data by {num_clients} Clients')

    # 表格
    cell_text = [[n, wrap_airports(aps), dc] for n, aps, dc in zip(client_numbers, airports_list, data_counts)]
    row_heights = [0.02 * (wrap_airports(aps).count('\n') + 1) for aps in airports_list]
    
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=cell_text, colLabels=final_data.columns, loc='center', cellLoc='center', colWidths=[0.1, 0.6, 0.3])
    
    # 动态调整每行的行高
    for i, height in enumerate(row_heights):
        for j in range(len(final_data.columns)):
            table[(i+1, j)].set_height(height)

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)  # 调整表格大小

    # 保存图像，文件名包含客户端数量
    plt.savefig(f'{output_dir}/{num_clients}_clients.png', bbox_inches='tight')
    plt.close()

# 使用该函数并设置需要的客户端数量
distribute_airports_evenly('datasetsByArrivalAirport/airport_data_counts.csv', num_clients=4)
