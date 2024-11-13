import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv('DASHlink_full_fourclass_raw_meta.csv')

# 指定保存目录
save_directory = 'datasetsByArrivalAirport'

# 如果保存目录不存在，则创建
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 获取所有唯一的到达机场
arrival_airports = df['arrival_airport'].unique()

# 用于记录每个机场对应的数据数量
airport_data_counts = []

for airport in arrival_airports:
    # 提取该到达机场的所有数据
    airport_data = df[df['arrival_airport'] == airport]
    
    # 获取该机场对应的数据数量
    count = len(airport_data)
    
    # 计算各个label的数量
    label_counts = airport_data['label'].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    
    # 记录机场和对应的数量以及各label的数量
    airport_data_counts.append({
        'arrival_airport': airport,
        'data_count': count,
        'label_0_count': label_counts[0],
        'label_1_count': label_counts[1],
        'label_2_count': label_counts[2],
        'label_3_count': label_counts[3]
    })
    
    # 生成文件名，并在前面加上保存目录路径
    filename = os.path.join(save_directory, f"{airport}.csv")
    
    # 保存数据到新的CSV文件
    airport_data.to_csv(filename, index=False)

# 创建一个DataFrame来保存机场数据数量信息
counts_df = pd.DataFrame(airport_data_counts)

# 生成包含所有机场的数据数量的CSV文件
counts_filename = os.path.join(save_directory, "airport_data_counts.csv")
counts_df.to_csv(counts_filename, index=False)

# 将airport_data_counts.csv文件保存为Excel文件
excel_filename = os.path.join(save_directory, "airport_data_counts.xlsx")
counts_df.to_excel(excel_filename, index=False)

print("数据提取和保存完成，包括airport_data_counts.csv和airport_data_counts.xlsx！")
