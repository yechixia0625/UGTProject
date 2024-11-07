import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv('DASHlink_full_fourclass_raw_meta.csv')

# 指定保存目录
save_directory = 'datasetsCategory'

# 如果保存目录不存在，则创建
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 获取所有唯一的往返机场对
airport_pairs = set([(row['departure_airport'], row['arrival_airport']) if row['departure_airport'] < row['arrival_airport'] else (row['arrival_airport'], row['departure_airport']) for index, row in df.iterrows()])

for pair in airport_pairs:
    # 提取出往返机场对的数据
    pair_data = df[((df['departure_airport'] == pair[0]) & (df['arrival_airport'] == pair[1])) | ((df['departure_airport'] == pair[1]) & (df['arrival_airport'] == pair[0]))]
    
    # 生成文件名，并在前面加上保存目录路径
    filename = os.path.join(save_directory, f"{pair[0]}-{pair[1]}.csv")
    
    # 保存数据到新的CSV文件
    pair_data.to_csv(filename, index=False)

print("数据提取和保存完成！")