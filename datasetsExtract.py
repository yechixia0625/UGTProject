import os
import pandas as pd

# 文件路径
excel_file = 'Airport_Label.xlsx'  # 这里替换为你的Excel文件路径
dataset_folder = 'datasetsByArrivalAirport'  # 存放机场数据的文件夹
output_folder = 'data_instance_iid_output'  # 输出数据的文件夹

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取Excel文件
excel_df = pd.read_excel(excel_file)

# 遍历每一行（每个机场）
for index, row in excel_df.iterrows():
    airport = row['Airport']
    label_counts = {f'Label{i}': int(row[f'Label{i}']) for i in range(4)}  # 获取每种Label的数量
    
    # 打开对应的机场CSV文件
    airport_file = os.path.join(dataset_folder, f"{airport}.csv")
    if not os.path.exists(airport_file):
        print(f"文件 {airport_file} 不存在，跳过此机场。")
        continue
    
    # 读取机场CSV数据
    airport_df = pd.read_csv(airport_file)
    
    # 初始化字典以存储每种Label的数据
    label_data = {}
    for label, count in label_counts.items():
        label_index = int(label[-1])  # 提取Label的编号 (0, 1, 2, 3)
        # 筛选出对应的Label数据
        label_df = airport_df[airport_df['label'] == label_index]
        
        # 检查是否有足够的数据，如果没有则输出全部可用数据
        if len(label_df) < count:
            print(f"警告：{airport} 中的 {label} 可用数据量少于请求数量，仅输出 {len(label_df)} 个数据实例。")
            count = len(label_df)  # 更新count为可用的最大数据量
        
        # 提取所需数量的数据实例
        label_data[label] = label_df['data_instance'].head(count).reset_index(drop=True)
    
    # 将数据按Label保存到输出DataFrame中
    output_df = pd.DataFrame(label_data)

    # 检查每列是否包含NaN值，如果包含则填充，然后转换为整数
    for column in output_df.columns:
        if output_df[column].isnull().any():
            print(f"警告：列 {column} 包含 NaN 值，将用 -1 填充。")
            output_df[column] = output_df[column].fillna(-1)  # 使用-1填充NaN值
        output_df[column] = output_df[column].astype(int)  # 转换为整数

    # 保存为新的CSV文件
    output_file = os.path.join(output_folder, f"{airport}.csv")
    output_df.to_csv(output_file, index=False)
    print(f"{airport}的数据已保存至 {output_file}")
