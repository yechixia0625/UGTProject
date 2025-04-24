import os

def delete_max_min_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件名是否符合要求
            if file == 'max_min_values_label.csv':
                # 构造文件的完整路径
                file_path = os.path.join(root, file)
                # 删除文件
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# 指定顶级目录路径
top_folder = 'Dataset_IID'
delete_max_min_files(top_folder)
