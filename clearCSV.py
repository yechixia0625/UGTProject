import os

def delete_csv_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

# 使用示例：将 'your_folder_path' 替换为要删除 .csv 文件的文件夹路径
# delete_csv_files('data_instance_output')
delete_csv_files('Dataset_IID')
