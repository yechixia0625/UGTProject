import os

def count_files_in_directory(directory):
    # 统计每个文件夹中文件的数量
    file_count = {}
    
    # 遍历文件夹
    for root, dirs, files in os.walk(directory):
        # 获取当前文件夹中的文件数量
        file_count[root] = len(files)
    
    return file_count

def display_file_counts(file_count):
    # 打印每个文件夹及其文件数量
    for folder, count in file_count.items():
        print(f"文件夹: {folder} | 文件数量: {count}")

if __name__ == "__main__":
    directory = input("请输入文件夹路径: ")
    file_count = count_files_in_directory(directory)
    display_file_counts(file_count)
