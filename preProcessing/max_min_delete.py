import os

def delete_max_min_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'max_min_values_label.csv':
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

top_folder = 'Dataset_IID'
delete_max_min_files(top_folder)
