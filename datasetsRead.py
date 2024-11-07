import numpy as np
import pandas as pd

# 加载 npz 文件
full_data = np.load('/Users/yechixia/Documents/VSCode/python/IndividualProject/DASHlink_full_fourclass_raw_comp.npz')  # 请将 'your_file.npz' 替换为你的实际文件名

# 提取 data 和 label 数组
data = full_data['data']
label = full_data['label']