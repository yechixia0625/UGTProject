import numpy as np
import pandas as pd

# 加载 npz 文件
data = np.load('DASHlink_full_fourclass_raw_comp.npz')

print(data.files)

for array_name in data.files:
    print(f"{array_name} shape: {data[array_name].shape}")

index = 5,
print("Data:", data['data'][index])
print("Label:", data['label'][index])