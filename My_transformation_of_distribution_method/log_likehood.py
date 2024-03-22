
#  绘制直方图#######################################

import numpy as np
import matplotlib.pyplot as plt

# 从文件中读取数据
with open('decoder_output_metal_nut2.txt', 'r') as file:
    lines = file.readlines()

# 将文本数据转换为浮点数列表
data_lines = [list(map(float, line.strip().split('\t')[0].split())) for line in lines]

# new_data = [np.mean(data) for data in data_lines]  # averages

# new_data = [np.median(data) for data in data_lines]  # median

# # 计算最大值
# data_lines = [data for data in data_lines if data]
# new_data = [np.max(data) for data in data_lines]

# # 计算最小值
# data_lines = [data for data in data_lines if data]
# new_data = [np.min(data) for data in data_lines]0


# # 划分数据为前50行和后150行  wood
# data1 = data_lines[:18]
# data2 = data_lines[19:68]


# # 划分数据为前50行和后150行  cable
# data1 = data_lines[:56]
# data2 = data_lines[57:139]

# 划分数据为前50行和后150行  metal_nut
data1 = data_lines[:20]
data2 = data_lines[21:104]

# # 划分数据为前50行和后150行  pill
# data1 = data_lines[:26]
# data2 = data_lines[27:157]


# # 划分数据为前50行和后150行  screw
# data1 = data_lines[:40]
# data2 = data_lines[41:150]

# # 划分数据为前50行和后150行  grid
# data1 = data_lines[:20]
# data2 = data_lines[21:67]

# # 划分数据为前50行和后150行  toothbrush
# data1 = data_lines[:11]
# data2 = data_lines[12:31]

# # 划分数据为前50行和后150行  zipper
# data1 = data_lines[:31]
# data2 = data_lines[32:140]

# # 划分数据为前50行和后150行  capsule
# data1 = data_lines[:22]
# data2 = data_lines[23:121]

# # 划分数据为前50行和后150行  hazelnut
# data1 = data_lines[:39]
# data2 = data_lines[40:99]


# # 计算最大值
# data1 = [np.mean(data) for data in data1 if data and len(data) > 0]   
# data2 = [np.mean(data) for data in data2 if data and len(data) > 0]

# new_data1 = [np.subtract(data , np.max(data)) for data in data1]
# new_data2 = [np.subtract(data , np.max(data)) for data in data2]

new_data1 = [np.mean(data) for data in data1]
new_data2 = [np.median(data) for data in data2]


# 定义颜色列表
colors1 = ['green'] 
colors2 = ['red']  

# 绘制直方图
plt.hist(new_data1, bins=20, color=colors1, alpha=0.5, label='Normal')
plt.hist(new_data2, bins=20, color=colors2, alpha=0.5, label='Abnormal')

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.legend()

plt.show()


#  绘制直方图#######################################