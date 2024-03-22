#
# import torch
#
# # 输入特征图的大小
# input_features = torch.randn(9, 3, 14, 14)
# print("input_features", input_features)
# # 每个通道 patch 的大小
# patch_size = (7, 7)
#
# # 划分的行数和列数
# num_rows = input_features.size(1) // patch_size[0]
# num_cols = input_features.size(2) // patch_size[1]
#
# # 划分后的 patch 数量
# num_patches = num_rows * num_cols
#
# # 创建一个空的张量来存储划分后的 patch
# patches = torch.empty(num_patches, input_features.size(0), *patch_size)
#
# # 遍历每个通道，按顺序划分 patch
# for channel in range(input_features.size(0)):
#     for row in range(num_rows):
#         for col in range(num_cols):
#             # 计算当前 patch 在输入特征图中的起始位置
#             start_row = row * patch_size[0]
#             start_col = col * patch_size[1]
#
#             # 获取当前 patch
#             patch = input_features[channel, start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]]
#             print("patch.shape", patch.shape)
#             # 将 patch 存储到 patches 张量中
#             patch_index = row * num_cols + col
#             patches[patch_index, channel, :, :] = patch
#
# # 输出划分后的 patch 张量
# print(patches.shape)
# # print(patches[0])
#
# # 乘以权重
# for i in range(num_patches):
#     patches[i] = patches[i] * 0.5  # 需要改成attention网络的输出
#
#
# # 重新组合 patch
# output_features = torch.empty_like(input_features)
# for channel in range(input_features.size(0)):
#     for row in range(num_rows):
#         for col in range(num_cols):
#             # 计算当前 patch 在输出特征图中的起始位置
#             start_row = row * patch_size[0]
#             start_col = col * patch_size[1]
#
#             # 获取当前 patch
#             patch_index = row * num_cols + col
#             patch = patches[patch_index, channel, :, :]
#
#             # 将 patch 存储到输出特征图中
#             output_features[channel, start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch
#
# # 输出重组后的特征图
# # print("output_features:", output_features)
# # print("output_features.shape:", output_features.shape)
#


# import torch
#
# def AttentionModule(input_features):
#     # if torch.cuda.is_available():
#     #     device = torch.device('cuda')
#     # else:
#     #     device = torch.device('cpu')
#     # input_features = input_features.to(device)
#
#     # 输入特征图的大小
#     batch_size, num_channels, height, width = input_features.size()
#
#     # 每个通道 patch 的大小
#     patch_size = (2, 2)
#     print("input_features.shape", input_features.shape)
#
#     # 划分的行数和列数
#     num_rows = height // patch_size[0]
#     num_cols = width // patch_size[1]
#
#     # 划分后的 patch 数量
#     num_patches = num_rows * num_cols
#
#     # 创建一个空的张量来存储划分后的 patch
#     patches = input_features.view(batch_size, num_channels, num_rows, patch_size[0], num_cols, patch_size[1])
#     print("patches1.shape", patches.shape)
#     patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
#     print("patches2.shape", patches.shape)
#     patches = patches.view(batch_size, num_patches, num_channels, *patch_size)
#     print("patches3.shape", patches.shape)
#     print("patches", patches)
#     # 输出划分后的 patch 张量,这个张量patches.shape torch.Size([batch_size, num_patches, num_channels, patch_height, patch_width])
#     print("patches.shape", patches.shape)
#     print("切分patch结束")
#
#     # 创建一个空的张量来存储注意力权重乘积后的 patch
#     weighted_patches = torch.empty_like(patches)
#
#     # attention_net = AttentionNet()
#
#     # 乘以权重_注意力网络得出
#     for batch in range(batch_size):
#         for i in range(num_patches):
#             patch = patches[batch, i]
#
#             # 获取权重值
#             # attention_output = attention_net(patch.unsqueeze(0))
#             weighted_patch = patch * 0.5
#
#             weighted_patches[batch, i] = weighted_patch
#
#     print("attention网络结束")
#
#     # 重新组合 patch
#     output_features = weighted_patches.view(batch_size, num_rows, num_cols, num_channels, *patch_size)
#     output_features = output_features.permute(0, 3, 1, 4, 2, 5).contiguous()
#     output_features = output_features.view(batch_size, num_channels, height, width)
#
#     print("output_features.shape:", output_features.shape)
#     print("output_features:", output_features)
#
#     print("组合patch结束")
#
#     return output_features
#
#
# if __name__ == '__main__':
#     # 输入特征图的大小
#     input_features = torch.randn(2, 3, 14, 14)
#     print("input_features", input_features)
#     AttentionModule(input_features)




#####
# import torch
# import random
#
# batch_size = 2
# num_scores = 4
#
# batch_attention_scores = []
# for batch in range(batch_size):
#     attention_scores = []  # 一张图片的多个注意力分数
#
#     for i in range(num_scores):
#         # 生成一个0到1之间的随机浮点数，并保留两位小数
#         random_float = round(random.random(), 2)
#         print("attention_output", random_float)
#         attention_scores.append(random_float)
#
#     print("attention_scores", attention_scores)
#     # 将一张图片多个注意力分数转化为张量
#     attention_scores_tensor = torch.tensor(attention_scores)
#     print("attention_scores_tensor.shape", attention_scores_tensor.shape)
#
#     batch_attention_scores.append(attention_scores_tensor)
#
# batch_attention_scores = torch.stack(batch_attention_scores, dim=0)
# print("batch_attention_scores.shape", batch_attention_scores.shape)
#

# # 求两个正态分布的交点
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# from scipy.optimize import fsolve

# def normal_distribution(x, mean, std):
#     return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

# def intersection_equation(x, params):
#     mean_a, std_a, mean_b, std_b = params
#     return normal_distribution(x, mean_a, std_a) - normal_distribution(x, mean_b, std_b)

# def find_intersection_point(mean_a, std_a, mean_b, std_b):
#     params = (mean_a, std_a, mean_b, std_b)
#     initial_guess = (mean_a + mean_b) / 2.0  # Initial guess for the intersection point

#     result = fsolve(intersection_equation, initial_guess, args=(params,))
#     intersection_x = result[0]

#     return intersection_x

# # Example usage:
# mean_a = 5
# std_a = 1
# mean_b = 8
# std_b = 2

# # Calculate the intersection point
# intersection_point = find_intersection_point(mean_a, std_a, mean_b, std_b)
# print("intersection_point", intersection_point)

# # Generate x values for plotting
# x = np.linspace(min(mean_a - 3 * std_a, mean_b - 3 * std_b), max(mean_a + 3 * std_a, mean_b + 3 * std_b), 1000)

# # Compute the corresponding y values for the two normal distributions
# y_a = normal_distribution(x, mean_a, std_a)
# y_b = normal_distribution(x, mean_b, std_b)


# # Print the coordinates of the intersection point
# print("Intersection point coordinates: ({}, {})".format(intersection_point, normal_distribution(intersection_point, mean_a, std_a)))

# # Plot the two normal distributions
# plt.plot(x, y_a, label='Distribution A (mean={}, std={})'.format(mean_a, std_a))
# plt.plot(x, y_b, label='Distribution B (mean={}, std={})'.format(mean_b, std_b))

# # Plot the intersection point
# plt.scatter(intersection_point, normal_distribution(intersection_point, mean_a, std_a), color='red', label='Intersection')

# plt.xlabel('X')
# plt.ylabel('Probability Density')
# plt.title('Intersection of Two Normal Distributions')
# plt.legend()
# plt.grid(True)
# plt.show()


# # 绝对路径减去dataset_path = 相对路径
# import os
# dataset_path = 'E:/CS dataset/Zhongshan Dataset 2023.3.19-1.12 combine/1.Zhongshan data combine1.12-3.19/dataset_anomaly_detection_WholeFace'
# absolute_path = 'E:/CS dataset/Zhongshan Dataset 2023.3.19-1.12 combine/1.Zhongshan data combine1.12-3.19/dataset_anomaly_detection_WholeFace/test/base/person105_2/44 (5).jpg'

# # 使用os.path.relpath函数获取相对路径
# relative_path = os.path.relpath(absolute_path, dataset_path)

# print(relative_path)



# split = 'train'
# name = f"datalist_{split}.txt"
# print(name)



# 两个随机列表
# import random

# # 两个大小相同的列表
# list1 = [1, 2, 3, 4, 5]
# list2 = ['A', 'B', 'C', 'D', 'E']

# # 使用相同的随机种子来打乱两个列表
# random_seed = 42
# random.Random(random_seed).shuffle(list1)
# random.Random(random_seed).shuffle(list2)

# # 打印两个列表
# print(list1)  # 输出类似 [3, 5, 2, 1, 4]
# print(list2)  # 输出类似 ['C', 'E', 'B', 'A', 'D']



# # 看tf_efficientnet_b6网络结构
# import timm

# # 加载模型，不加载权重
# model = timm.create_model('tf_efficientnet_b6', pretrained=False)

# # 打印模型结构
# print(model)



# # 判断数据加载器是否正常迭代
# from dataloaders.dataloader import initDataloader
# import argparse
# import torch 
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch_size", type=int, default=4, help="batch size used in SGD")  # default=48
#     parser.add_argument("--steps_per_epoch", type=int, default=10, help="the number of batches per epoch")
#     parser.add_argument("--epochs", type=int, default=100, help="the number of epochs")  # default=30

#     parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
#     parser.add_argument("--test_threshold", type=int, default=0,
#                         help="the outlier contamination rate in the training data")
#     parser.add_argument("--test_rate", type=float, default=0.0,
#                         help="the outlier contamination rate in the training data")
#     parser.add_argument("--dataset", type=str, default='mvtecad', help="a list of data set names")  ## 这个跑其他数据集也不用修改
#     parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
#     parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
#     parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
#     parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
#     # 修改数据集根目录
#     parser.add_argument('--dataset_root', type=str, default='E:/CS dataset/Zhongshan Dataset 2023.3.19-1.12 combine/1.Zhongshan data combine1.12-3.19', help="dataset root")
#     # parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
#     parser.add_argument('--experiment_dir', type=str, default='./experiment/experiment_14', help="dataset root")
#     parser.add_argument('--classname', type=str, default='dataset_anomaly_detection_WholeFace', help="dataset class")    # capsule
#     parser.add_argument('--img_size', type=int, default=448, help="dataset root")   # 图片大小448*448
#     parser.add_argument("--nAnomaly", type=int, default=10, help="the number of anomaly data in training set")
#     parser.add_argument("--n_scales", type=int, default=1, help="number of scales at which features are extracted")
#     parser.add_argument('--backbone', type=str, default='resnet18', help="backbone")
#     parser.add_argument('--criterion', type=str, default='deviation', help="loss")
#     parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
#     parser.add_argument('--know_class', type=str, default=None, help="set the know class for hard setting")
#     parser.add_argument('--pretrain_dir', type=str, default=None, help="root of pretrain weight")
#     parser.add_argument("--total_heads", type=int, default=1, help="number of head in training") # default=4,
#     parser.add_argument("--nRef", type=int, default=5, help="number of reference set")
#     parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")

#     parser.add_argument('--device', default=torch.device('cuda'),
#                         help='using gpu to train model')
#     # model hyperparameter
#     parser.add_argument('--backbone_arch', default='tf_efficientnet_b6', type=str, metavar='A',
#                         help='feature extractor: (default: efficientnet_b6)')
#     parser.add_argument('--flow_arch', default='conditional_flow_model', type=str, metavar='A',
#                         help='normalizing flow model (default: cnflow)')
#     parser.add_argument('--feature_levels', default=3, type=int, metavar='L',
#                         help='nudmber of feature layers (default: 3)')
#     parser.add_argument('--coupling_layers', default=8, type=int, metavar='L',
#                         help='number of coupling layers used in normalizing flow (default: 8)')
#     parser.add_argument('--clamp_alpha', default=1.9, type=float, metavar='L',
#                         help='clamp alpha hyperparameter in normalizing flow (default: 1.9)')
#     parser.add_argument('--pos_embed_dim', default=128, type=int, metavar='L',
#                         help='dimension of positional enconding (default: 128)')
#     parser.add_argument('--pos_beta', default=0.05, type=float, metavar='L',
#                         help='position hyperparameter for bg-sppc (default: 0.01)')
#     parser.add_argument('--margin_tau', default=0.1, type=float, metavar='L',
#                         help='margin hyperparameter for bg-sppc (default: 0.1)')
#     parser.add_argument('--normalizer', default=10, type=float, metavar='L',
#                         help='normalizer hyperparameter for bg-sppc (default: 10)')
#     parser.add_argument('--bgspp_lambda', default=1, type=float, metavar='L',
#                         help='loss weight lambda for bg-sppc (default: 1)')
#     parser.add_argument('--focal_weighting', action='store_true', default=False,
#                         help='asymmetric focal weighting (default: False)')
    
#     # learning hyperparamters
#     parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
#                         help='learning rate (default: 2e-4)')
#     parser.add_argument('--lr_decay_epochs', nargs='+', default=[50, 75, 90],
#                         help='learning rate decay epochs (default: [50, 75, 90])')
#     parser.add_argument('--lr_decay_rate', type=float, default=0.1, metavar='LR',
#                         help='learning rate decay rate (default: 0.1)')
#     parser.add_argument('--lr_warm', type=bool, default=True, metavar='LR',
#                         help='learning rate warm up (default: True)')
#     parser.add_argument('--lr_warm_epochs', type=int, default=2, metavar='LR',
#                         help='learning rate warm up epochs (default: 2)')
#     parser.add_argument('--lr_cosine', type=bool, default=True, metavar='LR',
#                         help='cosine learning rate schedular (default: True)')
#     parser.add_argument('--temp', type=float, default=0.5, metavar='LR',
#                         help='temp of cosine learning rate schedular (default: 0.5)')                    
#     parser.add_argument('--meta_epochs', type=int, default=25, metavar='N',
#                         help='number of meta epochs to train (default: 25)')
#     parser.add_argument('--sub_epochs', type=int, default=8, metavar='N',
#                         help='number of sub epochs to train (default: 8)')
#     args = parser.parse_args()

#     return args

# args = parse_args()
# kwargs = {'num_workers': args.workers}
# # 构建数据加载器
# normal_loader, train_loader, test_loader = initDataloader.build(args, **kwargs)

# # 尝试遍历数据加载器
# for batch_idx, (sample) in enumerate(train_loader):
#     image, ref_image, label = sample['image'], sample['ref_image'], sample['label'] 

#     print(f"Batch {batch_idx}: Data shape = {image.shape}")
    
#     # 在这里可以加入更多的打印语句来查看数据的内容和形状
    
#     if batch_idx >= 5:
#         break  # 限制打印批次数量，以免输出过多

# import torch
# # 扩充列表三倍
# label = [0, 1, 0]
# expanded_label = [x for x in label for _ in range(3)]

# # pixel_label = torch.tensor(expanded_label[1])
# # print("pixel_label",pixel_label)
# # pixel_label = torch.full((4096,), pixel_label.item(), dtype=torch.int64)
# # print("pixel_label",pixel_label)
# # print("pixel_label.shape",pixel_label.shape)

# pixel_label = [expanded_label[1]]*10
# print(pixel_label)

# 三个列表数据写到txt中，
# # 三个列表的数据
# img_scores = [0.8, 0.6, 0.7, 0.9, 0.4, 0.3, 0.6, 0.2]
# gt_label_list = [1, 0, 1, 1, 0, 0, 1, 0]
# file_names = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg", "image6.jpg", "image7.jpg", "image8.jpg"]

# # 将数据写入文本文件
# with open('output.txt', 'w') as f:
#     for img_score, gt_label, file_name in zip(img_scores, gt_label_list, file_names):
#         f.write(f"{img_score}\t{gt_label}\t{file_name}\n")

# print("Data written to output.txt")



# # 你可以使用Python的random模块来从一个列表中随机提取十个元素
# import random

# # 假设你的normal data列表如下
# normal_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# # 从normal data列表中随机提取十个元素
# random_selection = random.sample(normal_data, 10)

# print("随机提取的十个元素:", random_selection)


# # 绘制正态分布图
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# # 创建一个灰色背景的figure对象
# fig = plt.figure()


# # 第一个正态分布
# mu1 = 0
# sigma1 = 1.5
# x1 = np.linspace(mu1 - 3 * sigma1, mu1 + 3 * sigma1, 1000)
# y1 = norm.pdf(x1, mu1, sigma1)

# # 第二个正态分布
# mu2 = 7  # 调整均值以使两个分布交叉
# sigma2 = 0.8
# x2 = np.linspace(mu2 - 3 * sigma2, mu2 + 3 * sigma2, 1000)
# y2 = norm.pdf(x2, mu2, sigma2)

# # 填充整个图的区域（灰色）
# gray_color = 'gray'
# plt.fill_between(x1, 0, y1, color=gray_color)
# plt.fill_between(x2, 0, y2, color=gray_color)

# # 填充第一个正态分布下面的区域（自定义颜色：红色）
# fill_color1 = (192/255, 0, 0)  # RGB颜色值
# plt.fill_between(x1, y1, color=fill_color1, alpha=1, label='正态分布1下面积')

# # 填充第二个正态分布下面的区域（自定义颜色：绿色）
# fill_color2 = (90/255, 190/255, 142/255)  # RGB颜色值
# plt.fill_between(x2, y2, color=fill_color2, alpha=0.8, label='正态分布2下面积')

# # 寻找交叉区域
# intersection_area = np.trapz(np.minimum(y1, y2), dx=(x1[1] - x1[0]))

# # 添加标题和标签
# # plt.title('两个正态分布及其下面积')
# # plt.xlabel('X轴')
# # plt.ylabel('概率密度')

# # # 显示图例
# # plt.legend()

# # 显示图形
# plt.show()


# import torch

# # 假设您有一个名为 tensor 的3维张量
# tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# # 使用 .tolist() 方法将张量转换为 Python 列表
# tensor_list = tensor.tolist()

# # 打印转换后的列表
# print(tensor_list)




# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # 创建一个示例数据集，这里假设data是一个NxD的NumPy数组，其中N是样本数量，D是特征维度
# data = np.random.rand(100, 10)

# # 初始化t-SNE模型
# tsne = TSNE(n_components=2, random_state=0)

# # 对数据进行降维
# tsne_result = tsne.fit_transform(data)

# # 绘制t-SNE图
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
# plt.title("t-SNE Plot")
# plt.show()
# import numpy as np

# # 假设你有一个 (14, 14) 的列表 data
# data = np.random.rand(14, 14)

# # 使用 reshape 转换为一维向量
# vector = data.reshape(-1)

# # 输出一维向量
# print(vector)



# # 假设 score 是一个三维列表
# # 假设 score 是一个三维列表
# scores = [[[1, 2, 3],
#          [4, 5, 6],
#          [7, 8, 9]],
#         [[10, 11, 12],
#          [13, 14, 15],
#          [16, 17, 18]],
#         [[19, 20, 21],
#          [22, 23, 24],
#          [25, 26, 27]]]

# # 使用列表解析将每个内层的子列表组合成一个大的列表
# combined_data = [sum(sublist, []) for sublist in scores]

# # 输出组合后的列表
# # print(combined_data)

# # 假设 scores 是你的三维列表

# # 初始化一个空列表来存储结果
# combined_scores = []

# # 遍历三维列表的每个内层子列表
# for sublist in scores:
#     # 将内层子列表中的所有元素连接成一个大列表，然后添加到结果列表
#     combined_sublist = [item for inner_list in sublist for item in inner_list]
#     combined_scores.append(combined_sublist)


# # 使用列表推导式获取每行的第二个子列表
# second_sublists = [row[2] for row in scores]

# # combined_scores 现在包含了每个内层子列表连接而成的大列表
# print(second_sublists)




#  绘制直方图#######################################

import numpy as np
import matplotlib.pyplot as plt

# 从文件中读取数据
with open('decoder_output_wood.txt', 'r') as file:
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


# 划分数据为前50行和后150行
data1 = data_lines[:39]
data2 = data_lines[40:99]


# 计算最大值
# data1 = [np.mean(data) for data in data1 if data and len(data) > 0]   
# data2 = [np.mean(data) for data in data2 if data and len(data) > 0]


new_data1 = [np.min(data) for data in data1]
new_data2 = [np.min(data) for data in data2]


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
