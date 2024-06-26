#!/usr/bin/env Python
# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

from dataloaders.dataloader import initDataloader
from modeling.net import DRA
from tqdm import tqdm
import time
from sklearn.metrics import average_precision_score, roc_auc_score
from modeling.layers import build_criterion
import random

import matplotlib.pyplot as plt
from modeling.net import AttentionModule
import timm
from losses import get_logp_boundary, calculate_bg_spp_loss, normal_fl_weighting, abnormal_fl_weighting
from utils import t2np, get_logp, adjust_learning_rate, warmup_learning_rate, save_results, save_weights, load_weights
from utils.utils import MetricRecorder, calculate_pro_metric, convert_to_anomaly_scores, evaluate_thresholds
from modeling import positionalencoding2d, load_flow_model
import math
from sklearn.manifold import TSNE

log_theta = torch.nn.LogSigmoid()

WEIGHT_DIR = './weights'


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Dataloader
        kwargs = {'num_workers': self.args.workers}
        # data loaders
        self.normal_loader, self.train_loader, self.test_loader = initDataloader.build(self.args, **kwargs)


        # self.model = DRA(args, backbone=self.args.backbone)

        # if self.args.pretrain_dir != None:
        #     self.model.load_state_dict(torch.load(self.args.pretrain_dir))
        #     print('Load pretrain weight from: ' + self.args.pretrain_dir)

        # self.criterion = build_criterion(args.criterion)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    
    def train_meta_epoch(self, args, epoch, encoder, decoders, optimizer):
    # def train_meta_epoch(self, args, epoch, normal_dataloder, train_dataloder, encoder, decoders, optimizer):
        N_batch = 4096
        decoders = [decoder.train() for decoder in decoders]  # 3
        adjust_learning_rate(args, optimizer, epoch)

        # First epoch only training on normal samples to keep training steadily
        # 在第一个meta_epoch中，仅训练正常样本以保持稳定的训练。这里选择加载正常样本数据加载器。
        if epoch == 0:
            data_loader = self.normal_loader   # normal_dataloder    data_loader[0]
        else:
            data_loader = self.train_loader   # train_dataloder     data_loader[1] 
        # I = len(data_loader)   #  因为data_loader=[normal_loader, train_loader]，所以是2
        I = 2
        for sub_epoch in range(args.sub_epochs): # 遍历sub_epoch，sub_epochs=default=8
            total_loss, loss_count = 0.0, 1
            # 边界初始值
            boundaries = (0.0, 0.0)
            # print("--position1")
            # dataloader
            tbar = tqdm(data_loader)
            for index, sample in enumerate(tbar): 
                # print("--position2")
                # time.sleep(0.01)
                # warm-up learning rate
                #  计算并返回当前的学习率。这里使用了warm-up策略，在训练开始时学习率较小，逐渐增加到正常值，以稳定训练过程
                # lr = warmup_learning_rate(args, epoch, i+sub_epoch*I, I*args.sub_epochs, optimizer)
                file_names, gt_label_list = [], []
                image, ref_image, label, file_name = sample['image'], sample['ref_image'], sample['label'], sample['file_name']
                # print("label", label)
                # 记录文件名字与标签
                file_names.extend(file_name) # 记录文件名字
                gt_label_list.extend(t2np(label))
                with open('label in train.txt', 'a') as f:
                    for gt_l, file_name in zip(gt_label_list, file_names):
                        f.write(f"{gt_l}\t{file_name}\n")

                if args.cuda:
                    image, ref_image, label = image.cuda(), ref_image.cuda(), label.cuda()
                #     # print("image.shape", image.shape)
                image.to(args.device)
                ref_image.to(args.device)
                label.to(args.device)

                with torch.no_grad(): 
                    features = encoder(image)  # 使用特征提取器（Encoder）提取特征
                    # print("features.shape", features[0].shape)
                    # print("features.shape", features[1].shape)
                    # print("features.shape", features[2].shape)

                    ref_features = encoder(ref_image)  # 使用特征提取器（Encoder）提取特征

                # # # # # # # # # # # # # # # # # # # # # # # #
                # 这边先相减，再要加上注意力，再到后面的解码器
                # 因为encoder按特征层级输出，所以残差也需要按特征层级相减后存储
                # 相减后得到残差特征
                residual_features = []
                for feat, ref_feat in zip(features, ref_features):
                    residual_feature = ref_feat - feat
                    residual_features.append(residual_feature)
                features = residual_features
                # print("residual_features", residual_features)

                # 加上注意力模块,主要对最后一个层级(即features[2])进行修改
                # print("features.shape", features[2].shape)
                # features[2] = AttentionModule(features[2])

                # # # # # # # # # # # # # # # # # # # # # # # # 

                for l in range(args.feature_levels):  # 遍历不同的特征层级。
                    e = features[l].detach()   # 获取特征层级l的特征，使用detach()函数将其从计算图中分离，避免计算梯度。
                    bs, dim, h, w = e.size()  # 从特征层级中获取对应的batch、维度、长、宽
                    # mask_ = F.interpolate(mask, size=(h, w), mode='nearest').squeeze(1)
                    # mask_ = mask_.reshape(-1)
                    e = e.permute(0, 2, 3, 1).reshape(-1, dim)   # 将特征张量e进行维度的变换和重塑
                    
                    # (bs, 128, h, w)   # pos_embed_dim = default=128,
                    pos_embed = positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                    decoder = decoders[l]
                    
                    perm = torch.randperm(bs*h*w).to(args.device)
                    num_N_batches = bs*h*w // N_batch

                    #在第一个epoch中，只对正常样本进行训练，并计算正常样本的平均损失。
                    # 这样做的目的可能是为了在开始训练时稳定模型，避免异常样本对模型的训练产生较大影响。
                    # 在后续的epoch中，将会对所有样本（正常和异常）进行训练。
                    # 因为num_N_batches的数量是lable的args.feature_levels倍，比如此时num_N_batches的数量是lable的3倍
                    # 所以需要按顺序分配标签，使其对应
                    # expanded_label = [x for x in label for _ in range(args.feature_levels)]
                    # 把单个特征层的标签维度扩充为4096，相当于像素级
                    pixel_label = [x for x in label for _ in range(h*w)]
                    pixel_label = torch.tensor(pixel_label)
                    pixel_label = pixel_label.to(args.device)

                    # 检查标签是否正确对应
                    # for num in range(len(label)):
                    #     for j in range(h*w):
                    #         if label[num] != pixel_label[num*h*w + j]:
                    #             print("label is not match")


                    # 通过这样的采样方式，每次训练时，模型都会在不同位置、不同特征和不同掩码样本上进行训练，
                    # 从而增加了数据的多样性和随机性，有助于提高模型的鲁棒性和泛化能力。
                    for i in range(num_N_batches):
                        idx = torch.arange(i*N_batch, (i+1)*N_batch)
                        p_b = pos_embed[perm[idx]]  
                        e_b = e[perm[idx]]  
                        pixel_label_b = pixel_label[perm[idx]]
                        # m_b = mask_[perm[idx]]  
                        if args.flow_arch == 'flow_model': 
                            z, log_jac_det = decoder(e_b)  
                        else:
                            z, log_jac_det = decoder(e_b, [p_b, ])
                        
                        # pixel_label = torch.tensor(expanded_label[i])
                        # pixel_label = torch.full((N_batch,), pixel_label.item(), dtype=torch.int64)
            
                        # first epoch only training normal samples
                        if epoch == 0:   # 第一次meta_epoch时使用下面的损失
                            if pixel_label_b.sum() == 0:  # only normal loss  # 这个条件表示当前样本是正常样本
                                logps = get_logp(dim, z, log_jac_det)   # 计算样本的概率对数(log probability)
                                logps = logps / dim                    # 将概率对数除以维度dim，以归一化
                                loss = -log_theta(logps).mean()       # 计算损失函数

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                total_loss += loss.item()
                                loss_count += 1
                        else:  # 针对所有样本的处理（epoch>0时）
                        
                            if pixel_label_b.sum() == 0:  # only normal ml loss  正常样本
                                logps = get_logp(dim, z, log_jac_det)  
                                logps = logps / dim
                                if args.focal_weighting: # default = None
                                    normal_weights = normal_fl_weighting(logps.detach())
                                    loss = -log_theta(logps) * normal_weights
                                    loss = loss.mean()
                                else: 
                                    loss = -log_theta(logps).mean()  # 表示计算正常样本的负对数似然损失

                                # 计算正常样本的边界，再根据margin得到异常样本边界
                                # boundaries = get_logp_boundary(logps, pixel_label, args.pos_beta, args.margin_tau, args.normalizer)
                            if pixel_label_b.sum() > 0:  # normal ml loss and bg_sppc loss 这个条件表示当前样本是正负样本都有
                                logps = get_logp(dim, z, log_jac_det)  
                                logps = logps / dim 
                                if args.focal_weighting: # default = None
                                    # logps_detach = logps.detach()
                                    # normal_logps = logps_detach[m_b == 0]
                                    # anomaly_logps = logps_detach[m_b == 1]
                                    # nor_weights = normal_fl_weighting(normal_logps)
                                    # ano_weights = abnormal_fl_weighting(anomaly_logps)
                                    # weights = nor_weights.new_zeros(logps_detach.shape)
                                    # # weights[m_b == 0] = nor_weights
                                    # # weights[m_b == 1] = ano_weights
                                    loss_ml = -log_theta(logps[pixel_label_b == 0.0])  # (256, )
                                    loss_ml = torch.mean(loss_ml)
                                else:
                                    loss_ml = -log_theta(logps[pixel_label_b == 0.0])
                                    loss_ml = torch.mean(loss_ml)    #  # 表示计算正常样本的负对数似然损失
        
                                # 用于计算等效的对数似然决策边界，
                                # 函数的目标是从正常样本的对数似然分布中找到一个等效的决策边界，用于将样本划分为正常和异常。
                                # 这个边界是计算正负样本的边界，需要在存在异常样本的数据中才能计算，所以放在这里计算。
                                boundaries = get_logp_boundary(logps, pixel_label_b, args.pos_beta, args.margin_tau, args.normalizer)
                            #print('feature level: {}, pos_beta: {}, boudaris: {}'.format(l, args.pos_beta, boundaries))
                            if args.focal_weighting:  # default = None
                                loss_n_con, loss_a_con = calculate_bg_spp_loss(logps, pixel_label_b, boundaries, args.normalizer)
                            else:
                                loss_n_con, loss_a_con = calculate_bg_spp_loss(logps, pixel_label_b, boundaries, args.normalizer)
                                
                            loss = loss_ml + args.bgspp_lambda * (loss_n_con + loss_a_con)
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            loss_item = loss.item()
            
                            # if math.isnan(loss_item):
                            #     total_loss += 0.0
                            #     loss_count += 0
                            # else:
                            #     total_loss += loss.item()
                            #     loss_count += 1

                            total_loss += loss_item
                            loss_count += 1                                

            mean_loss = total_loss / loss_count
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, '.format(epoch, sub_epoch, mean_loss))
            # print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_loss, lr))


    def validate(self, args, epoch, data_loader, encoder, decoders):
        print('\nCompute loss and scores on category:')
        
        decoders = [decoder.eval() for decoder in decoders]
        
        # image_list, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
        image_list, ref_image_list, gt_label_list, file_names = [], [], [], []

        logps_list = [list() for _ in range(args.feature_levels)]
        total_loss, loss_count = 0.0, 0

        # 在循环之前创建两个列表来分别存储正例和负例的 t-SNE 数据
        positive_tsne_data_list = []
        negative_tsne_data_list = []
        # 创建一个空的 NumPy 数组来存储所有图像的 logps
        all_logps = []
        # 创建一个空的 NumPy 数组来存储所有图像的标签
        all_labels = []

        with torch.no_grad():
            # 加载数据
            tbar = tqdm(data_loader)
            for i, sample in enumerate(tbar): 
                image, ref_image, label, file_name = sample['image'], sample['ref_image'], sample['label'], sample['file_name']
                # image: (32, 3, 256); label: (32, ); mask: (32, 1, 256, 256)

                file_names.extend(file_name) # 记录文件名字
                gt_label_list.extend(t2np(label))
                # gt_mask_list.extend(t2np(mask))
                with open('label in test.txt', 'w') as f:
                    for  gt_l, file_name in zip(gt_label_list, file_names):
                        f.write(f"{gt_l}\t{file_name}\n")

                        
                image = image.to(args.device) # single scale
                ref_image = ref_image.to(args.device) # single scale
                label = label.to(args.device)
                features = encoder(image)  # BxCxHxW
                ref_features = encoder(ref_image) 

                # # # # # # # # # # # # # # # # # # # # # # # # #
                # # 这边先相减，再要加上注意力，再到后面的解码器
                # # 因为encoder按特征层级输出，所以残差也需要按特征层级相减后存储
                # # 相减后得到残差特征
                # residual_features = []
                # for feat, ref_feat in zip(features, ref_features):
                #     residual_feature = ref_feat - feat
                #     residual_features.append(residual_feature)
                # features = residual_features
                # # print("residual_features", residual_features)

                # 加上注意力模块,主要对最后一个层级(即features[2])进行修改
                # print("features.shape", features[2].shape)
                # features[2] = AttentionModule(features[2])




# ####这边添加一段画T-SANE图的代码,从encoder输出，不画的时候可以注释掉############

#                 # 计算 t-SNE 数据
#                 TSNE_data = features[2]
#                 # 假设 label 是标签数据，1 表示正例，0 表示负例
#                 labels = t2np(label)
#                 positive_mask = labels == 0
#                 negative_mask = labels == 1
#                 positive_tsne_data_list.append(TSNE_data[positive_mask, :])
#                 negative_tsne_data_list.append(TSNE_data[negative_mask, :])

#                 # print("TSNE_data.shape", TSNE_data.shape)
#                 gt_label = np.asarray(gt_label_list, dtype=bool)
#                 with open('encoder_output.txt', 'w') as f:
#                     for data, gt_l in zip(TSNE_data, gt_label):
#                         f.write(f"{data}\t{gt_l}\n")


# ####这边添加一段画T-SANE图的代码，不画的时候可以注释掉############

                # # # # # # # # # # # # # # # # # # # # # # # # 
                for l in range(args.feature_levels):
                    e = features[l]  # BxCxHxW
                    bs, dim, h, w = e.size()
                    e = e.permute(0, 2, 3, 1).reshape(-1, dim)

                    # (bs, 128, h, w)
                    pos_embed = positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                    # decoder
                    decoder = decoders[l]

                    if args.flow_arch == 'flow_model':
                        z, log_jac_det = decoder(e)  
                    else:
                        z, log_jac_det = decoder(e, [pos_embed, ])

                    
                    logps = get_logp(dim, z, log_jac_det)  
                    logps = logps / dim  
                    loss = -log_theta(logps).mean() 
                    total_loss += loss.item()
                    loss_count += 1
                    logps_list[l].append(logps.reshape(bs, h, w))
                    # print("logps_list[2]",logps_list[2])
                    if l==2:
                        logps_l2 = logps
                        flat_vector = torch.cat([tensor.view(-1) for tensor in logps_l2])

                        # 每次添加新的 flat_vector，按行堆叠到 all_logps
                        flat_vector = flat_vector.detach().cpu().numpy()
                        # print("flat_vector.shape", flat_vector.shape)
                        # print("lable", label)
                        all_logps.append(flat_vector)
                        all_labels.append(label.cpu().numpy())



                # # 计算 t-SNE 数据
                # TSNE_data = logps_list[2]
                # TSNE_data = torch.Tensor(TSNE_data)
                # print("TSNE_data.shape", TSNE_data.shape)
                # # 计算 t-SNE 数据
                # # TSNE_data = [tensor.cpu().numpy() for tensor in logps_list[2]]  # 将每个张量移到CPU并转换为NumPy数组
                # # TSNE_data = np.array(TSNE_data)  # 转换为NumPy数组

                # # 假设 label 是标签数据，1 表示正例，0 表示负例
                # labels = t2np(label)
                # positive_mask = labels == 1
                # negative_mask = labels == 0
                # positive_tsne_data_list.append(TSNE_data[positive_mask, :])
                # negative_tsne_data_list.append(TSNE_data[negative_mask, :])


# # # 这段是绘制直方图，经过训练后从logp来得到####################

#         # 将所有图像的 logps 和标签转换为 NumPy 数组
#         all_logps = np.array(all_logps)
#         # print("all_logps.shape", all_logps.shape)
#         all_labels = np.array(all_labels)
#         # print("all_labels.shape", all_labels.shape)

#         with open('decoder_output.txt', 'w') as f:
#             # f.write(f"{all_logps}\n\n\n")
#             # f.write(f"{all_labels}\n")
#             # 写入 all_logps
#             for data in all_logps:
#                 f.write(' '.join(map(str, data)) + '\n')
#             # 在数组之间插入一个分隔行
#             f.write('\n' + '-' * 20 + '\n\n')
#             # 写入 all_labels
#             for label in all_labels:
#                 f.write(str(label[0]) + '\n')

# # # 这段是绘制直方图，从logp来得到####################

# # 这段是绘制直方图，求原始图片的logp###################

        # 将所有图像的 logps 和标签转换为 NumPy 数组
        all_logps = np.array(all_logps)
        # print("all_logps.shape", all_logps.shape)
        all_labels = np.array(all_labels)
        # print("all_labels.shape", all_labels.shape)

        with open('decoder_output_bottle.txt', 'w') as f:
            # f.write(f"{all_logps}\n\n\n")
            # f.write(f"{all_labels}\n")
            # 写入 all_logps
            for data in all_logps:
                f.write(' '.join(map(str, data)) + '\n')
            # 在数组之间插入一个分隔行
            f.write('\n' + '\n\n')
            # 写入 all_labels
            for label in all_labels:
                f.write(str(label[0]) + '\n')

# # 这段是绘制直方图，求原始图片的logp###################




# # 这段是计算了从decoder输出的TSNE图
#         ### 还是绘制T-SNE图################
        
#         # 将所有图像的 logps 和标签转换为 NumPy 数组
#         all_logps = np.array(all_logps)
#         # print("all_logps.shape", all_logps.shape)
#         all_labels = np.array(all_labels)
#         # print("all_labels.shape", all_labels.shape)

#         with open('decoder_output.txt', 'w') as f:
#             # f.write(f"{all_logps}\n\n\n")
#             # f.write(f"{all_labels}\n")
#             # 写入 all_logps
#             for data in all_logps:
#                 f.write(' '.join(map(str, data)) + '\n')
#             # 在数组之间插入一个分隔行
#             f.write('\n' + '-' * 20 + '\n\n')
#             # 写入 all_labels
#             for label in all_labels:
#                 f.write(str(label[0]) + '\n')

#         # 初始化 t-SNE 模型
#         tsne = TSNE(n_components=2, random_state=0, perplexity=5)
#         # tsne = TSNE(n_components=2, random_state=0)
#         # 对数据进行降维
#         tsne_result = tsne.fit_transform(all_logps)

#         # 绘制 t-SNE 图，使用不同颜色表示正例和负例
#         positive_indices = all_labels == 0
#         negative_indices = all_labels == 1

#         # 绘制 t-SNE 图，使用不同颜色表示正例和负例
#         plt.figure(figsize=(8, 6))

#         # 使用 numpy.logical_and 创建正确的索引
#         positive_indices = all_labels[:, 0] == 1
#         negative_indices = all_labels[:, 0] == 0

#         plt.scatter(tsne_result[positive_indices, 0], tsne_result[positive_indices, 1], c='b', label='Positive Examples')
#         plt.scatter(tsne_result[negative_indices, 0], tsne_result[negative_indices, 1], c='r', label='Negative Examples')
#         plt.title("t-SNE Plot of Encoder Outputs")
#         plt.xlabel("Dimension 1")
#         plt.ylabel("Dimension 2")
#         plt.legend()
#         plt.show()
# # 这段是计算了从decoder输出的TSNE图

# # 这段是Encoder的输出的TSNE图##############################
#         # 合并正例和负例的 t-SNE 数据
#         positive_tsne_data = torch.cat(positive_tsne_data_list, dim=0)
#         negative_tsne_data = torch.cat(negative_tsne_data_list, dim=0)

#         # 将所有 t-SNE 数据降维并绘制 t-SNE 图
#         tsne = TSNE(n_components=2, random_state=0)
#         positive_features_2d = tsne.fit_transform(positive_tsne_data.view(-1, 200 * 14 * 14).cpu().numpy())
#         negative_features_2d = tsne.fit_transform(negative_tsne_data.view(-1, 200 * 14 * 14).cpu().numpy())

#         # 绘制 t-SNE 图，使用不同颜色表示正例和负例
#         plt.figure(figsize=(8, 6))
#         plt.scatter(positive_features_2d[:, 0], positive_features_2d[:, 1], c='b', label='Positive Examples')
#         plt.scatter(negative_features_2d[:, 0], negative_features_2d[:, 1], c='r', label='Negative Examples')
#         plt.title("t-SNE Plot of Encoder Outputs")
#         plt.xlabel("Dimension 1")
#         plt.ylabel("Dimension 2")
#         plt.legend()
#         plt.show()

#         ### 还是绘制T-SNE图################
# # 这段是Encoder的输出的TSNE图##############################

        mean_loss = total_loss / loss_count
        print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, mean_loss))
        
        scores = convert_to_anomaly_scores(args, logps_list)

# # 这段是计算了从score输出的TSNE图 ,列表行是268,列是224
#         # 初始化一个空列表来存储结果
#         combined_scores = []

#         # for sublist in scores:
#         #     # 将内层子列表中的所有元素连接成一个大列表，然后添加到结果列表
#         #     combined_sublist = [item for inner_list in sublist for item in inner_list]
#         #     combined_scores.append(combined_sublist)

#         # 使用列表推导式获取每行的第二个子列表
#         last_sublists = [row[2] for row in scores]


#         combined_scores = np.array(last_sublists)
#         # print(combined_scores)
#         all_labels = np.array(all_labels)

#         with open('decoder_output.txt', 'w') as f:
#             # 遍历 combined_scores 中的每个子列表
#             for sublist in combined_scores:
#                 # 将子列表中的元素连接成字符串，以制表符分隔
#                 line = '\t'.join(map(str, sublist))
#                 # 将这一行写入文本文件
#                 f.write(line + '\n')

#             # 写入 all_labels
#             for label in all_labels:
#                 f.write(str(label[0]) + '\n')

#         # 初始化 t-SNE 模型
#         tsne = TSNE(n_components=2, random_state=0, perplexity=10)
#         # tsne = TSNE(n_components=2, random_state=0)
#         # 对数据进行降维
#         tsne_result = tsne.fit_transform(combined_scores)

#         # 绘制 t-SNE 图，使用不同颜色表示正例和负例
#         positive_indices = all_labels == 0
#         negative_indices = all_labels == 1

#         # 绘制 t-SNE 图，使用不同颜色表示正例和负例
#         plt.figure(figsize=(8, 6))

#         # 使用 numpy.logical_and 创建正确的索引
#         positive_indices = all_labels[:, 0] == 0
#         negative_indices = all_labels[:, 0] == 1

#         plt.scatter(tsne_result[positive_indices, 0], tsne_result[positive_indices, 1], c='b', label='Positive Examples')
#         plt.scatter(tsne_result[negative_indices, 0], tsne_result[negative_indices, 1], c='r', label='Negative Examples')
#         plt.title("t-SNE Plot of Encoder Outputs")
#         plt.xlabel("Dimension 1")
#         plt.ylabel("Dimension 2")
#         plt.legend()
#         plt.show()


# # 这段是计算了从score输出的TSNE图 ,列表行是268,列是224‘’‘


        # calculate detection AUROC
        img_scores = np.max(scores, axis=(1, 2))
        gt_label = np.asarray(gt_label_list, dtype=bool)

        # 检验输出问题
        # print("img_scores:", img_scores)
        # print("gt_label_list:", gt_label)
        # print("file_names:", file_names)    
        # 将数据写入文本文件
        with open('output.txt', 'w') as f:
            for img_score, gt_l, file_name in zip(img_scores, gt_label, file_names):
                f.write(f"{img_score}\t{gt_l}\t{file_name}\n")
        

        # print("label:", gt_label, "img_scores:", img_scores)
        img_auc = roc_auc_score(gt_label, img_scores)
        # calculate segmentation AUROC
        # gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
        # pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        #pix_auc = -1
        # pix_pro = -1
        # if args.pro:
        #     pix_pro = calculate_pro_metric(scores, gt_mask)
        
        # # 接着？？？？？
        # if args.vis and epoch == args.meta_epochs - 1:
        #     img_threshold, pix_threshold = evaluate_thresholds(gt_label, gt_mask, img_scores, scores)
        #     save_dir = os.path.join(args.output_dir, args.exp_name, 'vis_results', args.class_name)
        #     os.makedirs(save_dir, exist_ok=True)
        #     plot_visualizing_results(image_list, scores, img_scores, gt_mask_list, pix_threshold, 
        #                             img_threshold, save_dir, file_names, img_types)
        print(' Image-AUC: {}'.format(img_auc))

        return img_auc
        # return img_auc, pix_auc, pix_pro


    def training(self):
        # Feature Extractor   # 创建特征提取器（Encoder）
        encoder = timm.create_model(self.args.backbone_arch, features_only=True, 
                    out_indices=[i+1 for i in range(self.args.feature_levels)], pretrained=True)
        # 将特征提取器移动到指定的设备（如GPU），并设置为评估模式，即禁用Dropout和BatchNormalization等操作
        encoder = encoder.to(self.args.device).eval()
        feat_dims = encoder.feature_info.channels()
        
        # Normalizing Flows
        # 创建多个正态化流（Normalizing Flows）解码器，每个解码器对应一个特征层级
        decoders = [load_flow_model(self.args, feat_dim) for feat_dim in feat_dims]
        decoders = [decoder.to(self.args.device) for decoder in decoders]
        params = list(decoders[0].parameters())
        for l in range(1, self.args.feature_levels):
            params += list(decoders[l].parameters())
        # optimizer 
        optimizer = torch.optim.Adam(params, lr=self.args.lr)
        # data loaders  在初始化函数中已经有 self_normal_loader, self.train_loader, self.test_loader

        # stats   # 初始化用于记录性能指标的MetricRecorder
        img_auc_obs = MetricRecorder('IMG_AUROC')
        # pix_auc_obs = MetricRecorder('PIX_AUROC')
        # pix_pro_obs = MetricRecorder('PIX_AUPRO')

        for epoch in range(self.args.meta_epochs):   # 共进行args.meta_epochs个epoch的训练。
            # if args.checkpoint:
            #     load_weights(encoder, decoders, args.checkpoint)

            # train 
            print('Train meta epoch: {}'.format(epoch))
            # self.train_meta_epoch(self.args, epoch, encoder, decoders, optimizer)
            
            # validate
            img_auc = self.validate(self.args, epoch, self.test_loader, encoder, decoders)
            img_auc_obs.update(100.0 * img_auc, epoch)
            # pix_auc_obs.update(100.0 * pix_auc, epoch)
            # pix_pro_obs.update(100.0 * pix_pro, epoch)
             

            # # 为了绘制TSNE图，改变调用validate函数的写法###########################
            # if epoch >= self.args.meta_epochs-3: # 最后两个图画出来
            #     img_auc = self.validate(self.args, epoch, self.test_loader, encoder, decoders)
            #     img_auc_obs.update(100.0 * img_auc, epoch)
            # # 为了绘制TSNE图，改变调用validate函数的写法###########################




        # # save_results(img_auc_obs, pix_auc_obs, pix_pro_obs, args.output_dir, args.exp_name, args.model_path, args.class_name)
        # save_results(img_auc_obs, self.args.output_dir, self.args.exp_name, self.args.model_path, self.args.class_name)
        # save_weights(encoder, decoders, self.args.output_dir, self.args.exp_name, self.args.model_path)  # avoid unnecessary saves

        return img_auc_obs.max_score


    def normalization(self, data):
        return data


    def save_weights(self, filename):
        # if not os.path.exists(WEIGHT_DIR):
        #     os.makedirs(WEIGHT_DIR)
        torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, filename))

    def load_weights(self, filename):
        path = os.path.join(WEIGHT_DIR, filename)
        self.model.load_state_dict(torch.load(path))

    def init_network_weights_from_pretraining(self):

        net_dict = self.model.state_dict()
        ae_net_dict = self.ae_model.state_dict()

        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.model.load_state_dict(net_dict)

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap)) 
    return roc_auc, ap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="batch size used in SGD")  # default=48    batch 会减去3，ex:8-3=5
    parser.add_argument('--meta_epochs', type=int, default=1, metavar='N',
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub_epochs', type=int, default=1, metavar='N',   # default=8
                        help='number of sub epochs to train (default: 8)')
    
    parser.add_argument("--steps_per_epoch", type=int, default=10, help="the number of batches per epoch")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_threshold", type=int, default=0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='mvtecad', help="a list of data set names")  ## default= mvtecad , eyesight
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    # 修改数据集根目录
    # parser.add_argument('--dataset_root', type=str, default='E:/CS dataset/Zhongshan Dataset 2023.3.19-1.12 combine/1.Zhongshan data combine1.12-3.19', help="dataset root")
    parser.add_argument('--dataset_root', type=str, default='E:/CS dataset/Anomaly Detection Dataset/MVTec_Anomaly_Detection', help="dataset root")
    # parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment/experiment_14', help="dataset root")
    parser.add_argument('--classname', type=str, default='toothbrush', help="dataset class")    # capsule   default='dataset_anomaly_detection_WholeFace'
    parser.add_argument('--img_size', type=int, default=224, help="dataset root")   # 图片大小448*448，改成224*224看看
    parser.add_argument("--nAnomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=1, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument('--know_class', type=str, default=None, help="set the know class for hard setting")
    parser.add_argument('--pretrain_dir', type=str, default=None, help="root of pretrain weight")
    parser.add_argument("--total_heads", type=int, default=1, help="number of head in training") # default=4,
    parser.add_argument("--nRef", type=int, default=5, help="number of reference set")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")

    parser.add_argument('--device', default=torch.device('cuda'),
                        help='using gpu to train model')
    # model hyperparameter
    parser.add_argument('--backbone_arch', default='tf_efficientnet_b6', type=str, metavar='A',
                        help='feature extractor: (default: efficientnet_b6)')
    parser.add_argument('--flow_arch', default='conditional_flow_model', type=str, metavar='A',
                        help='normalizing flow model (default: cnflow)')
    parser.add_argument('--feature_levels', default=3, type=int, metavar='L',
                        help='nudmber of feature layers (default: 3)')
    parser.add_argument('--coupling_layers', default=8, type=int, metavar='L',
                        help='number of coupling layers used in normalizing flow (default: 8)')
    parser.add_argument('--clamp_alpha', default=1.9, type=float, metavar='L',
                        help='clamp alpha hyperparameter in normalizing flow (default: 1.9)')
    parser.add_argument('--pos_embed_dim', default=128, type=int, metavar='L',
                        help='dimension of positional enconding (default: 128)')
    parser.add_argument('--pos_beta', default=0.05, type=float, metavar='L',    # eyesight:default=0.05/
                        help='position hyperparameter for bg-sppc (default: 0.01)')
    parser.add_argument('--margin_tau', default=0.1, type=float, metavar='L',   #  default=0.1, eyesight:default=0.8
                        help='margin hyperparameter for bg-sppc (default: 0.1)')
    parser.add_argument('--normalizer', default=10, type=float, metavar='L',
                        help='normalizer hyperparameter for bg-sppc (default: 10)')
    parser.add_argument('--bgspp_lambda', default=1, type=float, metavar='L',
                        help='loss weight lambda for bg-sppc (default: 1)')
    parser.add_argument('--focal_weighting', action='store_true', default=False,
                        help='asymmetric focal weighting (default: False)')
    
    # learning hyperparamters
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--lr_decay_epochs', nargs='+', default=[50, 75, 90],
                        help='learning rate decay epochs (default: [50, 75, 90])')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, metavar='LR',
                        help='learning rate decay rate (default: 0.1)')
    parser.add_argument('--lr_warm', type=bool, default=True, metavar='LR',
                        help='learning rate warm up (default: True)')
    parser.add_argument('--lr_warm_epochs', type=int, default=2, metavar='LR',
                        help='learning rate warm up epochs (default: 2)')
    parser.add_argument('--lr_cosine', type=bool, default=True, metavar='LR',
                        help='cosine learning rate schedular (default: True)')
    parser.add_argument('--temp', type=float, default=0.5, metavar='LR',
                        help='temp of cosine learning rate schedular (default: 0.5)')                    
    
    # saving hyperparamters
    parser.add_argument('--output_dir', default='output', type=str, metavar='C',
                        help='name of the run (default: output)')
    parser.add_argument('--exp_name', default='bgad_fas', type=str, metavar='C',
                        help='name of the run (default: 0)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='D',
                        help='used in test phase, set same with the exp_name')
    
    # misc hyperparamters
    parser.add_argument("--phase", default='train', type=str, metavar='T',
                        help='train or test phase (default: train)')
    parser.add_argument("--print_freq", default=2, type=int, metavar='T',
                        help='print frequency (default: 2)')                    
    parser.add_argument('--pro', action='store_true', default=False,
                        help='enables estimation of AUPRO metric')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='test data visualizations')
    parser.add_argument('--with_fas', action='store_true', default=True,
                        help='Wether to train with few abnormal samples (default: True)')
    parser.add_argument('--img_level', action='store_true', default=False,
                        help='Wether to train only on image-level features (default: False)')
    parser.add_argument('--not_in_test', action='store_true', default=True,
                        help='Wether to exclude the trained anomalies outside the testset (default: True)')

    args = parser.parse_args()



    return args

if __name__ == '__main__':

    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)

    # 记录参数
    # argsDict = args.__dict__
    # if not os.path.exists(args.experiment_dir):
    #     os.makedirs(args.experiment_dir)
    # with open(args.experiment_dir + '/setting.txt', 'w') as f:
    #     f.writelines('------------------ start ------------------' + '\n')
    #     for eachArg, value in argsDict.items():
    #         f.writelines(eachArg + ' : ' + str(value) + '\n')
    #     f.writelines('------------------- end -------------------')

    # # model path 
    # # 这行代码构建了模型路径的字符串，
    # # 它包含数据集名称、模型骨干架构、流架构和当前类别名称。这将用于保存训练好的模型。
    # args.model_path = "{}_{}_{}_{}".format(
    #     args.dataset, args.backbone_arch, args.flow_arch, args.classname)


    print('meta_epochs:', trainer.args.meta_epochs, 'sub_epochs:', trainer.args.sub_epochs)
    print('batch_size:', trainer.args.batch_size)
    print('pos_beta:', trainer.args.pos_beta, 'margin_tau:', trainer.args.margin_tau)
    print("classname", trainer.args.classname)
    # trainer.model = trainer.model.to('cuda')
    # trainer.criterion = trainer.criterion.to('cuda')
    # for epoch in range(0, trainer.args.meta_epochs):
    #     trainer.training(epoch)
    
    trainer.training()
    # trainer.eval()
    # trainer.save_weights(args.savename)

