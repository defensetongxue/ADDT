import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.networks.backbone import build_feature_extractor, NET_OUT_DIM
import timm

class HolisticHead(nn.Module):
    def __init__(self, in_dim, dropout=0):
        super(HolisticHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.abs(x)

# head是指分类器，输入是一张图片，每个图片会生成好多很多个异常分数
#（比如448*448的图片，生成14*14个异常score，相当于划分成14*14个大小为32*32的patch），取前面最大的K个。
class PlainHead(nn.Module):
    def __init__(self, in_dim, topk_rate=0.1):
        super(PlainHead, self).__init__()
        self.scoring = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, padding=0)  # 输出是通道为1，长宽14*14，相当于全连接层
        self.topk_rate = topk_rate

    def forward(self, x):
        # print("x1", x.shape)
        x = self.scoring(x)
        # print("x2", x.shape)
        x = x.view(int(x.size(0)), -1)
        # print("x3", x.shape)
        topk = max(int(x.size(1) * self.topk_rate), 1)   # 取前10%个
        # print("topk", topk)
        x = torch.topk(torch.abs(x), topk, dim=1)[0]  # 对张量x按照绝对值从大到小进行排序，并保留前topk个最大值，结果保存在x中
        # print("x4.shape", x.shape)
        # print("x4", x)
        x = torch.mean(x, dim=1).view(-1, 1)
        # print("x4", x)
        return x


class CompositeHead(PlainHead):
    def __init__(self, in_dim, topk=0.1):
        super(CompositeHead, self).__init__(in_dim, topk)
        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1),
                                  nn.BatchNorm2d(in_dim),
                                  nn.ReLU())

    def forward(self, x, ref):
        # ref = torch.mean(ref, dim=0).repeat([x.size(0), 1, 1, 1])
        x = ref - x     # 减去参考值作为卷积输入
        x = self.conv(x)
        x = super().forward(x)
        return x



# 注意力网络结构，用于给输入的patch赋权重值，输入是512*2*2，学习注意力权重值
class AttentionNet(nn.Module):
    # patch是7*7的大小
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=200, out_channels=200, kernel_size=2, padding=0)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = nn.Conv2d(in_channels=200, out_channels=512, kernel_size=2, padding=0)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 1)
        self.softmax = nn.Softmax()

    # # 尝试一下patch是2*2的大小
    # def __init__(self):
    #     super(AttentionNet, self).__init__()
    #     self.Conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0)
    #     self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.Conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0)
    #     self.relu = nn.ReLU()
    #     self.fc1 = nn.Linear(4096, 256)
    #     self.fc2 = nn.Linear(256, 1)
    #     self.softmax = nn.Softmax()

    def forward(self, x):
        # print("x1", x.shape)
        x = self.Conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.Conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        # print("x2", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return float(x)

# 用于将特征提取后的feature map划分为512*2*2的patch,
# 并将注意力权重值乘以patch的feature map，最后再组合导一起送到网络中计算各类head
def AttentionModule(input_features):
    # print("the size of input_features in AttentionModule", input_features.shape)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    input_features = input_features.to(device)

    # 输入特征图的大小
    batch_size, num_channels, height, width = input_features.size()

    # 每个通道 patch 的大小
    # patch_size = (7, 7)
    patch_size = (7, 7)
    # print("input_features.shape", input_features.shape)

    # 划分的行数和列数
    num_rows = height // patch_size[0]
    num_cols = width // patch_size[1]

    # 划分后的 patch 数量
    num_patches = num_rows * num_cols

    # 创建一个空的张量来存储划分后的 patch
    patches = input_features.view(batch_size, num_channels, num_rows, patch_size[0], num_cols, patch_size[1])
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(batch_size, num_patches, num_channels, *patch_size)

    # 输出划分后的 patch 张量,这个张量patches.shape torch.Size([batch_size, num_patches, num_channels, patch_height, patch_width])
    # print("patches.shape", patches.shape)
    # print("切分patch结束")

    # 创建一个空的张量来存储注意力权重乘积后的 patch
    weighted_patches = torch.empty_like(patches)

    attention_net = AttentionNet().to(device)

    # 乘以权重_注意力网络得出

    for batch in range(batch_size):
        for i in range(num_patches):
            patch = patches[batch, i]
            # 获取权重值
            attention_output = attention_net(patch.unsqueeze(0))
            weighted_patch = patch * attention_output
            weighted_patches[batch, i] = weighted_patch

    # 重新组合 patch
    output_features = weighted_patches.view(batch_size, num_rows, num_cols, num_channels, *patch_size)
    output_features = output_features.permute(0, 3, 1, 4, 2, 5).contiguous()
    output_features = output_features.view(batch_size, num_channels, height, width)

    return output_features


class DRA(nn.Module):
    def __init__(self, args, backbone="resnet18"):

        super(DRA, self).__init__()
        self.args = args    # 保存模型的配置信息
        self.feature_extractor = build_feature_extractor(backbone, args)  # 首先使用resnet18的卷积部分做特征提取，其输出dim为512
        self.in_c = NET_OUT_DIM[backbone]    # 输入特征图的通道数512
        print("in_c", self.in_c)
        self.holistic_head = HolisticHead(self.in_c)     # 整体异常评分模块，用于对整个特征图进行异常评分
        self.seen_head = PlainHead(self.in_c, self.args.topk)  # 已见类异常评分模块
        self.attention_head = PlainHead(self.in_c, self.args.topk)
        self.pseudo_head = PlainHead(self.in_c, self.args.topk)  # 伪异常评分模块,要修改为注意力头？？？？？？？
        self.composite_head = CompositeHead(self.in_c, self.args.topk)  # 复合异常评分模块，用于对特征图与参考特征图进行比较并生成异常评分。

    def forward(self, image, ref_image, label):  # 这边图像输入的大小是448*448的
        # # print("image.shape", image.shape)
        # # print("lable", label)
        # image_pyramid = list()
        # for i in range(self.cfg.total_heads):  # 长度为cfg.total_heads
        #     image_pyramid.append(list())
        # for s in range(self.cfg.n_scales):  # 然后对于每个尺度s，根据尺度对输入图像进行缩放，并通过特征提取器提取特征
        #     # print("image",image.shape)
        #     # # 通过在不同尺度下处理图像，模型可以对不同尺度上的异常进行检测，从而增强了模型的鲁棒性和适应性
        #     # image_scaled = F.interpolate(image, size=self.cfg.img_size // (2 ** s)) if s > 0 else image
        #     image_scaled = image
        #     # print("image_scaled", image_scaled.shape)
        #     feature = self.feature_extractor(image_scaled)
        #     # print("feature after feature_extractor", feature.shape)
        #     # 将预训练网络提取后的特征512*14*14的feature map传入注意力网络中
        #     feature = AttentionModule(feature)

        #     ref_feature = feature[:self.cfg.nRef, :, :, :]  # 前 self.cfg.nRef 个特征
        #     feature = feature[self.cfg.nRef:, :, :, :]      # 表示剩余的特征图，即除去参考特征之外的部分

        # 图片特征提取，参考图片特征提取
        feature = self.feature_extractor(image)
        ref_feature = self.feature_extractor(ref_image)
        # 图片特征提取后经过注意力网络，参考图片特征提取后经过注意力网络
        feature = AttentionModule(feature)
        ref_feature = AttentionModule(ref_feature)

        # # 即除去参考特征之外的部分注意力分数
        # batch_attention_scores = batch_attention_scores[self.cfg.nRef:, :]

        # # 接下来，根据训练状态，分别通过不同的评分模块计算不同类型的异常评分。将计算得到的异常评分添加到对应的图像金字塔列表中
        # if self.training:
        #     normal_scores = self.holistic_head(feature)
        #     abnormal_scores = self.seen_head(feature[label != 2])
        #     # dummy_scores = self.pseudo_head(feature[label != 1])
        #     # attention_scores = batch_attention_scores
        #     comparison_scores = self.composite_head(feature, ref_feature)
        # else:
        #     normal_scores = self.holistic_head(feature)
        #     abnormal_scores = self.seen_head(feature)
        #     # dummy_scores = self.pseudo_head(feature)
        #     # attention_scores = batch_attention_scores
        #     comparison_scores = self.composite_head(feature, ref_feature)
        # 计算异常分数
        comparison_scores = self.composite_head(feature, ref_feature)
        
        # for i, scores in enumerate([normal_scores, abnormal_scores, comparison_scores]):
        #     image_pyramid[i].append(scores)

        # # 最后，对每个评分列表进行拼接和均值池化操作，得到最终的图像金字塔
        # for i in range(self.cfg.total_heads):
        #     image_pyramid[i] = torch.cat(image_pyramid[i], dim=1)
        #     image_pyramid[i] = torch.mean(image_pyramid[i], dim=1)
        # # print("image_pyramid", image_pyramid)
        # # print("image_pyramid.shape", image_pyramid.shape)
        # return image_pyramid   # 返回一个包含多个异常评分的列表，称为图像金字塔

        return comparison_scores

