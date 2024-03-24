import numpy as np
import os, sys
from datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision import transforms
from datasets.cutmix import CutMix
import random
from utils import t2np
import torch

# My dataloader
class Eyesight(BaseADDataset):
    
    def __init__(self, args, train_type, train = True):
        super(Eyesight).__init__()
        self.args = args
        self.train = train
        self.train_type = train_type
        self.classname = self.args.classname
        self.know_class = self.args.know_class
        self.pollution_rate = self.args.cont_rate
        if self.args.test_threshold == 0 and self.args.test_rate == 0:
            self.test_threshold = self.args.nAnomaly
        else:
            self.test_threshold = self.args.test_threshold

        self.root = os.path.join(self.args.dataset_root, self.classname)
        self.transform = self.transform_train() if self.train else self.transform_test()
        self.transform_pseudo = self.transform_pseudo()

        if self.train is True:
            split = 'train'
        else:
            split = 'test'

        # trian目录下，添加数据集根目录
        normal_data = list()
        ref_data = list()
        test_labels = list()

        with open(f"datalist_{split}.txt", "r") as file:
            for line in file:
                line = line.strip() # 使用strip()方法去除行尾的换行符
                good_img = line.split(' _')[0]  
                normal_data.append(good_img)
                ref_img = line.split(' _')[-1]
                ref_data.append(ref_img)
                test_label = line.split(' _')[1] # test的标签
                test_labels.append(test_label)
                

        self.nPollution = int((len(normal_data)/(1-self.pollution_rate)) * self.pollution_rate)
        if self.test_threshold==0 and self.args.test_rate>0:
            self.test_threshold = int((len(normal_data)/(1-self.args.test_rate)) * self.args.test_rate) + self.args.nAnomaly

        self.ood_data = self.get_ood_data()


        outlier_data, outlier_ref_data, pollution_data = self.split_outlier()
        outlier_data.sort()
        outlier_ref_data.sort()

        normal_data = normal_data + pollution_data  # 训练图片
        ref_data = ref_data + pollution_data  # 参考图片

        normal_label = np.zeros(len(normal_data)).tolist()
        outlier_label = np.ones(len(outlier_data)).tolist()

        if self.train_type  == "normal_set":
            self.images = normal_data       # 正常的图片
            self.ref_images = ref_data      # 对应参考图片
            self.labels = np.array(normal_label)
        elif self.train_type  == "train_set":
            self.images = normal_data + outlier_data     # 训练或者测试图片
            self.ref_images = ref_data + outlier_ref_data   # 对应参考图片
            self.labels = np.array(normal_label + outlier_label)           
        else:
            self.images = normal_data      # 训练或者测试图片
            self.ref_images = ref_data    # 对应参考图片
            test_labels_int = [int(label) for label in test_labels]
            test_labels = torch.tensor(test_labels_int)
            # test_labels = test_labels.to('cuda')
            self.labels = np.array(test_labels)

        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()
        # print(len(self.images))

    def get_ood_data(self):
        ood_data = list()
        if self.args.outlier_root is None:
            return None
        dataset_classes = os.listdir(self.args.outlier_root)
        for cl in dataset_classes:
            if cl == self.args.classname:
                continue
            cl_root = os.path.join(self.args.outlier_root, cl, 'train', 'good')
            ood_file = os.listdir(cl_root)
            for file in ood_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    ood_data.append(os.path.join(cl_root, file))
        return ood_data

    def split_outlier(self):
        outlier_data_dir = os.path.join(self.root, 'test')
        outlier_classes = os.listdir(outlier_data_dir)
        # 函数检查是否存在名为self.know_class的异常类别。
        # 如果存在，将从异常数据文件夹中提取属于该异常类别的数据，并将其标记为已知异常数据。
        if self.know_class in outlier_classes:
            print("Know outlier class: " + self.know_class)
            outlier_data = list()
            know_class_data = list()
            for cl in outlier_classes:
                if cl == 'good':
                    continue
                outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
                for file in outlier_file:
                    if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                        if cl == self.know_class:
                            know_class_data.append('test/' + cl + '/' + file)
                        else:
                            outlier_data.append('test/' + cl + '/' + file)
            np.random.RandomState(self.args.ramdn_seed).shuffle(know_class_data)
            know_outlier = know_class_data[0:self.args.nAnomaly]
            unknow_outlier = outlier_data
            if self.train:
                return know_outlier, list()
            else:
                return unknow_outlier, list()

        split = "test"
        outlier_data = list()
        outlier_ref_data = list()
        with open(f"datalist_{split}.txt", "r") as file:
            for line in file:
                line = line.strip() # 使用strip()方法去除行尾的换行符
                label = int(line.split(' _')[1])
                if label == 1:
                    outlier_img = line.split(' _')[0]  
                    outlier_data.append(outlier_img)
                    ref_img = line.split(' _')[-1]
                    outlier_ref_data.append(ref_img)        
        np.random.RandomState(self.args.ramdn_seed).shuffle(outlier_data)
        np.random.RandomState(self.args.ramdn_seed).shuffle(outlier_ref_data)
        if self.train:
            # 训练模式：在这种情况下，函数返回 outlier_data 列表中前 self.args.nAnomaly 个数据作为已知异常数据，
            # 以及从 self.args.nAnomaly 位置开始直到 self.args.nAnomaly + self.nPollution 位置的数据作为污染数据
            return outlier_data[0:self.args.nAnomaly], outlier_ref_data[0:self.args.nAnomaly], outlier_data[self.args.nAnomaly:self.args.nAnomaly + self.nPollution]
        else:
            # 测试模式：在这种情况下，函数返回 outlier_data 列表中从 self.test_threshold 位置开始到列表末尾的数据作为未知异常数据，
            # 并返回一个空列表作为污染数据，表示在测试时不使用污染数据。
            return outlier_data[self.test_threshold:], outlier_ref_data[self.test_threshold:], list()


    def load_image(self, path):
        # if 'npy' in path[-3:]:
        #     img = np.load(path).astype(np.uint8)
        #     img = img[:, :, :3]
        #     return Image.fromarray(img)
        return Image.open(path).convert('RGB')

    def transform_train(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size,self.args.img_size)),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def transform_pseudo(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size,self.args.img_size)),
            CutMix(),  # 在图像分类任务中，CutMix() 通过从一个图像中随机裁剪一个区域，并将该区域与另一个图像中的对应区域进行混合，生成一个新的训练样本。
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def transform_test(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file_name = self.images[index]
        image = self.load_image(os.path.join(self.root, self.images[index]))  # 训练图片
        # print("image",image.shape)
        ref_image = self.load_image(os.path.join(self.root, self.ref_images [index])) # 参考图片
        transform = self.transform
        label = self.labels[index]
        sample = {'image': transform(image), 'ref_image': transform(ref_image), 'label': label, 'file_name': file_name}
 
        return sample
    



