import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import timm
import numpy as np
import os
import json
import cv2
from pycocotools.coco import COCO
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from PVTv2 import *
import math
from PIL import Image
BATCH_SIZE = 8
NUM_CLASSES = 6
# NUM_CLASSES = 7
LEARNING_RATE = 5e-5
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])
#
# import os
# import numpy as np
# import cv2
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image  # 确保导入PIL库
#
# class CustomSegmentationDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, geo_transform=None, color_transform=None):
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.geo_transform = geo_transform
#         self.color_transform = color_transform
#         self.img_names = os.listdir(img_dir)
#
#         self.imgs = [os.path.join(img_dir, img_name) for img_name in self.img_names]
#         self.masks = [os.path.join(mask_dir, img_name.replace('.jpg', '.bmp')) for img_name in self.img_names]
#
#     def __len__(self):
#         return len(self.img_names)
#
#     def __getitem__(self, index):
#         img_path = self.imgs[index]
#         mask_path = self.masks[index]
#
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
#
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask_tensor = self.convert_mask_to_channels(mask)
#
#         # 将每个通道视为灰度图进行处理
#         mask_images = [Image.fromarray((mask_tensor[idx].numpy() * 255).astype(np.uint8)) for idx in range(mask_tensor.shape[0])]
#
#         # 应用几何变换
#         if self.geo_transform:
#             img = self.geo_transform(img)
#             # 对每个掩码通道进行相同的变换
#             mask_images = [self.geo_transform(mask_image) for mask_image in mask_images]
#
#         # 对图像进行色彩变换
#         if self.color_transform:
#             img = self.color_transform(img)
#
#         # 重新组合掩码通道
#         mask_tensor = torch.stack([torch.from_numpy(np.array(mask_image) / 255) for mask_image in mask_images])
#
#         # 二值化处理
#         mask_tensor = (mask_tensor >= 0.8).float()  # 将大于等于0.8的值设为1，其余设为0
#
#         return img, mask_tensor
#
#     def convert_mask_to_channels(self, mask):
#         height, width = mask.shape
#         mask_tensor = torch.zeros((NUM_CLASSES, height, width), dtype=torch.float)
#
#         for idx in range(NUM_CLASSES):
#             color_mask = (mask == idx)
#             mask_tensor[idx][color_mask] = 1.0
#
#         return mask_tensor


# class CustomSegmentationDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, geo_transform=None, color_transform=None, mask_transform=None):
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.geo_transform = geo_transform
#         self.color_transform = color_transform
#         self.img_names = os.listdir(img_dir)
#         self.mask_transform = mask_transform
#
#         self.imgs = [os.path.join(img_dir, img_name) for img_name in self.img_names]
#         self.masks = [os.path.join(mask_dir, img_name.replace('.jpg', '.bmp')) for img_name in self.img_names]
#
#     def __len__(self):
#         return len(self.img_names)
#
#     def __getitem__(self, index):
#         img_path = self.imgs[index]
#         mask_path = self.masks[index]
#
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
#         img = Image.fromarray(img)  # 转换为PIL图像
#
#         # 读取灰度掩码
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
#         mask_tensor = torch.from_numpy(mask)
#         mask = self.convert_mask_to_channels(mask_tensor)
#
#         # print(mask.unsqueeze(1).shape)
#         mask = F.interpolate(mask.unsqueeze(1), size=[384, 384], mode='bilinear', align_corners=False).squeeze(1)
#         # print(mask.shape)
#         # mask = Image.fromarray(mask_tensor.numpy().astype(np.uint8))
#
#         # 对图像和掩码进行几何变换
#         # if self.geo_transform:
#         #     img = self.geo_transform(img)
#         #     mask = self.geo_transform(mask)
#
#         # 对图像进行色彩变换
#         if self.color_transform:
#             img = self.color_transform(img)
#             # mask = self.mask_transform(mask)
#         # print(mask.shape)
#         return img, mask
#
#     def convert_mask_to_channels(self, mask):
#         # 获取掩码的宽度和高度
#         # print(mask.shape)
#         height, width = mask.shape  # 使用mask.shape获取高度和宽度
#
#         # 创建一个空的通道 tensor
#         mask_tensor = torch.zeros((NUM_CLASSES, height, width), dtype=torch.float)
#
#         # 将每个像素根据像素值赋值到对应的通道
#         for idx in range(NUM_CLASSES):
#             # 创建灰度掩码
#             color_mask = (mask == idx)  # 使用 tensor 操作进行比较
#             mask_tensor[idx][color_mask] = 1.0  # 设置对应的通道为1
#
#         return mask_tensor

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
import torchvision.transforms as T
import torch

class BMPSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes, is_train=True):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.is_train = is_train
        self.img_names = os.listdir(img_dir)

        # 数据增强：根据是否为训练集，选择不同的增强方式
        if is_train:
            # 对应于TensorFlow中的数据增强方式
            self.transforms = A.Compose([
                A.Resize(384, 384, interpolation=1),  # 最近邻插值
                A.HorizontalFlip(p=0.5),  # 水平翻转
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5, border_mode=0),  # 平移、缩放和旋转
                A.Affine(shear=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),  # 亮度对比度变换
            ], additional_targets={'mask': 'mask'})
        else:
            self.transforms = A.Compose([
                A.Resize(384, 384, interpolation=1),  # 测试集同样使用最近邻插值
            ], additional_targets={'mask': 'mask'})

        # 归一化：无论是训练还是测试，图像都需要归一化
        self.img_normalize = T.Compose([
            T.ToTensor(),  # 转换为Tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和方差
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 获取图像路径
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 打开图像
        image = np.array(Image.open(img_path).convert("RGB"))

        # 获取对应的mask路径，并读取mask
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.bmp'))
        mask = np.array(Image.open(mask_path))  # mask读取为灰度图

        # 应用几何变换（同步作用于image和mask）
        transformed = self.transforms(image=image, mask=mask)

        # 确保mask使用的是"最近邻插值"，避免生成非类别的值
        image = transformed['image']
        mask = transformed['mask']

        # 对image进行归一化
        image = self.img_normalize(Image.fromarray(image))

        # 将mask转换为one-hot编码
        mask = torch.from_numpy(mask).long()  # [H, W] 先转换为long类型

        # 将值为0, 1, 6的元素设置为0（背景），其他元素从1开始计数
        mask[(mask == 0) | (mask == 2) | (mask == 7)] = 0  # 去掉的元素全部变为0

        # 减少类别值，使其他类别从1开始
        mask[mask == 1] = 1  # 类别2变为1
        mask[mask == 3] = 2  # 类别3变为2
        mask[mask == 4] = 3  # 类别4变为3
        mask[mask == 5] = 4  # 类别5变为4
        mask[mask == 6] = 5

        # 进行one-hot编码
        mask_onehot = torch.nn.functional.one_hot(mask, num_classes=self.num_classes)  # [H, W, num_classes]

        # 确保维度正确，进行转置
        mask_onehot = mask_onehot.permute(2, 0, 1).float()  # [num_classes, H, W]，转置维度

        return image, mask_onehot


# 示例用法
train_img_dir = "/data1/kangkejun/SUIM/train_val/images/"
train_mask_dir = "/data1/kangkejun/SUIM/train_val/masks_gray/"
num_classes = NUM_CLASSES  # 假设有16个类别，包括背景

test_img_dir = "/data1/kangkejun/SUIM/TEST/images/"
test_mask_dir = "/data1/kangkejun/SUIM/TEST/masks_gray/"

# 创建训练集dataset
train_dataset = BMPSegmentationDataset(img_dir=train_img_dir, mask_dir=train_mask_dir, num_classes=num_classes,
                                       is_train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 创建测试集dataset
test_dataset = BMPSegmentationDataset(img_dir=test_img_dir, mask_dir=test_mask_dir, num_classes=num_classes,
                                      is_train=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
# # 定义几何变换和色彩变换
# geo_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(degrees=90),
#     transforms.Resize((384, 384)),
# ])
#
# color_transform = transforms.Compose([
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.Resize((384, 384)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
#
# mask_transform = transforms.Compose([
#     transforms.Resize((384, 384)),
#     transforms.ToTensor()
# ])
#
# test_geo_transform = transforms.Lambda(transforms.Resize((384, 384)))  # 直接返回输入
# test_color_transform = transforms.Compose([
#     transforms.Resize((384, 384)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
# # 创建数据集实例
# train_dataset = CustomSegmentationDataset(
#     img_dir="/data1/kangkejun/SUIM/train_val/images/",
#     mask_dir="/data1/kangkejun/SUIM/train_val/masks_gray/",  # 这里假设标签存放在一个名为labels的文件夹中
#     geo_transform=geo_transform,
#     color_transform=color_transform
# )
#
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
# test_dataset = CustomSegmentationDataset(
#     img_dir="/data1/kangkejun/SUIM/TEST/images/",
#     mask_dir="/data1/kangkejun/SUIM/TEST/masks_gray/",  # 这里假设验证集标签存放在一个名为labels的文件夹中
#     geo_transform=test_geo_transform,
#     color_transform=test_color_transform
# )
#
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# class BoundaryLoss(nn.Module):
#     def __init__(self, sigma=1):
#         super(BoundaryLoss, self).__init__()
#         self.sigma = sigma
#
#     def forward(self, inputs, targets):
#         # 计算边界
#         targets_boundary = F.conv2d(targets.unsqueeze(1),
#                                      weight=self.create_gaussian_kernel().to(inputs.device),
#                                      stride=1, padding=1)
#         inputs_boundary = F.conv2d(inputs.sigmoid().unsqueeze(1),
#                                     weight=self.create_gaussian_kernel().to(inputs.device),
#                                     stride=1, padding=1)
#         boundary_loss = F.binary_cross_entropy_with_logits(inputs_boundary, targets_boundary)
#         return boundary_loss
#
#     def create_gaussian_kernel(self):
#         kernel_size = 3
#         kernel = torch.zeros((1, 1, kernel_size, kernel_size))
#         center = kernel_size // 2
#         for x in range(kernel_size):
#             for y in range(kernel_size):
#                 kernel[0, 0, x, y] = torch.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * self.sigma ** 2))
#         return kernel / kernel.sum()


class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算 softmax 输出
        inputs = torch.softmax(inputs, dim=1)  # 形状: [batch_size, num_classes, height, width]
        num_classes = inputs.size(1)  # 获取类别数量

        # 初始化损失
        dice_loss = 0.0

        for cls in range(num_classes):
            input_cls = inputs[:, cls]  # 获取每个类别的预测
            target_cls = targets[:, cls]  # 获取 one-hot 编码对应的类别

            intersection = (input_cls * target_cls).sum()
            dice_score = (2. * intersection + self.smooth) / (input_cls.sum() + target_cls.sum() + self.smooth)

            dice_loss += (1 - dice_score)

        # 计算平均损失
        if self.reduction == 'mean':
            return dice_loss / num_classes
        elif self.reduction == 'sum':
            return dice_loss
        else:
            return dice_loss


class BoundaryLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BoundaryLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 使用 softmax 获取类别概率
        inputs = torch.softmax(inputs, dim=1)  # [batch_size, num_classes, H, W]

        # 计算输入和目标的边界
        inputs_boundary = self.get_boundary(inputs)

        # 将 one-hot 编码的目标转换为类别索引
        if targets.dim() == 4:  # 检查是否为 one-hot 编码
            targets = torch.argmax(targets, dim=1)  # [batch_size, H, W]

        targets_boundary = self.get_boundary(targets.unsqueeze(1))  # 转换为 [batch_size, 1, H, W]

        # 计算交叉熵损失
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets.long(), reduction=self.reduction)

        # 计算边界损失 (L1 损失，其他可替换)
        boundary_loss = F.l1_loss(inputs_boundary, targets_boundary, reduction=self.reduction)

        # 总损失 = 交叉熵损失 + 边界损失
        loss = ce_loss + boundary_loss

        return loss

    def get_boundary(self, x):
        # 确保输入是浮点类型
        if x.dtype != torch.float32:
            x = x.float()

        # Sobel 卷积核
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().to(x.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).float().to(x.device)

        # 扩展卷积核与输入通道匹配
        kernel_x = kernel_x.repeat(x.size(1), 1, 1, 1)  # [num_channels, 1, 3, 3]
        kernel_y = kernel_y.repeat(x.size(1), 1, 1, 1)  # [num_channels, 1, 3, 3]

        # Sobel 卷积
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=x.size(1))

        # 计算梯度幅度
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return grad


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=2.0, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # 这里假设 targets 是 [batch_size, height, width]
        batch_size = inputs.size(0)
        height = inputs.size(2)
        width = inputs.size(3)

        # 计算 BCE 损失
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float() ,reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# class FocalLoss(nn.Module):
#     def __init__(self, num_classes, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.reduction = reduction
#         self.num_classes = num_classes
#
#     def forward(self, inputs, targets):
#         # 这里假设 targets 是 [batch_size, height, width]
#         batch_size = inputs.size(0)
#         height = inputs.size(2)
#         width = inputs.size(3)
#
#         # 将输入的 logits 转化为概率分布
#         log_probs = F.log_softmax(inputs, dim=1)
#
#         # 使用 one-hot 编码将 targets 转化为 [batch_size, num_classes, height, width]
#         # 计算交叉熵损失
#         loss = - (targets * log_probs).sum(dim=1)  # 在类维度上求和
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss


class PVTv2Extractor(nn.Module):
    def __init__(self):
        super(PVTv2Extractor, self).__init__()

        self.backbone = pvt_v2_b4(pretrained=True)

    def forward(self, x):
        endpoints = self.backbone.extract_endpoints(x)
        # print(endpoints)
        x2 = endpoints['reduction_2']  # 64
        x3 = endpoints['reduction_3']  # 128
        x4 = endpoints['reduction_4']  # 320
        x5 = endpoints['reduction_5']

        return x2, x3, x4, x5


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        self.output_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        feature_2, feature_3, feature_4 = features

        feature_2 = self.lateral_convs[0](feature_2)
        feature_3 = self.lateral_convs[1](feature_3)
        feature_4 = self.lateral_convs[2](feature_4)

        feature_2 = F.interpolate(feature_2, size=feature_3.shape[2:], mode='bilinear', align_corners=False)
        feature_4 = F.interpolate(feature_4, size=feature_3.shape[2:], mode='bilinear', align_corners=False)

        fused_features = torch.cat([feature_2, feature_3, feature_4], dim=1)

        output = self.output_conv(fused_features)

        return output


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, W, H = x.size()

        reduced_features = self.conv1x1(x)  # [batchsize, C//8, W, H]

        query = reduced_features.view(batch_size, -1, W * H)  # [batchsize, C//8, W*H]
        key = reduced_features.view(batch_size, -1, W * H).permute(0, 2, 1)  # [batchsize, W*H, C//8]

        attention_map = torch.bmm(query, key)  # [batchsize, C//8, C//8]
        attention_map = self.softmax(attention_map)  # [batchsize, C//8, C//8]

        value = reduced_features.view(batch_size, -1, W * H)  # [batchsize, C//8, W*H]
        attention_output = torch.bmm(attention_map, value)  # [batchsize, C//8, W*H]

        attention_output = attention_output.view(batch_size, -1, W, H)  # [batchsize, C//8, W, H]

        output = self.conv_out(attention_output)  # [batchsize, C, W, H]

        output = output + x

        return output


class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()

        self.backbone = PVTv2Extractor()

        self.fpn = FPN([128, 320, 512], 512)

        self.conv1x1_1 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(320, 512, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(512, 512, kernel_size=1)

        self.conv1x1_21 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv1x1_31 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv1x1_41 = nn.Conv2d(1024, 512, kernel_size=1)

        self.att5 = AttentionModule(512)
        self.att4 = AttentionModule(512)
        self.att3 = AttentionModule(512)
        self.att2 = AttentionModule(512)
        self.att1 = AttentionModule(512)

        self.seg5 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.seg4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.seg3 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.seg2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.seg1 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        feature_1, feature_2, feature_3, feature_4 = self.backbone(x)
        # P4 = self.seg4(feature_4)
        # print(feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape)
        feature_5 = self.fpn([feature_2, feature_3, feature_4])

        feature_1 = self.conv1x1_1(feature_1)
        feature_2 = self.conv1x1_2(feature_2)
        feature_3 = self.conv1x1_3(feature_3)
        feature_4 = self.conv1x1_4(feature_4)

        feature_54 = F.interpolate(feature_5, size=feature_4.shape[2:], mode='bilinear', align_corners=False)
        feature_4s = torch.cat([feature_4, feature_54], dim=1)

        feature_3s = torch.cat([feature_3, feature_5], dim=1)

        feature_52 = F.interpolate(feature_5, size=feature_2.shape[2:], mode='bilinear', align_corners=False)
        feature_2s = torch.cat([feature_2, feature_52], dim=1)

        feature_2s = self.conv1x1_21(feature_2s)
        feature_3s = self.conv1x1_31(feature_3s)
        feature_4s = self.conv1x1_41(feature_4s)

        D5 = self.att5(feature_5)

        D5 = F.interpolate(D5, size=feature_4s.shape[2:], mode='bilinear', align_corners=False)

        D5 = D5 + feature_4s

        # P5 = self.seg5(D5)

        D4 = self.att4(D5)

        D4 = F.interpolate(D4, size=feature_3s.shape[2:], mode='bilinear', align_corners=False)

        D4 = D4 + feature_3s

        # P4 = self.seg4(D4)

        D3 = self.att3(D4)

        D3 = F.interpolate(D3, size=feature_2s.shape[2:], mode='bilinear', align_corners=False)

        D3 = D3 + feature_2s

        # P3 = self.seg3(D3)

        D2 = self.att2(D3)

        D2 = F.interpolate(D2, size=feature_1.shape[2:], mode='bilinear', align_corners=False)

        D2 = D2 + feature_1

        P2 = self.seg2(D2)

        # P5 = F.interpolate(P5, size=x.size()[2:], mode='bilinear', align_corners=False)
        # P4 = F.interpolate(P4, size=x.size()[2:], mode='bilinear', align_corners=False)
        # P3 = F.interpolate(P3, size=x.size()[2:], mode='bilinear', align_corners=False)
        P2 = F.interpolate(P2, size=x.size()[2:], mode='bilinear', align_corners=False)

        return P2

import timm
import torch
import torch.nn as nn
from torch.nn import functional as F


# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, skip_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#
#         self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
#         self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x, skip_connection):
#         x = self.up(x)
#         if x.size() != skip_connection.size():
#             x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
#         x = torch.cat([x, skip_connection], dim=1)
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         return x
#
#
class DecoderBlock_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock_1, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super(TransformerBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Linearly project query, key, value from x
        self.proj = nn.Linear(dim, dim)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        B, N, C = x.shape

        # Normalize input
        x_norm = self.norm1(x)

        # Apply linear layer to get Q, K, V
        qkv = self.qkv(x_norm).reshape(B, N, 3, C).permute(2, 0, 1, 3)  # (3, B, N, C)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Multihead Attention
        attn_out, _ = self.attn(query, key, value)
        attn_out = self.proj(attn_out)

        # Add & Normalize
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_heads=8):
        super(TransformerDecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.transformer = TransformerBlock(dim=out_channels, num_heads=num_heads)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_connection): 
        x = self.up(x)
        if x.size() != skip_connection.size():
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.relu(self.conv1(x))
        x_flattened = x.view(x.shape[0], x.shape[1], -1).permute(2, 0, 1)  # Prepare for transformer
        x_transformed = self.transformer(x_flattened).permute(1, 2, 0).view_as(x)
        x = self.relu(self.conv2(x_transformed))
        return x


class FPNFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPNFusion, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        fpn_outs = []
        target_size = features[0].shape[2:]

        for i, feature in enumerate(features):
            fpn_out = self.convs[i](feature)
            fpn_out = F.interpolate(fpn_out, size=target_size, mode='bilinear', align_corners=True)
            fpn_outs.append(fpn_out)

        fpn_sum = sum(fpn_outs)
        return self.out_conv(fpn_sum)


class SwinTransUnet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SwinTransUnet, self).__init__()

        self.encoder = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)

        # self.decoder5 = DecoderBlock(1536, 1536, 1536)
        # self.decoder4 = DecoderBlock(1536, 1536, 768)
        # self.decoder3 = DecoderBlock(768, 768, 384)
        # self.decoder2 = DecoderBlock(384, 384, 128)
        # self.decoder1 = DecoderBlock_1(128, 128)

        self.decoder5 = TransformerDecoderBlock(1536, 1536, 1536)
        self.decoder4 = TransformerDecoderBlock(1536, 1536, 768)
        self.decoder3 = TransformerDecoderBlock(768, 768, 384)
        self.decoder2 = TransformerDecoderBlock(384, 384, 128)
        # self.decoder1 = TransformerDecoderBlock(128, 384, 128)
        self.decoder1 = DecoderBlock_1(128, 128)

        self.fpn_fusion = FPNFusion([1536, 1536, 768, 384], 768)

        self.adjust_fpn_enc3 = nn.Conv2d(768, 1536, kernel_size=1)
        self.adjust_fpn_enc2 = nn.Conv2d(768, 768, kernel_size=1)
        self.adjust_fpn_enc1 = nn.Conv2d(768, 384, kernel_size=1)

        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)
        self.patch_size = 4
        self.img_size = 384

    def forward(self, x):
        patches = self.encoder.patch_embed(x)
        x = self.encoder.pos_drop(patches)

        enc1 = self.encoder.layers[0](x)  # [B, 384, 48, 48]
        enc2 = self.encoder.layers[1](enc1)  # [B, 768, 24, 24]
        enc3 = self.encoder.layers[2](enc2)  # [B, 1536, 12, 12]
        enc4 = self.encoder.layers[3](enc3)  # [B, 1536, 12, 12]
        # print(enc1.shape,enc2.shape,enc3.shape,enc4.shape)

        B = enc4.shape[0]

        H1 = W1 = int(math.sqrt(enc1.shape[1]))  # 28x28
        enc1 = enc1.view(B, H1, W1, 384).permute(0, 3, 1, 2)

        H2 = W2 = int(math.sqrt(enc2.shape[1]))  # 14x14
        enc2 = enc2.view(B, H2, W2, 768).permute(0, 3, 1, 2)

        H3 = W3 = int(math.sqrt(enc3.shape[1]))  # 7x7
        enc3 = enc3.view(B, H3, W3, 1536).permute(0, 3, 1, 2)

        H4 = W4 = int(math.sqrt(enc4.shape[1]))  # 7x7
        enc4 = enc4.view(B, H4, W4, 1536).permute(0, 3, 1, 2)

        fpn_out = self.fpn_fusion([enc4, enc3, enc2, enc1])

        fpn_enc3 = self.adjust_fpn_enc3(
            F.interpolate(fpn_out, size=enc3.shape[2:], mode='bilinear', align_corners=True))
        fpn_enc2 = self.adjust_fpn_enc2(
            F.interpolate(fpn_out, size=enc2.shape[2:], mode='bilinear', align_corners=True))
        fpn_enc1 = self.adjust_fpn_enc1(
            F.interpolate(fpn_out, size=enc1.shape[2:], mode='bilinear', align_corners=True))

        # Decoder stages
        # dec4 = self.decoder4(enc4, enc3 + fpn_enc3 / 4)  #
        # dec3 = self.decoder3(dec4, enc2 + fpn_enc2 / 4)  #
        # dec2 = self.decoder2(dec3, enc1 + fpn_enc1 / 4)  #
        dec5 = self.decoder5(fpn_enc3, enc4)
        dec5 = F.interpolate(dec5, size=[12, 12], mode='bilinear', align_corners=False)
        dec4 = self.decoder4(dec5, enc3 + fpn_enc3 / 4)  #
        dec3 = self.decoder3(dec4, enc2 + fpn_enc2 / 4)  #
        dec2 = self.decoder2(dec3, enc1 + fpn_enc1 / 4)  #

        dec2 = F.interpolate(dec2, size=[96, 96], mode='bilinear', align_corners=False)
        enc1 = F.interpolate(enc1, size=[96, 96], mode='bilinear', align_corners=False)
        dec2 = self.decoder1(dec2)
        output = self.final_conv(dec2)
        output = F.interpolate(output, size=[self.img_size, self.img_size], mode='bilinear', align_corners=False)

        return output


class PVTv2Extractor(nn.Module):
    def __init__(self):
        super(PVTv2Extractor, self).__init__()

        self.backbone = pvt_v2_b4(pretrained=True)

    def forward(self, x):
        endpoints = self.backbone.extract_endpoints(x)
        # print(endpoints)
        x2 = endpoints['reduction_2']  # 64
        x3 = endpoints['reduction_3']  # 128
        x4 = endpoints['reduction_4']  # 320
        x5 = endpoints['reduction_5']

        return x2, x3, x4, x5


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        self.output_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        feature_2, feature_3, feature_4 = features

        feature_2 = self.lateral_convs[0](feature_2)
        feature_3 = self.lateral_convs[1](feature_3)
        feature_4 = self.lateral_convs[2](feature_4)

        feature_2 = F.interpolate(feature_2, size=feature_3.shape[2:], mode='bilinear', align_corners=False)
        feature_4 = F.interpolate(feature_4, size=feature_3.shape[2:], mode='bilinear', align_corners=False)

        fused_features = torch.cat([feature_2, feature_3, feature_4], dim=1)

        output = self.output_conv(fused_features)

        return output


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, W, H = x.size()

        reduced_features = self.conv1x1(x)  # [batchsize, C//8, W, H]

        query = reduced_features.view(batch_size, -1, W * H)  # [batchsize, C//8, W*H]
        key = reduced_features.view(batch_size, -1, W * H).permute(0, 2, 1)  # [batchsize, W*H, C//8]

        attention_map = torch.bmm(query, key)  # [batchsize, C//8, C//8]
        attention_map = self.softmax(attention_map)  # [batchsize, C//8, C//8]

        value = reduced_features.view(batch_size, -1, W * H)  # [batchsize, C//8, W*H]
        attention_output = torch.bmm(attention_map, value)  # [batchsize, C//8, W*H]

        attention_output = attention_output.view(batch_size, -1, W, H)  # [batchsize, C//8, W, H]

        output = self.conv_out(attention_output)  # [batchsize, C, W, H]

        output = output + x

        return output

import segmentation_models_pytorch as smp
# from transformers import Mask2FormerForSemanticSegmentation
#
# # 初始化模型并设置类别数量
# model = Mask2FormerForSemanticSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
# model.config.num_labels = NUM_CLASSES  # 设置类别数量
#
# model = timm.create_model('segmenter_vit_base', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
# model = smp.Unet(
#     encoder_name="efficientnet-b4",  # 选择Swin Transformer
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=NUM_CLASSES,
# ).to(DEVICE)
# 如果有多个GPU
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
#
# model.to(DEVICE)
model = SwinTransUnet(NUM_CLASSES).to(DEVICE)
# model = SegmentationModel(NUM_CLASSES).to(DEVICE)
# model.load_state_dict(torch.load("swin_segmentation_ori_3.pth"), strict=False)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# criterion = nn.CrossEntropyLoss(ignore_index=255)
criterion = FocalLoss(NUM_CLASSES)
criterion1 = BoundaryLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
class CosineAnnealingWithDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay_factor=0.2, last_epoch=-1):
        self.T_0 = T_0  # 初始的循环周期
        self.T_mult = T_mult  # 每次周期增长的倍率
        self.eta_min = eta_min  # 最低学习率
        self.T_i = T_0  # 当前周期长度
        self.T_cur = last_epoch  # 当前周期中的 epoch 计数
        self.decay_factor = decay_factor  # 衰减因子，表示每次最大最小学习率的缩小比例
        super(CosineAnnealingWithDecay, self).__init__(optimizer, last_epoch)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]  # 初始的学习率

    def get_lr(self):
        # 计算新的学习率
        return [self.eta_min + (lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * self.T_cur / self.T_i))
                for lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.T_cur += 1

        # 每次达到周期末尾，重置周期并缩小学习率范围
        if self.T_cur >= self.T_i:
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult  # 增加周期长度
            self.base_lrs = [lr * self.decay_factor for lr in self.base_lrs]  # 每次周期结束后，学习率缩小
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        # 更新学习率
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# 使用该调度器
# optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# scheduler = CosineAnnealingWithDecay(optimizer, T_0=50, T_mult=2, eta_min=1e-7, decay_factor=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

import numpy as np 
import torch 
import torch.nn.functional as F

def calculate_metrics(predictions, targets, num_classes):
    # Initialize lists for metrics
    pixel_accuracy = (predictions == targets).sum() / targets.size
    ious = []
    mean_dsc = []
    fmeasures = []

    # for cls in range(1, num_classes-1):  # Skip the first and last class
    for cls in range(num_classes):
        pred_cls = (predictions == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        tp = intersection  # True positives
        fp = (pred_cls & ~target_cls).sum().item()  # False positives
        fn = (~pred_cls & target_cls).sum().item()  # False negatives

        if union == 0:
            ious.append(np.nan)
            mean_dsc.append(np.nan)
            fmeasures.append(np.nan)
        else:
            iou = intersection / union
            dsc = (2 * intersection) / (pred_cls.sum().item() + target_cls.sum().item()) if (
                pred_cls.sum().item() + target_cls.sum().item()) > 0 else np.nan
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fmeasure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

            ious.append(iou)
            mean_dsc.append(dsc)
            fmeasures.append(fmeasure)

        # Print individual class metrics
        print(f"Class {cls} - IoU: {ious[-1]:.4f}, DSC: {mean_dsc[-1]:.4f}, F-measure: {fmeasures[-1]:.4f}")

    mean_iou = np.nanmean(ious)
    mean_dsc = np.nanmean(mean_dsc)
    mean_fmeasure = np.nanmean(fmeasures)

    return pixel_accuracy, mean_iou, mean_dsc, mean_fmeasure


import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 定义类别与RGB颜色的映射
color_map = {
    0: (0, 0, 0),  # Background waterbody (BW)
    1: (0, 0, 255),  # Human divers (HD)
    2: (0, 255, 255),  # Wrecks/ruins (WR)
    3: (255, 0, 0),  # Robots (RO)
    4: (255, 0, 255),  # Reefs and invertebrates (RI)
    5: (255, 255, 0)  # Fish and vertebrates (FV)
}


def save_predictions(predictions, save_dir, file_names):
    """
    将预测的类别保存为bmp格式图片，每个像素使用对应类别的RGB值表示。
    :param predictions: 预测的类别 (N, H, W) 数组
    :param save_dir: 保存预测结果的文件夹
    :param file_names: 对应的文件名列表
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, pred in enumerate(predictions):
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        # 将类别ID映射到RGB颜色
        for class_id, color in color_map.items():
            pred_rgb[pred == class_id] = color

        # 将结果保存为bmp图片
        save_path = os.path.join(save_dir, file_names[idx].replace('.jpg', '_pred.bmp'))
        Image.fromarray(pred_rgb).save(save_path)


def evaluate_model(model, test_loader, num_classes, save_dir):
    model.eval()
    all_predictions = []
    all_targets = []
    file_names = test_loader.dataset.img_names  # 获取测试集文件名

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)  # (B, H, W)
            masks = torch.argmax(masks, dim=1)

            # 将预测结果从GPU转移到CPU，并转换为numpy格式
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 保存预测结果到指定文件夹
    save_predictions(all_predictions, save_dir, file_names)

    # 计算并返回指标
    return calculate_metrics(all_predictions.flatten(), all_targets.flatten(), num_classes)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, (images, masks) in enumerate(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        #print(masks.shape)

        optimizer.zero_grad()
        outputs = model(images)

        if outputs.dim() == 4 and outputs.shape[1] != NUM_CLASSES:
            outputs = outputs.permute(0, 2, 3, 1)

        masks = masks.long()
        masks = F.interpolate(masks, size=[384, 384], mode='bilinear', align_corners=False)

        loss = criterion(outputs, masks) + criterion1(outputs, masks)/50
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.8f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Average Loss: {avg_loss:.8f}")

    if True:  # 每 5 个 epoch 进行评估
        pixel_accuracy, mIoU, wIoU, mean_dsc = evaluate_model(model, test_loader, NUM_CLASSES, save_dir = "/data1/kangkejun/SUIM/train_val/predict_mine/")
        print(
            f"Test Metrics after Epoch {epoch + 1}: Pixel Accuracy: {pixel_accuracy:.4f}, mIoU: {mIoU:.4f}, WIoU: {wIoU:.4f}, Mean DSC: {mean_dsc:.4f}")
    # if epoch%10==1:  # 每 5 个 epoch 进行评估
    #     pixel_accuracy, mIoU, wIoU, mean_dsc = evaluate_model(model, train_loader, NUM_CLASSES)
    #     print(
    #         f"Test Metrics after Epoch {epoch + 1}: Pixel Accuracy: {pixel_accuracy:.4f}, mIoU: {mIoU:.4f}, WIoU: {wIoU:.4f}, Mean DSC: {mean_dsc:.4f}")

    torch.save(model.state_dict(), 'swin_segmentation_ori_3.pth')

torch.save(model.state_dict(), 'swin_segmentation_ori_3.pth')


