# torch libraries
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# customized libraries
from .EfficientNet import EfficientNet
from .PVTv2 import *
from .decoder_p import Decoder
from .decoder_p import rearrange
from abc import ABC
from torch import einsum
from .swin_encoder import SwinTransformer,SwinB,SwinL,SwinS,SwinT
from .ResNet import Resnet50
from .Res2Net import res2net50

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
torch.backends.cudnn.benchmark = False


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, edge, grad):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(edge))
        k = self.qkv2conv(self.qkv_1(grad))
        v = self.qkv3conv(self.qkv_2(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)
        attn1 = (q @ v.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)

        attn2 = (k @ v.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)

        attn = attn1 @ attn2

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)

###################################################################################################

class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class Fusion(nn.Module): #gConv
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.c1 = ConvBR(channel, channel, kernel_size=1, stride=1, padding=0)
        self.c2 = ConvBR(channel, channel, kernel_size=1, stride=1, padding=0)

        self.g = nn.Sequential(
            ConvBR(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, y):
        return self.g(torch.cat((self.c1(x), self.c2(y)), 1))


class MultiDecoder(nn.Module): #FPN
    def __init__(self, channel, pathNum, flag):
        super(MultiDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.f1 = Fusion(channel)
        self.f2 = Fusion(channel)
        self.f3 = Fusion(channel)
        self.f4 = Fusion(channel)
        self.f5 = Fusion(channel)
        self.f6 = Fusion(channel)
        self.f7 = Fusion(channel)
        self.f8 = Fusion(channel)
        self.f9 = Fusion(channel)
        self.f10 = Fusion(channel)
        self.f11 = nn.Conv2d(3*channel, channel, kernel_size=1, stride=1, padding=0)
 
        self.pathNum = pathNum
        self.flag = flag
    def forward(self, x3, x4, x5):
        # print(x3.shape,x4.shape,x5.shape)
        if self.pathNum == 0:
            # max_size = max(x3.size(2), x4.size(2), x5.size(2))

            x3 = F.interpolate(x3, size=x4.size(2), mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, size=x4.size(2), mode='bilinear', align_corners=False)
            x5 = F.interpolate(x5, size=x4.size(2), mode='bilinear', align_corners=False)
            x = self.f11(torch.cat((x3,x4,x5),1))
        if self.pathNum == 1:
            x4 = self.f1(x4, self.upsample(x5))
            x3 = self.f2(x3, self.upsample(x4))
        if self.pathNum == 2:
            x4 = self.f1(x4, self.upsample(x5))
            x3 = self.f2(x3, self.upsample(x4))
            x4 = self.f3(x4, self.upsample(x5))
            x3 = self.f4(x3, self.upsample(x4))
        if self.pathNum == 3:
            if (self.flag != 1 and self.flag != 2):
                x4 = self.f1(x4, x5)
                x3 = self.f2(x3, self.upsample(x4))
                x4 = self.f3(x4, x5)
                x3 = self.f4(x3, self.upsample(x4))
                x4 = self.f5(x4, x5)
                x3 = self.f6(x3, self.upsample(x4))
            else:
                #if self.flag==2:
                #    x3,x4,x5=x5,x4,x3
                x4 = self.f1(x4, self.upsample(x5))
                x3 = self.f2(x3, self.upsample(x4))
                x4 = self.f3(x4, self.upsample(x5))
                x3 = self.f4(x3, self.upsample(x4))
                x4 = self.f5(x4, self.upsample(x5))
                x3 = self.f6(x3, self.upsample(x4))
        if self.pathNum == 4:
            x4 = self.f1(x4, self.upsample(x5))
            x3 = self.f2(x3, self.upsample(x4))
            x4 = self.f3(x4, self.upsample(x5))
            x3 = self.f4(x3, self.upsample(x4))
            x4 = self.f5(x4, self.upsample(x5))
            x3 = self.f6(x3, self.upsample(x4))
            x4 = self.f7(x4, self.upsample(x5))
            x3 = self.f8(x3, self.upsample(x4))
        if self.pathNum == 5:
            x4 = self.f1(x4, self.upsample(x5))
            x3 = self.f2(x3, self.upsample(x4))
            x4 = self.f3(x4, self.upsample(x5))
            x3 = self.f4(x3, self.upsample(x4))
            x4 = self.f5(x4, self.upsample(x5))
            x3 = self.f6(x3, self.upsample(x4))
            x4 = self.f7(x4, self.upsample(x5))
            x3 = self.f8(x3, self.upsample(x4))
            x4 = self.f9(x4, self.upsample(x5))
            x3 = self.f10(x3, self.upsample(x4))
        return x3

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x

        new_features = super(_DenseLayer, self).forward(F.relu(x1))  # F.relu()
        # if new_features.shape[-1]!=x2.shape[-1]:
        #     new_features =F.interpolate(new_features,size=(x2.shape[2],x2.shape[-1]), mode='bicubic',
        #                                 align_corners=False)
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride,
                 use_bs=True
                 ):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x

#######################################################################################

class PreNorm(nn.Module, ABC):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention2(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FeedForward2(nn.Module, ABC):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

import numpy as np

class grad_net(nn.Module):
    def __init__(self,channel):
        super(grad_net, self).__init__()
        self.resnet = ResNet(ResBlock, num_classes=10)  # 实例化ResNet
        self.conv_f = nn.Conv2d(512, channel, 1)
        self.conv = nn.Conv2d(channel, 1, 1)  # Change the output channels to 1

    def forward(self, x):
        # FFT and high-pass filter
        x1 = x
        x = torch.fft.fftn(x, dim=(2, 3))
        x = torch.fft.fftshift(x, dim=(2, 3))
        rows, cols = x.shape[2:]
        crow, ccol = rows // 2, cols // 2
        mask = torch.ones((1, 3, rows, cols), dtype=torch.bool, device=x.device)
        r = 80
        y, x_np = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        mask_area = x_np*x_np + y*y <= r*r
        mask_area_torch = torch.from_numpy(mask_area).to(x.device)
        mask[..., mask_area_torch] = False
        x = x * mask

        # IFFT
        x = torch.fft.ifftshift(x, dim=(2, 3))
        x = torch.fft.ifftn(x, dim=(2, 3))
        x = torch.abs(x)

        # ResNet and final conv
        x = torch.cat((x1,x),1)
        x = self.resnet(x)  
        x = self.conv_f(x)
        x1 = x
        x = self.conv(x)
        return x, x1


def resize_a(x6, x5, x4, x3, x2, x_g):
    # print("111", x6.shape,x5.shape,x4.shape,x3.shape,x2.shape,x_g.shape)
    x6 = torch.cat((x6, F.interpolate(x_g, size=x6.size()[2:], mode='bilinear', align_corners=False)), dim=1)
    x5 = torch.cat((x5, F.interpolate(x_g, size=x5.size()[2:], mode='bilinear', align_corners=False)), dim=1)
    x4 = torch.cat((x4, F.interpolate(x_g, size=x4.size()[2:], mode='bilinear', align_corners=False)), dim=1)
    x3 = torch.cat((x3, F.interpolate(x_g, size=x3.size()[2:], mode='bilinear', align_corners=False)), dim=1)
    x2 = torch.cat((x2, F.interpolate(x_g, size=x2.size()[2:], mode='bilinear', align_corners=False)), dim=1)
    
    return x6, x5, x4, x3, x2



# 终极版
class SAE(nn.Module):
    def __init__(self, channel=32, arc='B0', M=[8, 8, 8], N=[4, 8, 16], mode='train'):
        super(SAE, self).__init__()
        channel = channel
        if arc == 'EfficientNet-B1':
            print('--> using efficientnet-b1 right now')
            self.context_encoder = EfficientNet.from_pretrained('efficientnet-b1')
            in_channel_list = [16, 40, 112, 320]
            self.flag = 1
        elif arc == 'EfficientNet-B4':
            print('--> using efficientnet-b4 right now')
            self.context_encoder = EfficientNet.from_pretrained('efficientnet-b4')
            in_channel_list = [32, 56, 160, 448]
            self.flag = 1
        elif arc == 'PVTv2-B0':
            print('--> using PVTv2-B0 right now')
            self.context_encoder = pvt_v2_b0(pretrained=True)
            in_channel_list = [16, 64, 160, 256]
            self.flag = 1
        elif arc == 'PVTv2-B1':
            print('--> using PVTv2-B1 right now')
            self.context_encoder = pvt_v2_b1(pretrained=True)
            in_channel_list = [32, 128, 320, 512]
            self.flag = 1
        elif arc == 'PVTv2-B2':
            print('--> using PVTv2-B2 right now')
            self.context_encoder = pvt_v2_b2(pretrained=True)
            in_channel_list = [64 ,128, 320, 512]
            self.flag = 1
        elif arc == 'PVTv2-B3':
            print('--> using PVTv2-B3 right now')
            self.context_encoder = pvt_v2_b3(pretrained=True)
            in_channel_list = [64 ,128, 320, 512]
            self.flag = 1
        elif arc == 'PVTv2-B4':
            print('--> using PVTv2-B4 right now')
            self.context_encoder = pvt_v2_b4(pretrained=True)
            in_channel_list = [64, 128, 320, 512]
            self.flag = 1
        elif arc == 'SwinTransformer-Base':
            print('--> using SwinTransformer-Base right now')
            self.context_encoder = SwinB()
            in_channel_list = [256, 512, 1024, 1024]
            self.flag = 384
        elif arc == 'SwinTransformer-Large':
            print('--> using SwinTransformer-Large right now')
            self.context_encoder = SwinL()
            in_channel_list = [384, 768, 1536, 1536]
            self.flag = 384
        elif arc == 'SwinTransformer-Small':
            print('--> using SwinTransformer-Small right now')
            self.context_encoder = SwinS()
            in_channel_list = [192, 384, 768, 768]
            self.flag = 224
        elif arc == 'SwinTransformer-Tiny':
            print('--> using SwinTransformer-Tiny right now')
            self.context_encoder = SwinT()
            in_channel_list = [192, 384, 768, 768]
            self.flag = 224
        elif arc == 'Res':
            print('--> using Resnet50 right now')
            #self.context_encoder = Resnet50
            self.context_encoder = res2net50(True)
            in_channel_list = [256, 512, 1024, 2048]
            #in_channel_list = [64, 128, 256, 512]
            self.flag = 2
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))

        #self.grad_net = grad_net(int(channel/2))
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample1 = Decoder(channel,mode=mode,channel_list=in_channel_list)
        
        self.dr2 = DimensionalReduction(in_channel=in_channel_list[0], out_channel=channel)
        self.dr3 = DimensionalReduction(in_channel=in_channel_list[1], out_channel=channel)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=channel)
        self.dr5 = DimensionalReduction(in_channel=in_channel_list[3], out_channel=channel)

        self.multi = MultiDecoder(channel, 3, self.flag) #FPN

    def forward(self, x):
        # context path (encoder)
        #if self.flag == 384:
        #    x = F.interpolate(x,size=[384,384],mode='bilinear')
        #elif (self.flag == 224):
        #    x = F.interpolate(x,size=[224,224],mode='bilinear')
        #elif (self.flag == 2):
        #    x = F.interpolate(x,size=[352,352],mode='bilinear')
        x0 = x
        # print(x.shape)
        if self.flag == 1:
            endpoints = self.context_encoder.extract_endpoints(x)
            # print(endpoints)
            x2 = endpoints['reduction_2']  # 64
            x3 = endpoints['reduction_3']  # 128
            x4 = endpoints['reduction_4']  # 320
            x5 = endpoints['reduction_5']  # 512
        else:
            features = self.context_encoder(x)
            x2 = features[0]
            x3 = features[1]
            x4 = features[2]
            x5 = features[3]
        #print(x2.shape,x3.shape,x4.shape,x5.shape)
        #xr2 = self.dr2(x2)
        #xr3 = self.dr3(x3)
        #xr4 = self.dr4(x4)
        #xr5 = self.dr5(x5)

        # print(xr3.shape,xr4.shape,xr5.shape)
        #x6 = self.multi(xr3, xr4, xr5)
        # x_gf, x_g = self.grad_net(x)
        # x6,x5,x4,x3,x2 = resize_a(x6,xr5,xr4,xr3,xr2)
        # x6, x5, x4, x3, x2 = torch.cat((x6, x_g), 1), torch.cat((x5, x_g), 1), torch.cat((x4, x_g), 1), torch.cat((x3, x_g), 1), torch.cat((x2, x_g), 1)
        # print(x6.shape)
        pc = self.upsample1(x5,x4,x3,x2,[352,352])

        return pc

if __name__ == '__main__':
    net = SAE(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).eval()
    inputs = torch.randn(1, 3, 1024, 1024)
    outs = net(inputs)
    print(outs[0].shape)
