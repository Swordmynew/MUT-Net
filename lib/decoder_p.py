import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

torch.backends.cudnn.benchmark = False

from einops import rearrange

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)

class fusion(nn.Module): #gConv
    def __init__(self, channel):
        channel = channel
        super(fusion, self).__init__()
        self.g1 = nn.Sequential(
            ConvBR(channel * 4, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=9, stride=1, padding=4)
        )
        self.g2 = nn.Sequential(
            ConvBR(channel * 4, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=5, stride=1, padding=2)
        )
        self.g3 = nn.Sequential(
            ConvBR(channel * 4, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=7, stride=1, padding=3)
        )
        self.g4 = nn.Sequential(
            ConvBR(channel * 4, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.g5 = nn.Sequential(
            ConvBR(channel * 4, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        # self.g6 = ConvBR(channel,64,kernel_size=3,stride=1,padding=1)

    def forward(self, x, xa, xb, xc, xd):
        xa = F.interpolate(xa,size=x.size()[2:],mode='bilinear')
        xb = F.interpolate(xb,size=x.size()[2:],mode='bilinear')
        xc = F.interpolate(xc,size=x.size()[2:],mode='bilinear')
        # xc = xc * F.interpolate(xd,size=x.size()[2:],mode='bilinear')


        xx = torch.cat((x, xa, xb, xc), 1)
        xx1 = self.g1(xx)       
        xx2 = self.g2(xx)
        xx3 = self.g3(xx)
        xx4 = self.g4(xx)
        xxx = self.g5(torch.cat((xx1, xx2, xx3, xx4), 1))
        # xxx1 = self.g6(xxx)
        # print(xxx.shape)
        return xxx
    def initialize(self):
        weight_init(self)

class fusion3(nn.Module): #gConv with different channels
    def __init__(self, channel):
        channel = channel
        super(fusion3, self).__init__()
        self.g1 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=9, stride=1, padding=4)
        )
        self.g2 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=5, stride=1, padding=2)
        )
        self.g3 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=7, stride=1, padding=3)
        )
        self.g4 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.g5 = nn.Sequential(
            ConvBR(channel * 4, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        # self.g6 = ConvBR(channel,64,kernel_size=3,stride=1,padding=1)

    def forward(self, x, xa, xb):
        xa = F.interpolate(xa, size=x.size()[2:], mode='bilinear')
        xb = F.interpolate(xb, size=x.size()[2:], mode='bilinear')


        xx = torch.cat((x, xa, xb), 1)
        xx1 = self.g1(xx)
        xx2 = self.g2(xx)
        xx3 = self.g3(xx)
        xx4 = self.g4(xx)
        xxx = self.g5(torch.cat((xx1, xx2, xx3, xx4), 1))
        # xxx1 = self.g6(xxx)
        # print(xxx.shape)
        return xxx
    def initialize(self):
        weight_init(self)

class Attention(nn.Module):
    def __init__(self, dim, window_size, heads, mode):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        if mode =='train':
            self.mode = True
        else:
            self.mode = False
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.dropout = Dropout(0.3)

    def forward(self, x, mask=None, edge=None):
        b, c, h, w = x.shape
        x0 = x
        self.window_size = int(self.window_size)
        # print(x.shape)
        x = x.view(b, c, h // self.window_size, self.window_size, w // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(b * h * w // (self.window_size ** 2), self.window_size ** 2, c)
        if edge is not None:
            edge = F.interpolate(edge, size=[h, w], mode='bilinear')
            edge = edge.view(b, c, h // self.window_size, self.window_size, w // self.window_size, self.window_size)
            edge = edge.permute(0, 2, 4, 3, 5, 1).contiguous().view(b * h * w // (self.window_size ** 2),self.window_size ** 2, c)
            q = self.to_q(edge)
            q = F.dropout(q,0.3,training=self.mode)
        else:
            q = self.to_q(x)
            q = F.dropout(q,0.3,training=self.mode)

        if mask is not None:
            mask = F.interpolate(mask, size=[h, w], mode='bilinear')
            x0 = x0 * mask
            x0 = x0.view(b, c, h // self.window_size, self.window_size, w // self.window_size, self.window_size)
            x0 = x0.permute(0, 2, 4, 3, 5, 1).contiguous().view(b * h * w // (self.window_size ** 2),self.window_size ** 2, c)
            k = self.to_k(x0)
            k = F.dropout(k,0.3,training=self.mode)
            v = self.to_v(x0)
            v = F.dropout(v,0.3,training=self.mode)
        else:
            k = self.to_k(x)
            k = F.dropout(k,0.3,training=self.mode)
            v = self.to_v(x)
            v = F.dropout(v,0.3,training=self.mode)
        # qkv = torch.cat([q,k,v],dim=-1)
        qkv = [q,k,v]
        # print(qkv.shape)
        q, k, v = map(lambda t: t.reshape(b * h * w // (self.window_size ** 2), self.window_size ** 2, self.heads, -1), qkv)
        dots = torch.einsum('bhij,bhik->bhjk', q, k) * self.scale
        attn = self.softmax(dots)
        # print(attn.shape)
        # b, h, w, n = attn.shape
        # attn_map = attn.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, n * h, n * w)
        # print(attn_map.shape)

        out = torch.einsum('bhjk,bhik->bhij', attn, v)
        out = out.reshape(b * h * w // (self.window_size ** 2), self.window_size ** 2, c)
        out = out.view(b, h // self.window_size, w // self.window_size, self.window_size ** 2 * c)
        out = out.permute(0, 3, 1, 2).contiguous().view(b, -1 , h , w)
        return out
    def initialize(self):
        weight_init(self)

####################################################################################################

class EEA_head(nn.Module):
    def __init__(self, dim=128, num_heads=2, ffn_expansion_factor=4, bias=False, size=44,
                 LayerNorm_type='WithBias',mode='train'):
        super(EEA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, window_size=size/11, heads=num_heads,mode=mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None, edge_f=None):
        # print(x.shape)
        x = x + self.attn(self.norm1(x), mask, edge_f)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)

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
    def initialize(self):
        weight_init(self)


class EEA_module(nn.Module):
    def __init__(self, dim=128,size=44,mode='train'):
        super(EEA_module, self).__init__()
        #self.B_TA = EEA_head()
        #self.F_TA = EEA_head()
        self.TAE = EEA_head(dim=dim, size=size, mode=mode)
        self.EAE = EEA_head(dim=dim, size=size, mode=mode)
        self.TA = EEA_head(dim=dim, size=size, mode=mode)
        self.EA = EEA_head(dim=dim, size=size, mode=mode)
        self.EMA = EEA_head(dim=dim, size=size, mode=mode)
        self.EFMA = EEA_head(dim=dim, size=size, mode=mode)

        self.Fuse = nn.Sequential(nn.Conv2d(6 * dim, 3 * dim, kernel_size=1), nn.Conv2d(3 * dim, dim, kernel_size=3, padding=1))
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        channel = dim
        self.g1 = nn.Sequential(
            ConvBR(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=9, stride=1, padding=4)
        )
        self.g2 = nn.Sequential(
            ConvBR(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=5, stride=1, padding=2)
        )
        self.g3 = nn.Sequential(
            ConvBR(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=7, stride=1, padding=3)
        )
        self.g4 = nn.Sequential(
            ConvBR(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.g5 = nn.Sequential(
            ConvBR(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, side_x, mask, edge, edge_f):
        N, C, H, W = x.shape
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear')
        mask_d = mask.detach()
        edge = edge.detach()
        edge_f = edge_f
        mask_d = torch.sigmoid(mask_d)
        edge = torch.sigmoid(edge)
        # side_x, mask, edge, edge_f = side_x.fill_(1).detach(), mask.fill_(1).detach(), edge.fill_(1).detach(), edge_f.fill_(1).detach()
        #xf = self.F_TA(x, mask_d)
       # xb = self.B_TA(x, 1 - mask_d)
       #  print(x.shape,edge_f.shape)
        xee = self.EAE(x, edge, edge_f)
        # xte = self.TAE(x, edge)
        # xme = self.EMA(x)
        # xmfe = self.EFMA(x, edge, edge_f)
        # x = self.TA(x)
        xe = self.EA(xee)
        x = torch.cat([xee,xe],dim=1)
        side_x = F.interpolate(side_x, size=x.size()[2:], mode='bilinear')
        #side_x = F.interpolate(side_x,size=x.size()[2:],mode='bilinear')
        # x = torch.cat((xee), 1)
        # x = x.view(N, 3 * C, H, W)
        # x = self.Fuse(x)

        
        xx1 = self.g1(x)
        xx2 = self.g2(x)
        xx3 = self.g3(x)
        x = self.g5(torch.cat((xx1, xx2, xx3), 1))
        x = self.TA(x)
        x = self.Fuse2(side_x + side_x * x)

        return x

    def initialize(self):
        weight_init(self)
class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

class Conv_Bn(nn.Module):
    def __init__(self, channels):
        super(Conv_Bn, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1):
        fuse = self.bn1(self.conv1(input1))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

def get_edge(x,y):
    x = torch.where(x < 0.8, 0.0, x.to(torch.double))
    x = torch.where(x >= 1.0, 1.0, x.to(torch.double))
    y = torch.where(y < 0.8, 0.0, y.to(torch.double))
    y = torch.where(y >= 1.0, 1.0, y.to(torch.double))
    z = x - y
    pe = z.to(torch.float32)
    # print(pe.dtype)
    pe = torch.where(pe < 0.8, 0.0, pe.to(torch.double))
    # pe = torch.where(pe >= 0.8, 1.0, pe.to(torch.double))
    pe = pe.to(torch.float32)
    return pe
class BN(nn.Module):
    def __init__(self,channel):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm1d(channel)
    def forward(self,x):
        return self.bn(x)
    def initialize(self):
        weight_init(self)
class ch_resize(nn.Module):
    def __init__(self, channel_list, channel):
        super(ch_resize, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(channel_list[0] + int(channel/2), int(channel_list[0]/4), kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(int(channel_list[0]/4), channels, kernel_size=1, stride=1, bias=False))
        self.c2 = nn.Conv2d(channel_list[1] + int(channel/2), int(channel_list[1]/4), kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(channel_list[2] + int(channel/2), int(channel_list[2]/4), kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(channel_list[3] + int(channel/2), int(channel_list[3]/4), kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(channel + int(channel/2), channel, kernel_size=3, stride=1, padding=1)

    def forward(self,x5,x4,x3,x2,x1):
        return self.c5(x5), self.c4(x4),self.c3(x3),self.c2(x2),self.c1(x1)
    def initialize(self):
        weight_init(self)

def resize_tensors(E5, E4, E3, E2, E1):
    if E5.nelement() % 11 != 0 or E5.nelement() % 12 != 0:
        E5 = F.interpolate(E5, size=[44,44], mode='bilinear')
    if E4.nelement() % 11 != 0 or E4.nelement() % 12 != 0:
        E4 = F.interpolate(E4, size=[22,22], mode='bilinear')
    if E3.nelement() % 11 != 0 or E3.nelement() % 12 != 0:
        E3 = F.interpolate(E3, size=[22,22], mode='bilinear')
    if E2.nelement() % 11 != 0 or E2.nelement() % 12 != 0:
        E2 = F.interpolate(E2, size=[44,44], mode='bilinear')
    if E1.nelement() % 11 != 0 or E1.nelement() % 12 != 0:
        E1 = F.interpolate(E1, size=[88,88], mode='bilinear')

    return E5, E4, E3, E2, E1

def save_images(tensor: torch.Tensor,index):
    assert tensor.is_cuda, "Input tensor must be on GPU"

    mean_tensor = torch.mean(tensor, dim=1, keepdim=False)

    for i in range(mean_tensor.shape[0]):
        index += 1
        image = mean_tensor[i]
        image_np = image.cpu().detach().numpy()
        image_np = (image_np * 255).astype('uint8')
        from PIL import Image
        im = Image.fromarray(image_np)
        im.save(f'image/image_{index}.png')


def get_images(tensor: torch.Tensor):
    # 确保输入的tensor在GPU上
    assert tensor.is_cuda, "Input tensor must be on GPU"

    # 按照通道维度求平均值
    mean_tensor = torch.mean(tensor, dim=1, keepdim=False)

    # 遍历batch中的每个图片并保存
    for i in range(mean_tensor.shape[0]):
        image = mean_tensor[i]
        return image

class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

class Decoder(nn.Module):
    def __init__(self, channels, mode, channel_list):
        super(Decoder, self).__init__()

        #self.ch_resize = ch_resize(channel_list, channels)
        
        self.side_conv1 = nn.Sequential(nn.Conv2d(channel_list[3], int(channel_list[3]/4), kernel_size=1, stride=1),
                                 nn.Conv2d(int(channel_list[3]/4), channels, kernel_size=1, stride=1, bias=False))
                                 
        self.side_conv2 = nn.Sequential(nn.Conv2d(channel_list[2], int(channel_list[2]/4), kernel_size=1, stride=1),
                                 nn.Conv2d(int(channel_list[2]/4), channels, kernel_size=1, stride=1, bias=False))
                                 
        self.side_conv3 = nn.Sequential(nn.Conv2d(channel_list[1], int(channel_list[1]/4), kernel_size=1, stride=1),
                                 nn.Conv2d(int(channel_list[1]/4), channels, kernel_size=1, stride=1, bias=False))
                                 
        self.side_conv4 = nn.Sequential(nn.Conv2d(channel_list[0], int(channel_list[0]/4), kernel_size=1, stride=1),
                                 nn.Conv2d(int(channel_list[0]/4), channels, kernel_size=1, stride=1, bias=False))

        # get state feature of ES-CRF
        self.side_conv1x = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2x = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3x = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4x = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv1n = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2n = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3n = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4n = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.fuse1 = nn.Sequential(nn.Conv2d(int(channels), channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse2 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse3 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse4 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse5 = nn.Sequential(nn.Conv2d(int(channels), channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))

        self.fuse = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        
        self.conv_block = Conv_Block(channels)
        
        self.EEA5 = EEA_module(dim=channels,size=44,mode=mode)
        self.EEA4 = EEA_module(dim=channels,size=22,mode=mode)
        self.EEA3 = EEA_module(dim=channels,size=44,mode=mode)
        self.EEA2 = EEA_module(dim=channels,size=88,mode=mode)

        self.predtrans1 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans5 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        
        # transition feature function of ES-CRF
        
        self.tffx = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.tffn = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self.get_edge_feature5 = fusion3(channels)
        self.get_edge_feature4 = fusion(channels)
        self.get_edge_feature3 = fusion(channels)
        self.get_edge_feature2 = fusion(channels)


        self.edge_out5 = ConvBR(channels, 1, kernel_size=3, stride=1, padding=1)
        self.edge_out2 = ConvBR(channels, 1, kernel_size=3, stride=1, padding=1)
        self.edge_out3 = ConvBR(channels, 1, kernel_size=3, stride=1, padding=1)
        self.edge_out4 = ConvBR(channels, 1, kernel_size=3, stride=1, padding=1)

        # get edge posibility map from each state

        self.mkx = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True)
                                 )
        self.mkn = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True)
                                 )
        self.mkx0 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
                                 
        self.mkn0 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.initialize()

    
    def forward(self, E4, E3, E2, E1, shape):
        # For skip connection
        
        # E5, E4, E3, E2, E1 = self.ch_resize(E5, E4, E3, E2, E1)
        # E4, E3, E2, E1 = E4.fill_(1).detach(), E3.fill_(1).detach(), E2.fill_(1).detach(), E1.fill_(1).detach()
        # print(E4.shape, E3.shape, E2.shape, E1.shape)
        E4, E3, E2, E1 = self.side_conv1(E4), self.side_conv2(E3), self.side_conv3(E2), self.side_conv4(E1)
        # print(E4.shape, E3.shape, E2.shape, E1.shape)
        # E5, E4, E3, E2, E1 = resize_tensors(E5, E4, E3, E2, E1)
        if E4.size()[2:] != E2.size()[2:]:
            E4 = F.interpolate(E4, size=E2.size()[2:], mode='bilinear')
        if E3.size()[2:] != E2.size()[2:]:
            E3 = F.interpolate(E3, size=E2.size()[2:], mode='bilinear')
        E5 = self.conv_block(E4, E3, E2)
        # print(E4.shape,E54.shape,E3.shape,E53.shape,E2.shape,E52.shape,E1.shape)
        E4 = torch.cat((E4, E5), 1)
        E3 = torch.cat((E3, E5), 1)
        E2 = torch.cat((E2, E5), 1)
        #print(E5.shape, E4.shape, E3.shape, E2.shape, E1.shape)
        E5 = F.relu(self.fuse5(E5), inplace=True)
        E4 = F.relu(self.fuse4(E4), inplace=True)
        E3 = F.relu(self.fuse3(E3), inplace=True)
        E2 = F.relu(self.fuse2(E2), inplace=True)
        E1 = F.relu(self.fuse1(E1), inplace=True)
        # print(E4.shape, E3.shape, E2.shape, E1.shape)
        #state1

        P5 = self.predtrans5(E5)
        ###################ES-CRF################START
        # Dmax5 = self.side_conv4x(E5)
        # Pmax5 = self.tffx(Dmax5)
        # Dmin5 = self.side_conv4n(E5)
        # Pmin5 = self.tffn(Dmin5)
        # Dmax5mk = self.mkx(F.interpolate(Dmax5,size=[88,88],mode='bilinear'))
        # Dmin5mk = self.mkn(F.interpolate(Dmin5,size=[88,88],mode='bilinear'))
        # E5e = get_edge(Pmax5,Pmin5)
        # P5e = E5e
        # D5e = Dmax5 - Dmin5
        # E5e1 = E5e.detach()
        D5e = self.get_edge_feature5(E5,E5,E5)
        E5e = self.edge_out5(F.interpolate(D5e,size=shape,mode='bilinear'))
        D5e = F.interpolate(D5e, size=shape, mode='bilinear')
        ###################ES-CRF################END

        E5e2 = E5e.detach()
        D4 = self.EEA5(E5, E4, E5e2, E5e2, D5e)
        D4 = F.interpolate(D4, size=E3.size()[2:], mode='bilinear')
        P4 = self.predtrans4(D4)

        #state2

        # Dmax4 = self.side_conv3x(Dmax5)
        # Dmax5 = torch.cat([F.interpolate(Dmax4,size=Dmax5mk.size()[2:]),self.mkx0(Dmax5mk)],dim=1)
        # Dmax4 = self.fuse(Dmax5)
        # Pmax4 = self.tffx(Dmax4)
        # Dmin4 = self.side_conv3n(Dmin5)
        # Dmin5 = torch.cat([F.interpolate(Dmin4,size=Dmin5mk.size()[2:]),self.mkn0(Dmin5mk)],dim=1)
        # Dmin4 = self.fuse(Dmin5)
        # Pmin4 = self.tffn(Dmin4)
        # Dmax4mk = self.mkx(F.interpolate(Dmax4,size=[88,88],mode='bilinear'))
        # Dmin4mk = self.mkn(F.interpolate(Dmin4,size=[88,88],mode='bilinear'))
        # E4e = get_edge(Pmax4,Pmin4)
        # P4e = E4e
        # D4e = Dmax4 - Dmin4
        # E4e1 = E4e.detach()
        # D5e = F.interpolate(D5e, size=D5e.size()[2:], mode='bilinear')
        D4e = self.get_edge_feature4(D4, D5e, D5e, D4, D4)
        E4e = self.edge_out4(F.interpolate(D4e,size=shape,mode='bilinear'))
        D4e = F.interpolate(D4e,size=shape,mode='bilinear')
        E4e2 = E4e.detach()
        D3 = self.EEA4(D4, E3, E4e2, E4e2, D4e)
        D3 = F.interpolate(D3, size=E2.size()[2:], mode='bilinear')
        P3 = self.predtrans3(D3)

        #state3

        # Dmax3 = self.side_conv2x(Dmax4)
        # Dmax4 = torch.cat([F.interpolate(Dmax3,size=Dmax4mk.size()[2:]),self.mkx0(Dmax4mk)],dim=1)
        # Dmax3 = self.fuse(Dmax4)
        # Pmax3 = self.tffx(Dmax3)
        # Dmin3 = self.side_conv2n(Dmin4)
        # Dmin4 = torch.cat([F.interpolate(Dmin3,size=Dmin4mk.size()[2:]),self.mkn0(Dmin4mk)],dim=1)
        # Dmin3 = self.fuse(Dmin4)
        # Pmin3 = self.tffn(Dmin3)
        # Dmax3mk = self.mkx(F.interpolate(Dmax3, size=[88,88], mode='bilinear'))
        # Dmin3mk = self.mkn(F.interpolate(Dmin3, size=[88,88], mode='bilinear'))
        # E3e = get_edge(Pmax3,Pmin3)
        # P3e = E3e
        # D3e = Dmax3 - Dmin3
        # E3e1 = E3e.detach()
        # D4e = F.interpolate(D4e, size=D3e.size()[2:], mode='bilinear')
        D3e = self.get_edge_feature3(D4e, D4e, D3, D3, D3)
        E3e = self.edge_out3(F.interpolate(D3e,size=shape,mode='bilinear'))
        E3e2 = E3e.detach()
        D3e = F.interpolate(D3e,size=shape,mode='bilinear')
        E3e2 = F.interpolate(E3e2, size=[22,22], mode='bilinear')
        D2 = self.EEA3(D3, E2, E3e2, E3e2, D3e)
        D2 = F.interpolate(D2, size=E1.size()[2:], mode='bilinear')
        P2 = self.predtrans2(D2)

        #state4

        # Dmax2 = self.side_conv1x(Dmax3)
        # Dmax3 = torch.cat([F.interpolate(Dmax2,size=Dmax3mk.size()[2:]),self.mkx0(Dmax3mk)],dim=1)
        # Dmax2 = self.fuse(Dmax3)
        # Pmax2 = self.tffx(Dmax2)
        # Dmin2 = self.side_conv1n(Dmin3)
        # Dmin3 = torch.cat([F.interpolate(Dmin2,size=Dmin3mk.size()[2:]),self.mkn0(Dmin3mk)],dim=1)
        # Dmin2 = self.fuse(Dmin3)
        # Pmin2 = self.tffn(Dmin2)
        # E2e = get_edge(Pmax2,Pmin2)
        # P2e = E2e
        # D2e = Dmax2 - Dmin2
        # E2e1 = E2e.detach()
        # D3e = F.interpolate(D3e, size=D2e.size()[2:], mode='bilinear')
        D2e = self.get_edge_feature2(D3e, D3e, D2, D2, D2)
        E2e = self.edge_out2(F.interpolate(D2e,size=shape,mode='bilinear'))
        E2e2 = E2e.detach()
        E1 = F.interpolate(E1,size=shape,mode='bilinear')
        D1 = self.EEA2(D2, E1, E2e2, E2e2, D2e)
        P1 = self.predtrans1(D1)



        P1 = F.interpolate(P1, size=shape, mode='bilinear')
        P2 = F.interpolate(P2, size=shape, mode='bilinear')
        P3 = F.interpolate(P3, size=shape, mode='bilinear')
        P4 = F.interpolate(P4, size=shape, mode='bilinear')
        P5 = F.interpolate(P5, size=shape, mode='bilinear')


        # Pmax2 = F.interpolate(Pmax2, size=shape, mode='bilinear')
        # Pmax3 = F.interpolate(Pmax3, size=shape, mode='bilinear')
        # Pmax4 = F.interpolate(Pmax4, size=shape, mode='bilinear')
        # Pmax5 = F.interpolate(Pmax5, size=shape, mode='bilinear')
        #
        # Pmin2 = F.interpolate(Pmin2, size=shape, mode='bilinear')
        # Pmin3 = F.interpolate(Pmin3, size=shape, mode='bilinear')
        # Pmin4 = F.interpolate(Pmin4, size=shape, mode='bilinear')
        # Pmin5 = F.interpolate(Pmin5, size=shape, mode='bilinear')

        E5e = F.interpolate(E5e, size=shape, mode='bilinear')
        E4e = F.interpolate(E4e, size=shape, mode='bilinear')
        E3e = F.interpolate(E3e, size=shape, mode='bilinear')
        E2e = F.interpolate(E2e, size=shape, mode='bilinear')
        #
        # P5e = F.interpolate(P5e, size=shape, mode='bilinear')
        # P4e = F.interpolate(P4e, size=shape, mode='bilinear')
        # P3e = F.interpolate(P3e, size=shape, mode='bilinear')
        # P2e = F.interpolate(P2e, size=shape, mode='bilinear')

        # x_g = F.interpolate(x_g, size=shape, mode='bilinear')


        F1 = get_images(D1)

        F2 = get_images(D2)

        F3 = get_images(D3)

        F4 = get_images(D4)

        return [P5, P4, P3, P2, P1], [E5e, E4e, E3e, E2e], [F1, F2, F3, F4]

    def initialize(self):
        weight_init(self)
