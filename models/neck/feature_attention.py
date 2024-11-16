import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

class ScaleChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        print(self.avgpool)
        self.fc1 = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.fc2 = nn.Conv2d(out_planes, num_features, 1, bias=False)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = self.avgpool(x)
        global_x = self.fc1(global_x)
        global_x = F.relu(self.bn(global_x))
        global_x = self.fc2(global_x)
        global_x = F.softmax(global_x, 1)
        return global_x

class ScaleChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelSpatialAttention, self).__init__()
        self.channel_wise = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes , 1, bias=False),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, in_planes, 1, bias=False)
        )
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # global_x = self.avgpool(x)
        #shape Nx4x1x1
        global_x = self.channel_wise(x).sigmoid()
        #shape: NxCxHxW
        global_x = global_x + x
        #shape:Nx1xHxW
        x = torch.mean(global_x, dim=1, keepdim=True)
        global_x = self.spatial_wise(x) + global_x
        global_x = self.attention_wise(global_x)
        return global_x

class ScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleSpatialAttention, self).__init__()
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid() 
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = torch.mean(x, dim=1, keepdim=True)
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        return global_x
    
class LVCScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(LVCScaleSpatialAttention, self).__init__()
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid() 
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        self.lvc = LVCBlock(in_channels=in_planes, out_channels=out_planes)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = torch.mean(x, dim=1, keepdim=True)
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        y= self.lvc(x)
        return global_x * y 

class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels, inter_channels , out_features_num=4, attention_type='scale_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels=in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.type = attention_type
        if self.type == 'scale_spatial':
            self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels//4, out_features_num)
        elif self.type == 'scale_channel_spatial':
            self.enhanced_attention = ScaleChannelSpatialAttention(inter_channels, inter_channels // 4, out_features_num)
        elif self.type == 'scale_channel':
            self.enhanced_attention = ScaleChannelAttention(inter_channels, inter_channels//2, out_features_num)
        elif self.type == 'lvc_scale_spatial':
            self.enhanced_attention = LVCScaleSpatialAttention(inter_channels, inter_channels//4, out_features_num)

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    def forward(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        assert len(features_list) == self.out_features_num
        if self.type not in ['scale_channel_spatial', 'scale_spatial']:
            shape = features_list[0].shape[2:]
            score = F.interpolate(score, size=shape, mode='bilinear')
        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i+1] * features_list[i])
        return torch.cat(x, dim=1)
    
    
class LVCBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, num_codes=64, channel_ratio=0.25, base_channel=64):
        super(LVCBlock, self).__init__()
        self.out_channels = out_channels
        self.num_codes = num_codes
        # num_codes = 64

        self.conv_1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, res_conv=True, stride=1)

        self.LVC = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Encoding(in_channels=in_channels, num_codes=self.num_codes),
            nn.BatchNorm1d(self.num_codes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_1(x, return_x_2=False)
        en = self.LVC(x)
        gam = self.fc(en)
        b, in_channels, _, _ = x.size()
        y = gam.view(b, in_channels, 1, 1)
        # x = F.relu_(x + x * y) 
        return y
    
class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)   

# LVC
class Encoding(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.in_channels, self.num_codes = in_channels, num_codes
        # num_codes = 64
        std = 1. / ((num_codes * in_channels)**0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        # [num_codes]
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, in_channels = codewords.size()
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))

        # ---处理codebook (num_code, c1)
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))

        # 把scale从1, num_code变成   batch, c2, N, num_codes
        reshaped_scale = scale.view((1, 1, num_codes))  # N, num_codes

        # ---计算rik = z1 - d  # b, N, num_codes
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, in_channels = codewords.size()

        # ---处理codebook
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        b = x.size(0)

        # ---处理特征向量x b, c1, N
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))

        #变换rei  b, N, num_codes,-
        assignment_weights = assignment_weights.unsqueeze(3)  # b, N, num_codes,

        # ---开始计算eik,必须在Rei计算完之后
        encoded_feat = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()

        # [batch_size, height x width, channels]
        x = x.view(b, self.in_channels, -1).transpose(1, 2).contiguous()

        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)

        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat
    
#  1*1 3*3 1*1
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        expansion = 4
        c = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0, bias=False)  # [64, 256, 1, 1]
        self.bn1 = norm_layer(c)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) #if x_t_r is None else self.conv2(x + x_t_r)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x
