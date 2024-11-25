# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 10:29
# @Author  : zhoujun
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from .feature_attention import ScaleFeatureSelection

from typing import Tuple

from models.basic import ConvBnRelu


class FPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        self.use_attention = kwargs.get('use_attention',False)
        self.attention_type = kwargs.get('attention_type','scale_spatial')

        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels*4, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels*4, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels*4, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels*4, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_p5 = ConvBnRelu(inner_channels*4, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p4 = ConvBnRelu(inner_channels*4, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels*4, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels*4, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )
        if self.use_attention:
            self.concat_attention = ScaleFeatureSelection(inner_channels*4, inner_channels, attention_type=self.attention_type)
        self.out_channels = self.conv_out

    def forward(self, x:Tuple[Tensor,Tensor,Tensor,Tensor]):
        c2,c3,c4,c5=x
        # Top-down
        p5 = self.reduce_conv_c5(c5)
        p4 = self.reduce_conv_c4(c4)
        p3 = self.reduce_conv_c3(c3)
        p2 = self.reduce_conv_c2(c2)

        p4 = self._upsample_add(p5, p4)
        p3 = self._upsample_add(p4, p3)
        p2 = self._upsample_add(p3, p2)

        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        fuse = torch.cat((p5, p4, p3, p2), dim=1)
        if self.use_attention:
            fuse = self.concat_attention(fuse, [p5, p4, p3, p2])
        return fuse
