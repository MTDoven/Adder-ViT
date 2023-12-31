'''
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math


_help_words = \
""" 
reference: https://github.com/LingYeAI/AdderNetCUDA
Use this adder_cuda library to accelarate your codes
You may meet some strange problem when install this library. Just be patient
"""
try:
    import adder_cuda
except ImportError:
    raise ImportError("Cannot import adder_cuda.\n"+_help_words)



def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    # n = n_x * n_filters if depth_wise else n_x
    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    out = adder.apply(W_col,X_col)
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    return out

class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col) 
        Co = W_col.size(0)
        HoWoN = X_col.size(1)
        output = torch.zeros((Co, HoWoN),device=W_col.device)
        adder_cuda.ADDER_CONV(X_col, W_col, output)
        ###############test code################
        # ground_truth = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1)
        # sub = output - ground_truth
        # print("check result:", torch.sum(sub), torch.var(sub), torch.max(sub), torch.min(sub))
        ###############test end#################
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = torch.zeros_like(W_col,device=W_col.device)
        grad_X_col = torch.zeros_like(X_col,device=X_col.device)
        adder_cuda.ADDER_BACKWARD(grad_output, X_col, W_col, grad_W_col, grad_X_col)
        ###############test code################
        # gt_w = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        # sub = grad_W_col - gt_w
        # print("check grad_W_col result:", torch.sum(sub), torch.var(sub), torch.max(sub), torch.min(sub))
        # gt_x = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        # sub = grad_X_col - gt_x
        # print("check grad_X_col result:", torch.sum(sub), torch.var(sub), torch.max(sub), torch.min(sub))
        ###############test end#################
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        return grad_W_col, grad_X_col


class adder2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias = False, depth_wise = False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.depth_wise = depth_wise
        if depth_wise:
            self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,1,kernel_size,kernel_size)))
        else:
            self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))
    def forward(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return output

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return ConvBase(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride = stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
         
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ConvBase(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes = self.inplanes, planes = planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)
        return x.view(x.size(0), -1)



    
    
OperateBase = None
ConvBase = adder2d
def get_model(operate_base=None, conv_base=None, **kwargs):
    global OperateBase
    global ConvBase
    OperateBase = operate_base
    ConvBase = adder2d if conv_base is None else conv_base
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)