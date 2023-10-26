from torch import nn
import torch


class EuclidBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.randn(in_channel,out_channel)))
        self.bias = nn.Parameter(torch.randn(1,out_channel)-out_channel-in_channel)
        self.norm = nn.LayerNorm(out_channel, elementwise_affine=True) if norm else nn.Identity()
    def forward(self, x):
        t1 = torch.square(x).sum(-1, keepdim=True)
        t3 = torch.matmul(x, self.weight)
        return self.norm(t3.add_(t1).add_(self.bias))
    @staticmethod
    def operate_function(x,y):
        t1 = torch.square(x).sum(-1, keepdim=True)
        t2 = torch.square(y).sum(-2, keepdim=True)
        t3 = 2 * torch.matmul(x,y)
        return t3.add_(t2).add_(t1)
    
    
class EuclidConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False, groups=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.param = torch.nn.Parameter(nn.init.kaiming_normal_(torch.randn(out_channel,in_channel,kernel_size,kernel_size))) if groups \
                else torch.nn.Parameter(nn.init.kaiming_normal_(torch.randn(out_channel,in_channel,kernel_size,kernel_size)))
        self.batchnorm = nn.BatchNorm2d(out_channel, affine=True)
    def forward(self, x):
        x = self.conv2d_function(x, self.param, self.stride, self.padding)
        return self.batchnorm(x).contiguous()
    # conv2d_function
    @staticmethod
    def conv2d_function(X, W, stride=1, padding=0):
        # get dims
        n_filters, d_filter, h_filter, w_filter = W.size()
        n_x, d_x, h_x, w_x = X.size()
        h_out = int((h_x - h_filter + 2 * padding) / stride + 1)
        w_out = int((w_x - w_filter + 2 * padding) / stride + 1)
        # reshape x and w
        X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), 
                    h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
        X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1).transpose(-1,-2)
        W_col = W.view(n_filters, -1).transpose(-1,-2)
        # get out 
        out = EuclidBlock.operate_function(X_col, W_col).transpose(-1,-2)
        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2)
        return out
    

OperateBase = EuclidBlock
ConvBase = EuclidConv