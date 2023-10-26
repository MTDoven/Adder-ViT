import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
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

try:
    from .cuda import *
except ImportError:
    from cuda import *



class operator_quick(Function):
    """
    This class follows 'https://github.com/LingYeAI/AdderNetCUDA' to make the train and inference of AdderNet quicker.
    It has been checked that it completely follows the AdderNet, which use FP backward on both X and W.
    """
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x,w) 
        output = torch.zeros((x.size(0), w.size(1)),device=x.device)
        adder_cuda.ADDER_CONV(w, x, output) # w,x is the right order
        return output
    @staticmethod
    def backward(ctx,grad_output):
        x,w = ctx.saved_tensors
        grad_x = torch.zeros_like(x,device=x.device)
        grad_w = torch.zeros_like(w,device=w.device)
        adder_cuda.ADDER_BACKWARD(grad_output, w, x, grad_x, grad_w) # what's the fucking order?
        grad_w = grad_w/grad_w.norm(p=2).clamp(min=1e-12)*math.sqrt(w.size(1)*w.size(0))/5
        return grad_x, grad_w

    

class operator_quicker(Function):
    """
    This class use cuda functions from 'cuda.py', which uses pycuda to compile a kernal function of cuda. 
    In those cuda codes, I redefine the FP backward, which is illustrate in README.md.
    Attention: The "quicker" is not as quick as the "quick" mostly.
    """
    @staticmethod
    def forward(ctx, x, w, dims):
        # save for backward
        x_shape_tensor, shapes_tensor, width_cuda, length_cuda, outdim_cuda = dims
        ctx.save_for_backward(x,w, x_shape_tensor, shapes_tensor, width_cuda, length_cuda, outdim_cuda)
        width, length, outdim = width_cuda.item(), length_cuda.item(), outdim_cuda.item()
        # calculate
        o = torch.empty(length, outdim, device="cuda")
        x = x.contiguous()
        w = w.contiguous()
        ADD(Holder(x), Holder(w), 
            Holder(width_cuda), Holder(o),
            grid=(length,1,1), block=(outdim,1,1))
        # reshape for out
        o = o.view(*x_shape_tensor[:-1], outdim)
        return o
    @staticmethod
    def backward(ctx, grad_output):
        x,w, x_shape_tensor, shapes_tensor, width_cuda, length_cuda, outdim_cuda = ctx.saved_tensors
        width, length, outdim = width_cuda.item(), length_cuda.item(), outdim_cuda.item()
        grad_output = grad_output.view(length, outdim).contiguous()
        # calculate
        grad_w = torch.zeros(width, outdim, device="cuda")
        #print(x.shape, w.shape, grad_output.shape)
        ADD_BACKWARD_W(Holder(x), Holder(w), Holder(grad_output),
                       Holder(length_cuda), Holder(grad_w),
                       grid=(width,1,1), block=(outdim,1,1))
        grad_x = torch.zeros(length, width, device="cuda")
        ADD_BACKWARD_X(Holder(x), Holder(w), Holder(grad_output),
                       Holder(outdim_cuda), Holder(grad_x),
                       grid=(length,1,1), block=(width,1,1))
        # grad_x = grad_x.view(*x_shape_tensor)
        # adaptive lr
        grad_w = grad_w/grad_w.norm(p=2, dim=[-1,-2], keepdim=True).clamp(min=1e-12)*math.sqrt(w.size(-1)*w.size(-2))/5
        return grad_x, grad_w, None
    
    

class operator_stable(Function):
    """
    This class use pytorch function to finish the 'add' operation, but it's really slow and memory-hungry. 
    It mostly used to check if there's a bug with cuda codes. Never try to use these codes directly for model training.
    """
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x.unsqueeze(-1),w.unsqueeze(-3))
        return -(x.unsqueeze(-1)-w.unsqueeze(-3)).abs().sum(-2)
    @staticmethod
    def backward(ctx, grad_output):
        x,w = ctx.saved_tensors
        temp = x-w    
        grad_x = (-(4*temp).clamp(-0.25,0.25)*grad_output.unsqueeze(-2)).sum(-1)
        grad_w = (temp.clamp(-2,2)*grad_output.unsqueeze(-2)).sum(-3)
        grad_w = grad_w/grad_w.norm(p=2, dim=[-1,-2], keepdim=True).clamp(min=1e-12)*math.sqrt(w.size(-1)*w.size(-2))/5
        return grad_x, grad_w

    
      

    
class AdderBase(nn.Module):
    """
    This class can be used in the same way as nn.Linear 
    """
    def __init__(self, in_channel, out_channel, bias=True, norm=True, w=None):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.weight = nn.Parameter(nn.init.normal_(torch.randn(in_channel,out_channel))) if w is None else nn.Parameter(w.clone())
        self.multi_bias_in = nn.Parameter(torch.ones(1,in_channel, device="cuda"))
        self.multi_bias_out = nn.Parameter(torch.ones(1,out_channel, device="cuda"))
        self.norm = nn.LayerNorm(out_channel) if norm else nn.Identity()
    def forward(self, x):
        self.batch_shape = x.shape[:-1]
        assert x.shape[-1]==self.weight.shape[0], \
            "Shapes do not match. With x:{} and param:{}".format(x.shape, self.weight.shape)
        if self.out_channel<=1024 and self.in_channel<=1024:
            print("quicker")
            x_shape_tensor = torch.tensor(x.shape)
            width_cuda = torch.tensor(x.shape[-1], device="cuda")
            length_cuda = torch.tensor(x.contiguous().view(-1, width_cuda.item()).shape[0], device="cuda")
            outdim_cuda = torch.tensor(self.weight.shape[1], device="cuda")
            shapes_tensor = torch.tensor([width_cuda.item(), length_cuda.item(), outdim_cuda.item()])
            self.dims = (x_shape_tensor, shapes_tensor, width_cuda, length_cuda, outdim_cuda)
            self.forward = self.__forward_quicker
        elif self.in_channel%16==0 and self.out_channel%16==0\
                and x.contiguous().view(-1, self.in_channel).shape[0]%16==0:
            print("quick")
            self.forward = self.__forward_quick
        else: # cannot accclerate
            print("slow")
            self.forward = self.__forward_stable
        return self.forward(x)
    def __forward_quicker(self,x):
        x = x * self.multi_bias_in
        x = x.contiguous().view(-1, self.in_channel)
        x = operator_quicker.apply(x, self.weight, self.dims)
        x = x.view(*self.batch_shape, self.out_channel)
        x = x * self.multi_bias_out
        return self.norm(x)
    def __forward_quick(self,x):
        x = x * self.multi_bias_in
        x = x.contiguous().view(-1, self.in_channel)
        x = operator_quick.apply(x, self.weight)
        x = x.view(*self.batch_shape, self.out_channel)
        x = x * self.multi_bias_out
        return self.norm(x)
    def __forward_stable(self,x):
        x = x * self.multi_bias_in
        x = operator_stable.apply(x, self.weight)
        x = x * self.multi_bias_out
        return self.norm(x)
    @staticmethod
    def operate_function(x,y):
        o = multi_head_adder.apply(x,y)
        return o

class multi_head_adder(Function):
    """
    This multi_head_adder.apply can be used the same as torch.matmul.
    It was designed for multi head attention computation.
    """
    @staticmethod
    def forward(ctx, x, y):
        #ctx.save_for_backward(x, y) 
        x_shape, y_shape = x.shape, y.shape
        x = x.contiguous().view(-1, x_shape[-2], x_shape[-1])
        y = y.contiguous().view(-1, y_shape[-2], y_shape[-1])
        shapes = torch.cat((torch.tensor(x.shape), torch.tensor(y.shape)), dim=0).cuda()
        ctx.save_for_backward(x, y, torch.tensor(x_shape), torch.tensor(y_shape), shapes) 
        o = torch.empty(x.shape[0], x_shape[-2], y_shape[-1], device="cuda")
        MH_ADD(Holder(x), Holder(y), Holder(shapes), Holder(o),
               grid=(x_shape[-3],x_shape[-2],1), block=(y_shape[-1],1,1))
        o = o.view(*x_shape[:-2], o.shape[-2], o.shape[-1])
        return o
    @staticmethod
    def backward(ctx, grad_output):
        x, y, x_shape, y_shape, shapes = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        grad_y = torch.empty_like(y)
        x = x.contiguous()
        y = y.contiguous()
        grad_output = grad_output.contiguous()
        MH_ADD_BACKWARD_Q(Holder(x), Holder(y), Holder(grad_output), Holder(shapes), Holder(grad_x),
                          grid=(x_shape[-3].item(),x_shape[-2].item(),1), block=(x_shape[-1].item(),1,1))
        MH_ADD_BACKWARD_K(Holder(x), Holder(y), Holder(grad_output), Holder(shapes), Holder(grad_y),
                          grid=(y_shape[-3].item(),y_shape[-2].item(),1), block=(y_shape[-1].item(),1,1))
        grad_x = grad_x.view(*x_shape)
        grad_y = grad_y.view(*y_shape)
        grad_x = grad_x/grad_x.norm(p=2).clamp(min=1e-12)*math.sqrt(x.size(-1)*x.size(-2))/50
        grad_y = grad_y/grad_y.norm(p=2).clamp(min=1e-12)*math.sqrt(y.size(-1)*y.size(-2))/50
        return grad_x, grad_y
    # # stable python function
    # @staticmethod 
    # def backward(ctx, grad_output):
    #     x,y = ctx.saved_tensors
    #     ##### to optimize #####
    #     temp = (x.unsqueeze(-1)-y.unsqueeze(-3))*grad_output.unsqueeze(-2) 
    #     grad_x = -torch.sign(temp).sum(-1)
    #     grad_y = torch.sign(temp).sum(-3)
    #     return grad_x, grad_y

class AdderConv(nn.Module):
    """
    This class can be used the same as nn.Conv2d
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False, groups=False):
        super(AdderConv, self).__init__()
        self.stride = stride
        self.padding = padding
        if groups:
            self.param = torch.nn.Parameter(nn.init.normal_(torch.randn(out_channel,1,kernel_size,kernel_size)))
        else:
            self.param = torch.nn.Parameter(nn.init.normal_(torch.randn(out_channel,in_channel,kernel_size,kernel_size)))
        self.batchnorm = nn.BatchNorm2d(out_channel, affine=True)
    def forward(self, x):
        x = self.adder2d_function(x, self.param, self.stride, self.padding)
        return self.batchnorm(x).contiguous()
    # adder2d_function
    @staticmethod
    def adder2d_function(X, W, stride=1, padding=0):
        # get dims
        n_filters, d_filter, h_filter, w_filter = W.size()
        n_x, d_x, h_x, w_x = X.size()
        h_out = int((h_x - h_filter + 2 * padding) / stride + 1)
        w_out = int((w_x - w_filter + 2 * padding) / stride + 1)
        # reshape x and w
        X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
        X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1).transpose(-1,-2)
        W_col = W.view(n_filters, -1).transpose(-1,-2)
        # get out 
        out = operator_quicker.apply(X_col, W_col).transpose(-1,-2)
        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2)
        return out

    

OperateBase = AdderBase
ConvBase = AdderConv