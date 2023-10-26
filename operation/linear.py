from torch import nn
import torch
    
class MultiBlock(nn.Linear):
    @staticmethod
    def operate_function(x,y):
        x = torch.matmul(x,y)
        return x
    
OperateBase = MultiBlock
ConvBase = nn.Conv2d