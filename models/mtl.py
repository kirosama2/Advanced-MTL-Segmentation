

""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_mtl import UNetMtl
from utils.losses import FocalLoss,CE_DiceLoss,LovaszSoftmax

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vars = nn.ParameterList()
        self.wt = nn.Parameter(torch.ones([self.args.way,self.args.num_classes,3,3]))
        self.vars.append(self.wt)
        self.bias = nn.Parameter(torch.zeros([self.args.way]))
        self.vars.append(self.bias)
        self.norm1 = nn.BatchNorm2d(self.args.num_classes)
        self.norm2 = nn.ReLU(inplace=True)
        
    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        norm1=self.norm1
        norm2=self.norm2
        wt = the_vars[0]
        bias = the_vars[1]
        net=F.conv2d(norm2(norm1(input_x)),wt,bias,stride=1,padding=1)
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta'):
        super().__init__()
        self.args = args
        self.mode = mode