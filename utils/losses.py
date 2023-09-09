import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    if(torch.cuda.is_available()):
        one_hot=one_hot.cuda()
    targe