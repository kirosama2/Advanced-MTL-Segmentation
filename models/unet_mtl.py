
""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv2d_mtl import Conv2dMtl, ConvTranspose2dMtl