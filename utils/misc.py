
""" Additional utility functions. """
import os
import time
import pprint
import torch
import numpy as np
import torch.nn.functional as F

def ensure_path(path):
    """The function to make log path.
    Args:
      path: the ge