


""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import torch
import numpy as np
from utils.metrics import eval_metrics
from torch.utils.data import DataLoader
from models.mtl import MtlLearner
from tensorboardX import SummaryWriter
from dataloader.samplers import CategoriesSampler 
from utils.losses import FocalLoss,CE_DiceLoss,LovaszSoftmax
from utils.misc import Averager, Timer, ensure_path
from dataloader.dataset_loader import DatasetLoader as Dataset
from dataloader.mdataset_loader import mDatasetLoader as mDataset

#torch.cuda.set_device(1)
class PreTrainer(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = '../logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):