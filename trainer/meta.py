


""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, ensure_path
from tensorboardX import SummaryWriter
from dataloader.samplers import CategoriesSampler
from utils.metrics import eval_metrics
from utils.losses import FocalLoss,CE_DiceLoss,LovaszSoftmax
from dataloader.mdataset_loader import mDatasetLoader as mDataset
from torchvision import transforms

class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        save_image_dir='../results/'
        if not osp.exists(save_image_dir):
            os.mkdir(save_image_dir)        
        
        log_base_dir = '../logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
            '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
            '_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) + \
            '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
            '_stepsize' + str(args.step_size) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        args.save_image_dir=save_image_dir
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load meta-train set
        self.trainset = mDataset('meta', self.args)
        self.train_sampler = CategoriesSampler(self.trainset.labeln, self.args.num_batch, self.args.way, self.args.shot + self.args.train_query,self.args.shot)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=8, pin_memory=True)

        # Load meta-val set
        self.valset = mDataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.labeln, self.args.num_batch, self.args.way, self.args.shot + self.args.val_query,self.args.shot)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)
        
        # Build meta-transfer learning model
        self.model = MtlLearner(self.args)
        self.FL=FocalLoss()
        self.CD=CE_DiceLoss()
        self.LS=LovaszSoftmax()
        
        # Set optimizer 
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
            {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)        
        
        # load pretrained model
        self.model_dict = self.model.state_dict()