


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
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
            str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load pretrain set
        self.trainset = Dataset('train', self.args)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # Load pre-val set
        self.valset = mDataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.labeln,self.args.num_batch , self.args.way, self.args.shot + self.args.val_query,self.args.shot)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)


        # Build pretrain model
        self.model = MtlLearner(self.args, mode='train')
        print(self.model)
        
        '''
        if self.args.pre_init_weights is not None:
            self.model_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
            print(pretrained_dict.keys())
            self.model_dict.update(pretrained_dict)
            self.model.load_state_dict(self.model_dict)   
        '''
        
        self.FL=FocalLoss()
        self.CD=CE_DiceLoss()
        self.LS=LovaszSoftmax()
        # Set optimizer 
        # Set optimizer 
        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr}], \
                momentum=self.args.pre_custom_momentum, nesterov=True, weight_decay=self.args.pre_custom_weight_decay)

            # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, \
            gamma=self.args.pre_gamma)        

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  