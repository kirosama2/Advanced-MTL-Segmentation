

""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

#For COCO
class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args, train_aug=False):
        self.args=args
        # Set the path according to train, val and test        
        if setname=='train':
            THE_PATH = osp.join(args.dataset_dir, 'train/')
            THE_PATHL = osp.join(args.dataset_dir, 'labels/train/')
        elif setname=='val':
            THE_PATH = osp.join(args.dataset_dir, 'val/')
            THE_PATHL = osp.join(args.dataset_dir, 'labels/val/')
        else: