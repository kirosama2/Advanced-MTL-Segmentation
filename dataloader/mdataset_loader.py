

""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

#For FewShot
class mDatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args, train_aug=False):
        # Set the path according to train, val and test
        self.args=args
        if setname=='meta':
            THE_PATH = osp.join(args.mdataset_dir, 'train/images')
            THE_PATHL = osp.join(args.mdataset_dir, 'train/labels/')
        elif setname=='val':
            THE_PATH = osp.join(args.mdataset_dir, 'val/images/')
            THE_PATHL = osp.join(args.mdataset_dir, 'val/labels/')
        elif setname=='test':
            THE_PATH = osp.join(args.mdataset_dir, 'test/images/')
            THE_PATHL = osp.join(args.mdataset_dir, 'test/labels/')            
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label       
        
        # exit()    
        data = []
        label = []
        labeln=[]
        
        # Get the classes' names
        folders = os.listdir(THE_PATH)      
        
        for idx, this_folder in enumerate(folders):
            
            imf=osp.join(THE_PATH,this_folder)
            imf=imf+'/'
            lbf=osp.join(THE_PATHL,this_folder)
            lbf=lbf+'/'
            
            this_folder_images = os.listdir(imf)
            for im in this_folder_images:
                data.append(osp.join(imf, im))
                label.append(osp.join(lbf, im[:-3]+'png'))    
                labeln.append(idx)
            
        # Set data, label and class number to be accessable from outside
        self.data = data