

""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='UNet', choices=['UNet']) # The network architecture
    parser.add_argument('--dataset', type=str, default='COCO', choices=['COCO','Fewshot']) # Dataset
    parser.add_argument('--phase', type=str, default='meta_eval', choices=['pre_train', 'meta_train', 'meta_eval']) # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', default='1') # GPU id
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/COCO/') # Dataset folder

    # Parameters for meta-train phase    
    parser.add_argument('--mdataset_dir', type=str, default='../Datasets/Fewshot/') # Dataset folder
    parser.add_argument('--max_epoch', type=int, default=200) # Epoch number for meta-train phase
    parser.add_argument('--num_batch', type=int, default=50) # The number for different tasks used for meta-train
    parser.add_argument('--num_classes', type=int, default=5)# Total number of pre-labelled classes 
    parser.add_argument('--way', type=int, default=2) # Way number, how many classes in a task
    parser.add_argument('--shot', type=int, default=3) # Shot number, how many samples for one class in a task
    parser.add_argument('--teshot', type=int, def