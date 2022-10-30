

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
    parser.add_argument('-