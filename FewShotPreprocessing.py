
import os
import os.path as osp
from PIL import Image
    
PATH='../Fewshot/Fewshot/'
classes= os.listdir(PATH)
trainp='../Fewshot/train/'
valp='../Fewshot/val/'
testp='../Fewshot/test/'

for classv in classes:
