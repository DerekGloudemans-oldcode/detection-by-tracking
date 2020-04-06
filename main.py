"""
Created on Sat Mar  7 15:45:48 2020

@author: derek
"""

#%% 0. Imports 
import os
import numpy as np
import random 
import math
random.seed = 0

import cv2
from PIL import Image
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms,models
from torchvision.transforms import functional as F
import matplotlib.pyplot  as plt

from detrac_files.detrac_train_localizer import ResNet_Localizer, load_model, class_dict
from pytorch_yolo_v3.yolo_detector import Darknet_Detector


if __name__ == "__main__":
    #%% 1. Set up models, etc

    yolo_checkpoint =   "/home/worklab/Desktop/checkpoints/yolo/yolov3.weights"
    resnet_checkpoint = "/home/worklab/Desktop/checkpoints/detrac_localizer/best_ResNet18_epoch4.pt"
    track_directory =   ""
    det_step = 10
    
    
    # get CNNs
    try:
        detector
        localizer
    except:
        detector = Darknet_Detector(
            'pytorch_yolo_v3/cfg/yolov3.cfg',
            yolo_checkpoint,
            'pytorch_yolo_v3/data/coco.names',
            'pytorch_yolo_v3/pallete'
            )
        
        localizer = ResNet_Localizer()
        localizer,_,_,all_metrics = load_model(resnet_checkpoint, localizer, None)
    
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()   
    
    localizer = localizer.to(device)
    print("Detector and Localizer on {}.".format(device))
    
     
    
