# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:45:48 2020

@author: derek
"""
import os
import numpy as np
import random 
import math
random.seed = 0

import cv2
from PIL import Image
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms,models
from torchvision.transforms import functional as F

from detrac_localization_dataset import Localize_Dataset


# define ResNet18 based network structure

class ResNet_Localizer(nn.Module):
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_Localizer, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        self.feat = models.resnet18(pretrained=True)

        # get size of some layers
        start_num = self.feat.fc.out_features
        mid_num = int(np.sqrt(start_num))
        
        cls_out_num = 13 # car or non-car (for now)
        reg_out_num = 4 # bounding box coords
        embed_out_num = 128
        
        # define classifier
        self.classifier = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,cls_out_num,bias = True),
                          nn.Softmax(dim = 1)
                          )
        
        # define regressor
        self.regressor = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,reg_out_num,bias = True),
                          nn.ReLU()
                          )
        
        self.embedder = nn.Sequential(
                  nn.Linear(start_num,start_num // 3,bias=True),
                  nn.ReLU(),
                  nn.Linear(start_num // 3,embed_out_num,bias = True),
                  nn.ReLU()
                  )

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        features = self.feat(x)
        cls_out = self.classifier(features)
        reg_out = self.regressor(features)
        #embed_out = self.embedder(features)
        #out = torch.cat((cls_out, reg_out), 0) # might be the wrong dimension
        
        return cls_out,reg_out