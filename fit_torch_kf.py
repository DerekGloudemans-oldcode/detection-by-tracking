#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:20:41 2020

@author: worklab
"""

import torch
import numpy
import time
import random
random.seed  = 0

from detrac_files.detrac_tracking_dataset import Track_Dataset
from torch_kf import Torch_KF
from torch.utils.data import DataLoader

# need to make the set of matrices that are optimized over variable
# need to make variable rollout and pre-rollout lengths
# need to have variable learning rate
# need to have IoU metric of evaluation
# need to report time metric for each round of evaluations
# need to have variable batch size 


def iou(a,b):
    """
    Description
    -----------
    Calculates intersection over union for all sets of boxes in a and b

    Parameters
    ----------
    a : a torch of size [batch_size,4] of bounding boxes.
    b : a torch of size [batch_size,4] of bounding boxes.

    Returns
    -------
    mean_iou - float between [0,1] with average iou for a and b
    """
    
    area_a = a[:,2] * a[:,2] * a[:,3]
    area_b = b[:,2] * b[:,2] * b[:,3]
    
    minx = torch.max(a[:,0]-a[:,2]/2, b[:,0]-b[:,2]/2)
    maxx = torch.min(a[:,0]+a[:,2]/2, b[:,0]+b[:,2]/2)
    miny = torch.max(a[:,1]-a[:,2]*a[:,3]/2, b[:,1]-b[:,2]*b[:,3]/2)
    maxy = torch.min(a[:,1]+a[:,2]*a[:,3]/2, b[:,1]+b[:,2]*b[:,3]/2)
    zeros = torch.zeros(minx.shape,dtype = float)
    
    intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
    union = area_a + area_b - intersection
    mean_iou = (torch.mean(intersection) + 1e-06) / (torch.mean(union) + 1e-06)
    
    return mean_iou

# define parameters
lr        = 0.01  # learning rate
e         = 1e-06 # gradient estimation step size
b         = 2000 # batch size
n_pre     = 3     # number of frames used to initially tune tracker
n_post    = 5     # number of frame rollouts used to evaluate tracker
tune_kf   = ['P','Q','R','H'] # which KF parameters to optimize over

 


try:
    loader
except:
    # initialize dataset
    dataset = Track_Dataset("/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3")
    
    # 3. create training params
    params = {'batch_size' : b,
              'shuffle'    : True,
              'num_workers': 0,
              'drop_last'  : True
              }
    
    # returns batch_size x (n_pre + n_post) x 4 tensor
    loader = DataLoader(dataset, **params)


# create initial values for each matrix
P = torch.zeros((7,7))
Q = torch.zeros((7,7))
R = torch.zeros((4,4))
F = torch.zeros((7,7))
H = torch.zeros((4,7))
kf_params = {
        "P":P,
        "Q":Q,
        "R":R,
        "F":F,
        "H":H
        }

delete_keys = []
for key in kf_params:
    if key not in tune_kf:
        delete_keys.append(key)
delete_keys.reverse()
for key in delete_keys:
    del kf_params[key]
    
    
temp = next(iter(loader))
a1 = temp[:,0,:]
a2 = temp[:,1,:]
print(iou(a1,a2).item())

