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



# define parameters
lr        = 0.01  # learning rate
e         = 1e-06 # gradient estimation step size
b         = 2000  # batch size
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
    
    
# main loop