#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:20:41 2020

@author: worklab
"""

import torch
import numpy as np
import time
import random
import _pickle as pickle
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
    iou = torch.div(intersection,union)
    mean_iou = torch.mean(iou)
    
    return mean_iou

def abs_err(a,b):
    mean = (a + b) / 2.0
    diff = torch.abs(a-b)
    avg_err = diff#/mean
    return 1 - torch.mean(avg_err)

def score_tracker(tracker,batch,n_pre,n_post):
    """
    Evaluates a tracker by inially updating it with n_pre frames of object locations,
    then rolls out predictions for n_post frames and evaluates iou with ground
    truths for these frames. Returns average iou for all objects and rollout frames

    Parameters
    ----------
    tracker : Torch_KF object
        a Kalman Filter tracker
    batch : Float Tensor - [batch_size,(n_pre + n_post),4] 
        bounding boxes for several objects for several frames
    n_pre : int
        number of frames used to initially update the tracker
    n_post : int
        number of frames used to evaluate the tracker

    Returns
    -------
    score : average bbox iou for all objects evaluated over n_post rollout predictions
    """
    obj_ids = [i for i in range(len(batch))]
    
    tracker.add(batch[:,0,:],obj_ids)
    
    for frame in range(1,n_pre):
        tracker.predict()
        tracker.update(batch[:,frame,:],obj_ids)
    
    running_mean_iou = 0
    for frame in range(n_pre,n_pre+n_post):
        # roll out a frame
        tracker.predict()
        
        # get predicted object locations
        objs = tracker.objs()
        objs = [objs[key] for key in objs]
        pred = torch.from_numpy(np.array(objs)).double()[:,:4]
        
        
        # evaluate
        #val = iou(batch[:,frame,:],pred)
        val = abs_err(batch[:,frame,:],pred)
        running_mean_iou += val.item()
    
    score = running_mean_iou / n_post
    
    return score


##############################################################################
    ##############################################################################
##############################################################################
    ##############################################################################
##############################################################################
    ##############################################################################
##############################################################################
    ##############################################################################
##############################################################################
    ##############################################################################




# define parameters
lr        = 0.1 # learning rate
e         = 1e-05 # gradient estimation step size
b         = 3000 # batch size
n_pre     = 2      # number of frames used to initially tune tracker
n_post    = 1     # number of frame rollouts used to evaluate tracker
tune      = ['Q'] # which KF parameters to optimize over


    


try:
    loader
except:
    # initialize dataset
    dataset = Track_Dataset("/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3", n = (n_pre + n_post))
    
    # 3. create training params
    params = {'batch_size' : b,
              'shuffle'    : True,
              'num_workers': 0,
              'drop_last'  : True
              }
    
    # returns batch_size x (n_pre + n_post) x 4 tensor
    loader = DataLoader(dataset, **params)


# create initial values for each matrix
tracker = Torch_KF("cpu",INIT = None)
kf_params = {
        "P":tracker.P0.squeeze(0),
        "Q":tracker.Q.squeeze(0),
        "R":tracker.R.squeeze(0),
        "F":tracker.F,
        "H":tracker.H
        }
#kf_params['Q'] = torch.ones(kf_params['Q'].shape)
#kf_params['P'] = torch.zeros(kf_params['P'].shape)
with open("1_step_Q.cpkl", 'rb') as f:
    kf_params = pickle.load(f)
#kf_params['F'] = torch.eye(kf_params['F'].shape[0])
kf_params['F'] = tracker.F
    
if False:    
    temp = next(iter(loader))
    a1 = temp[:,0,:]
    a2 = temp[:,1,:]
    print(iou(a1,a2).item())


for iteration in range(10000):
    start = time.time()
    
    # grab batch
    batch = next(iter(loader))
    
    # initialize tracker
    #tracker = Torch_KF("cpu",INIT = kf_params)
    tracker = Torch_KF("cpu",INIT = kf_params)

    # score tracker on batch
    baseline = score_tracker(tracker,batch,n_pre = n_pre,n_post = n_post)
    del tracker
    
    if iteration % 1 == 0:
        print("Iteration: {} ---> score {:.3f} --->".format(iteration, baseline),end =" ")
    # calculate gradients
    
    if 'P' in tune:         # note must be diagonal symmetrical
        P = kf_params['P'].clone()
        P_grad = torch.zeros(P.shape)
        
        for i in range(P.shape[0]):
            for j in range(i,P.shape[1]):
                # tweak Q
                P[i,j] = P[i,j] + e
                P[j,i] = P[j,i] + e
                
                new_params = kf_params.copy()
                new_params['P'] = P
                    
                tracker = Torch_KF("cpu",INIT = new_params)
                new_score = score_tracker(tracker,batch,n_pre = n_pre, n_post = n_post)
                
                grad = (new_score - baseline)/e
                P_grad[i,j] = grad
                P_grad[j,i] = grad
    
    if 'Q' in tune:         # note must be diagonal symmetrical
        Q = kf_params['Q'].clone()
        Q_grad = torch.zeros(Q.shape)
        
        for i in range(Q.shape[0]):
            for j in range(i,Q.shape[1]):
                # tweak Q
                Q[i,j] = Q[i,j] + e
                Q[j,i] = Q[j,i] + e
                
                new_params = kf_params.copy()
                new_params['Q'] = Q
                    
                tracker = Torch_KF("cpu",INIT = new_params)
                new_score = score_tracker(tracker,batch,n_pre = n_pre, n_post = n_post)
                
                grad = (new_score - baseline)/e
                Q_grad[i,j] = grad
                Q_grad[j,i] = grad
        
    if 'F' in tune:
        F = kf_params['F'].clone()
        F_grad = torch.zeros(F.shape)
        
        for i in range(F.shape[0]):
            for j in range(0,F.shape[1]):
                # tweak Q
                F[i,j] = F[i,j] + e
                
                new_params = kf_params.copy()
                new_params['F'] = F
                    
                tracker = Torch_KF("cpu",INIT = new_params)
                new_score = score_tracker(tracker,batch,n_pre = n_pre, n_post = n_post)
                
                grad = (new_score - baseline)/e
                F_grad[i,j] = grad

    if 'R' in tune:         # note must be diagonal symmetrical\
        pass
    
    # adjust each matrix based on gradients
    if 'Q' in tune:
        kf_params['Q'] += lr * Q_grad 
    if 'F' in tune:
        kf_params['F'] += lr_mod * F_grad 
    if 'P' in tune:
        kf_params['P'] += lr * P_grad 
    
    if iteration % 1 == 0:
        print("Took {:.2f} sec. Avg grad {}".format(time.time() -start,torch.mean(torch.abs(Q_grad))))

    if iteration % 50 == 0:
        with open("temp{}.cpkl".format(iteration),'wb') as f:
            pickle.dump(kf_params,f)