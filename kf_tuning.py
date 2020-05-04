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




from PIL import Image
from torchvision.transforms import functional as F
from torchvision.ops import roi_align

from detrac_files.detrac_train_localizer import ResNet_Localizer, load_model, class_dict

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

b         = 3000 # batch size
n_pre     = 3      # number of frames used to initially tune tracker
n_post    = 1     # number of frame rollouts used to evaluate tracker


localizer = ResNet_Localizer()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.empty_cache() 
    
resnet_checkpoint = "/home/worklab/Desktop/checkpoints/detrac_localizer_retrain2/cpu_resnet18_epoch14.pt"
cp = torch.load(resnet_checkpoint)
localizer.load_state_dict(cp['model_state_dict']) 
localizer = localizer.to(device)

try:
    loader
except:
    # initialize dataset
    image_dir = "/home/worklab/Desktop/detrac/DETRAC-all-data"
    label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
    dataset = Track_Dataset(image_dir,label_dir, n = (n_pre + n_post+1))
    
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

with open("velocity_fitted_Q.cpkl","rb") as f:
    kf_params = pickle.load(f)


# fit Q and mu_Q
if False:
    error_vectors = []
    for iteration in range(1000):
        
        # grab batch
        batch, ims = next(iter(loader))
        
        # initialize tracker
        #tracker = Torch_KF("cpu",INIT = kf_params)
        tracker = Torch_KF("cpu",INIT = kf_params)
    
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
            pred = torch.from_numpy(np.array(objs)).double()
            
            
            # evaluate
            
            # get ground truths
            pos = batch[:,frame,:]
            pos_prev = batch[:,frame-1,:]
            pos_next = batch[:,frame+1,:]
            
            vel = ( (pos_next - pos) + (pos - pos_prev) ) / 2.0
            
            gt = torch.cat((pos, vel[:,:3]),dim = 1)
            error = gt - pred
            error = error.mean(dim = 0)
            error_vectors.append(error)
            
            print("Finished iteration {}".format(iteration))
    
    error_vectors = torch.stack(error_vectors,dim = 0)
    mean = torch.mean(error_vectors, dim = 0)
    
    covariance = torch.zeros((7,7))
    for vec in error_vectors:
        covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
        
    kf_params["mu_Q"] = mean
    kf_params["Q"] = covariance
    
    #with open("velocity_fitted_Q.cpkl","wb") as f:
    #    pickle.dump(kf_params,f)
    
    
# fit H and mu_H
if True:
    error_vecs = []
    
    for iteration in range(50):
        batch, paths = next(iter(loader))
        
        
        error = np.zeros(4)
        # second frame is used to get a Localizer measurement and compare to ground truth
        for i in range(len(batch)): # i indexes items in batch
            
            
            # note that for some strange reason dataloader collates the image lists as
            # len track snippet, batch size so indexing is in reverse
            with Image.open(paths[0][i]) as im:
               im = F.to_tensor(im)
               frame = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
               frame = frame.to(device)
            
            # crop image
            boxes = batch[i,0,:].unsqueeze(0)
            
            # convert xysr boxes into xmin xmax ymin ymax
            # first row of zeros is batch index (batch is size 0) for ROI align
            new_boxes = np.zeros([len(boxes),5]) 

            # use either s or s x r for both dimensions, whichever is larger,so crop is square
            #box_scales = np.max(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1)
            box_scales = np.min(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
                
            #expand box slightly
            ber = 1
            box_scales = box_scales * ber# box expansion ratio
            
            new_boxes[:,1] = boxes[:,0] - box_scales/2
            new_boxes[:,3] = boxes[:,0] + box_scales/2 
            new_boxes[:,2] = boxes[:,1] - box_scales/2 
            new_boxes[:,4] = boxes[:,1] + box_scales/2 
            torch_boxes = torch.from_numpy(new_boxes).float().to(device)
            
            # crop using roi align
            crops = roi_align(frame.unsqueeze(0),torch_boxes,(224,224))
            
            # 4b. Localize objects using localizer
   
            cls_out,reg_out = localizer(crops)
            torch.cuda.synchronize()
            

            # 5b. convert to global image coordinates 
                
            # these detections are relative to crops - convert to global image coords
            wer = 3 # window expansion ratio, was set during training
            
            detections = (reg_out* 224*wer - 224*(wer-1)/2)
            detections = detections.data.cpu()
            
            # add in original box offsets and scale outputs by original box scales
            detections[:,0] = detections[:,0]*box_scales/224 + new_boxes[:,1]
            detections[:,2] = detections[:,2]*box_scales/224 + new_boxes[:,1]
            detections[:,1] = detections[:,1]*box_scales/224 + new_boxes[:,2]
            detections[:,3] = detections[:,3]*box_scales/224 + new_boxes[:,2]

            # convert into xysr form 
            output = np.zeros([len(detections),4])
            output[:,0] = (detections[:,0] + detections[:,2]) / 2.0
            output[:,1] = (detections[:,1] + detections[:,3]) / 2.0
            output[:,2] = (detections[:,2] - detections[:,0])
            output[:,3] = (detections[:,3] - detections[:,1]) / output[:,2]
            pred = output
            
            error += boxes.data.numpy()[0] - pred[0]   
                
        error = error/len(batch) 
        error_vecs.append(error)
        print("Finished iteration {}".format(iteration))
        
    error_vectors = np.array(error_vecs)
    error_vectors = error_vectors * len(paths)
    error_vectors = torch.from_numpy(error_vectors)
    
    mean = torch.mean(error_vectors, dim = 0)
    
    covariance = torch.zeros((4,4))
    for vec in error_vectors:
        covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
        
    covariance = covariance / 99.0
    
    kf_params["mu_R"] = mean
    kf_params["R"] = covariance
    
    with open("velocity_fitted_Q_R.cpkl",'wb') as f:
        pickle.dump(kf_params,f)