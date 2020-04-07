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
from scipy.optimize import linear_sum_assignment

from detrac_files.detrac_train_localizer import ResNet_Localizer, load_model, class_dict
from pytorch_yolo_v3.yolo_detector import Darknet_Detector
from torch_kf import Torch_KF#, filter_wrapper


def parse_detections(detections):
    # remove duplicates
    detections = detections.unique(dim = 0)
    
    # input form --> batch_idx, xmin,ymin,xmax,ymax,objectness,max_class_conf, class_idx 
    # output form --> x_center,y_center, scale, ratio, class_idx, max_class_conf
    
    output = torch.zeros(detections.shape[0],6)
    detections  =  detections[:,1:]
    
    output[:,0] = (detections[:,0] + detections[:,2]) / 2.0
    output[:,1] = (detections[:,1] + detections[:,3]) / 2.0
    output[:,2] = (detections[:,2] - detections[:,0])
    output[:,3] = (detections[:,3] - detections[:,1]) / output[:,2]
    output[:,4] =  detections[:,6]
    output[:,5] =  detections[:,5]
    
    return output

def match_hungarian(first,second,iou_cutoff = 0.5):
    """
    performs  optimal (in terms of sum distance) matching of points 
    in first to second using the Hungarian algorithm
    inputs - N x 2 arrays of object x and y coordinates from different frames
    output - M x 1 array where index i corresponds to the second frame object 
    matched to the first frame object i
    """
    # find distances between first and second
    dist = np.zeros([len(first),len(second)])
    for i in range(0,len(first)):
        for j in range(0,len(second)):
            dist[i,j] = np.sqrt((first[i,0]-second[j,0])**2 + (first[i,1]-second[j,1])**2)
            
    a, b = linear_sum_assignment(dist)
    
    # convert into expected form
    matchings = np.zeros(len(first))-1
    for idx in range(0,len(a)):
        matchings[a[idx]] = b[idx]
    matchings = np.ndarray.astype(matchings,int)
    
    if True:
        # calculate intersection over union  (IOU) for all matches
        for i,j in enumerate(matchings):
            x1_left = first[i][0] -first[i][2]*first[i][3]/2
            x2_left = second[j][0] -second[j][2]*second[j][3]/2
            x1_right= first[i][0] + first[i][2]*first[i][3]/2
            x2_right = second[j][0] +second[j][2]*second[j][3]/2
            x_intersection = min(x1_right,x2_right) - max(x1_left,x2_left) 
            
            y1_left = first[i][1] -first[i][2]/2.0
            y2_left = second[j][1] -second[j][2]/2.0
            y1_right= first[i][1] + first[i][2]/2.0
            y2_right = second[j][1] +second[j][2]/2.0
            y_intersection = min(y1_right,y2_right) - max(y1_left,y2_left)
            
            a1 = first[i,3]*first[i,2]**2 
            a2 = second[j,3]*second[j,2]**2 
            intersection = x_intersection*y_intersection
             
            iou = intersection / (a1+a2-intersection) 
            
            # supress matchings with iou below cutoff
            if iou < iou_cutoff:
                matchings[i] = -1
    out_matchings = []
    for i in range(len(matchings)):
        if matchings[i] != -1:
            out_matchings.append([i,matchings[i]])
    return np.array(out_matchings)   
    
    
if __name__ == "__main__":
    #%% 1. Set up models, etc.

    yolo_checkpoint =   "/home/worklab/Desktop/checkpoints/yolo/yolov3.weights"
    resnet_checkpoint = "/home/worklab/Desktop/checkpoints/detrac_localizer/resnet18_cpu.pt"
    track_directory =   "/home/worklab/Desktop/detrac/DETRAC-train-data/MVI_20011"
    det_step = 10               
    PLOT = True
    fsld_max = 1
    
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
        #localizer,_,_,all_metrics = load_model(resnet_checkpoint, localizer, None)
        
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache() 
    localizer = localizer.to(device)
    print("Detector and Localizer on {}.".format(device))
    
    tracker = Torch_KF("cpu",mod_err = 1000)

     
    #%% 2. Loop Setup
    
    files = []
    frames = []
    for item in [os.path.join(track_directory,im) for im in os.listdir(track_directory)]:
        files.append(item)
        files.sort()
        
    # open and parse images    
    for f in files:
         im = Image.open(f)
         im = F.to_tensor(im)
         im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
         #im = im.to(device)
         frames.append(im)
    n_frames = len(frames)
     
    print("All frames loaded into memory")     
   
    frame_num = 0               # iteration counter   
    next_obj_id = 0             # next id for a new object (incremented during tracking)
    fsld = {}                   # fsld[id] stores frames since last detected for object id
    all_tracks = {}
    #%% 3. Main Loop


    for frame in frames:
        
        # 1. Predict next object locations
        try:
            tracker.predict()
            pre_locations = tracker.objs()
        except:
            # in the case that there are no active objects will throw exception
            pre_locations = []
            
        # 2. Detect, either with ResNet or Yolo
        if frame_num % det_step == 0:
            frame = frame.to(device)
            detections = detector.detect_tensor(frame).cpu()
            detections = parse_detections(detections)
        else:
            pass
        
        
        # 3. Match, using Hungarian Algorithm
        pre_ids = []
        pre_loc = []
        for id in pre_locations:
            pre_ids.append(id)
            pre_loc.append(pre_locations[id])
        pre_loc = np.array(pre_loc)
        
        # matchings[i] = [a,b] where a is index of pre_loc and b is index of detection
        matchings = match_hungarian(pre_loc,detections[:,:4])
        
        
        # 4. Update tracked objects
        update_array = np.zeros([len(matchings),4])
        update_ids = []
        for i in range(len(matchings)):
            a = matchings[i,0] # index of pre_loc
            b = matchings[i,1] # index of detections
           
            update_array[i,:] = detections[b,:4]
            update_ids.append(pre_ids[a])
            fsld[pre_ids[a]] = 0 # fsld = 0 since this id was detected this frame
        
        if len(update_array) > 0:    
            tracker.update(update_array,update_ids)
                        
        
        # 5. For each detection not in matchings, add a new object
        new_array = np.zeros([len(detections) - len(matchings),4])
        new_ids = []
        cur_row = 0
        for i in range(len(detections)):
            if len(matchings) == 0 or i not in matchings[:,1]:
                
                new_ids.append(next_obj_id)
                new_array[cur_row,:] = detections[i,:4]

                fsld[next_obj_id] = 0
                all_tracks[next_obj_id] = np.zeros([n_frames,7])
                
                next_obj_id += 1
                cur_row += 1
       
        if len(new_array) > 0:        
            tracker.add(new_array,new_ids)
        
        
        # 6. For each untracked object, increment fsld
        for i in range(len(pre_ids)):
            if i not in matchings[:,0]:
                fsld[pre_ids[i]] += 1
                
        
        # 7. remove lost objects
        removals = []
        for id in pre_ids:
            if fsld[id] > fsld_max:
                removals.append(id)
       
        if len(removals) > 0:
            tracker.remove(removals)    
        
        
        # 8. Get all object locations and store in output dict
        post_locations = tracker.objs()
        for id in post_locations:
            all_tracks[id][frame_num,:] = post_locations[id]
            
            
        # 9. Plot
        if PLOT:
            # convert tensor back to CV im
            frame = frame.data.cpu().numpy()
            im   = frame.transpose((1,2,0))
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            im   = std * im + mean
            im   = np.clip(im, 0, 1)
            im   = im[:,:,[2,1,0]]
            im = im.copy()
            
            for id in post_locations:
                # plot bbox
                label = "Object {}".format(id)
                bbox = post_locations[id][:4]

                color = (0.7,0.7,0.4) #colors[int(obj.cls)]
                c1 = (int(bbox[0]-bbox[3]*bbox[2]/2),int(bbox[1]-bbox[2]/2))
                c2 =  (int(bbox[0]+bbox[3]*bbox[2]/2),int(bbox[1]+bbox[2]/2))
                cv2.rectangle(im,c1,c2,color,1)
                
                # plot label
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,1 , 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(im, c1, c2,color, -1)
                cv2.putText(im, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,1, [225,255,255], 1);
            
            
            cv2.imshow("window",im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # plot each rectange with text label
            
        
        print("Finished frame {}".format(frame_num))
        frame_num += 1