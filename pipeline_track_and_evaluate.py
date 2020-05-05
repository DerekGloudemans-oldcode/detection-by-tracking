import mot_eval as mot
import track_utils as track_utils

import os
import numpy as np
import random 
import time
import math
import _pickle as pickle
random.seed = 0

import cv2
from PIL import Image
import torch

import matplotlib.pyplot  as plt

from detrac_files.detrac_train_localizer import ResNet_Localizer, load_model, class_dict
from torch_kf import Torch_KF#, filter_wrapper



if __name__ == "__main__":
    # input parameters
    det_step = 11
    srr = 0
    ber = 1
    

    
    tracks = [40243,20011,20012,63562,63563]
    tracks = [20012,20034,63525,63544,63552,63553,63554,63561,63562,63563]
    tracks = [20034,63525]
    tracks = [63563]
    SHOW = True
    
    # get list of all files in directory and corresponding path to track and labels
    track_dir = "/home/worklab/Desktop/detrac/DETRAC-all-data"
    label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
    track_list = [os.path.join(track_dir,item) for item in os.listdir(track_dir)]  
    label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)] 
    track_dict = {}
    for item in track_list:
        id = int(item.split("MVI_")[-1])
        track_dict[id] = {"frames": item,
                          "labels": None}
    for item in label_list:
        id = int(item.split("MVI_")[-1].split("_v3.xml")[0])
        track_dict[id]['labels'] = item
    
    running_metrics = {}
    
    # for each track and for specified det_step, track and evaluate
    for id in tracks:
        # track
        #with open("velocity_fitted_Q.cpkl", 'rb') as f:
        with open("filter_states/velocity_Q_R.cpkl", 'rb') as f:
        #with open("filter_states/acceleration_Q.cpkl",'rb') as f:
            kf_params = pickle.load(f)
        
        tracker = Torch_KF("cpu",mod_err = 1, meas_err = 1, state_err = 0, INIT = kf_params)
        frames = track_dict[id]["frames"]
        preds, Hz, time_metrics = track_utils.skip_track(frames,
                                                         tracker,
                                                         det_step = det_step, 
                                                         ber = ber, 
                                                         srr = srr,
                                                         PLOT = SHOW)
  
        # get ground truth labels
        gts,metadata = mot.parse_labels(track_dict[id]["labels"])
        ignored_regions = metadata['ignored_regions']

        # match and evaluate
        metrics,acc = mot.evaluate_mot(preds,gts,ignored_regions,threshold = 0.9)
        metrics = metrics.to_dict()
        metrics["framerate"] = {0:Hz}

        # add results to aggregate results
        try:
            for key in metrics:
                running_metrics[key] += metrics[key][0]
        except:
            for key in metrics:
                running_metrics[key] = metrics[key][0]
                
    # average results  
    print(ber,srr)
    print("\n\nAverage Metrics for {} tracks with det step = {}:".format(len(tracks),det_step))    
    for key in running_metrics:
        running_metrics[key] /= len(tracks)
        print(" {}: {}".format(key,running_metrics[key]))
    
