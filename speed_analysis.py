#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:02:50 2020

@author: worklab
"""


import sys
sys.path.append("py_motmetrics")
import motmetrics
import os 
import numpy as np
import xml.etree.ElementTree as ET
import _pickle as pickle
import matplotlib.pyplot as plt

relevant_info = []

# get track, det_step, speed, MOTA and # objects?
results_directory = "/home/worklab/Documents/code/detection-by-tracking/speed_analysis_results"

for file in os.listdir(results_directory): 
    try:
        with open(os.path.join(results_directory,file),"rb") as f:
            preds, metrics,avg_speed = pickle.load(f)
    except:
        with open(os.path.join(results_directory,file),"rb") as f:
            preds, metrics,avg_speed,_avg_cov = pickle.load(f)
            
    keep = {"track":int(file.split("_")[1]),
            "det_step":int(file.split("_")[2].split(".")[0]),
            "avg_longevity":metrics["num_objects"][0]/metrics["num_unique_objects"][0],
            "mota":metrics["mota"][0],
            "avg_speed":avg_speed,
            "Hz":metrics["framerate"][0]
            }
    relevant_info.append(keep)
    
# get unique values for id
track_ids = []
for item in relevant_info:
    track_ids.append(item["track"])
track_ids = list(set(track_ids))


# get unique det_steps
det_steps = []
for item in relevant_info:
    det_steps.append(item["det_step"])
det_steps = list(set(det_steps))
det_steps.sort()

# create nd array with track x det_step x [mota,framerate,avg_longevity,avg_speed]

metrics = np.zeros([len(track_ids),len(det_steps),4])

for item in relevant_info:
    metrics[track_ids.index(item["track"]),det_steps.index(item["det_step"])] = np.array([item["mota"],item["Hz"],item["avg_longevity"],item["avg_speed"]])
    
    
    
    
    
    
# sort metrics by average speed when det_step = 15
speeds = metrics[:,4,3]
idxs = np.argsort(speeds)
speeds = speeds[idxs]
metrics = metrics[idxs,:,:]

# generate plot
fig, ax = plt.subplots(figsize = (10,10))
im = ax.imshow(metrics[:,:,0])  
 
ax.set_xlabel("Detection Step")
ax.set_ylabel("Avg Object Speed")
fig.colorbar(im)
ax.set_title("Accuracy sorted by avg speed")   

# We want to show all ticks...
ax.set_xticks(np.arange(len(det_steps)))
ax.set_yticks(np.arange(len(speeds)))
# ... and label them with the respective list entries
ax.set_xticklabels(det_steps)
ax.set_yticklabels(speeds)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
         rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=0, va="center",
     rotation_mode="anchor")







# sort metrics by average_longevity when det_step = 15
longevities = metrics[:,4,2].round()
idxs = np.argsort(longevities)
longevities = longevities[idxs]
metrics = metrics[idxs,:,:]

# generate plot
fig, ax = plt.subplots(figsize = (10,10))
im = ax.imshow(metrics[:,:,0])  
 
ax.set_xlabel("Detection Step")
ax.set_ylabel("Avg Object Longevity")
fig.colorbar(im)
ax.set_title("Accuracy sorted by longevity")   

# We want to show all ticks...
ax.set_xticks(np.arange(len(det_steps)))
ax.set_yticks(np.arange(len(longevities)))
# ... and label them with the respective list entries
ax.set_xticklabels(det_steps)
ax.set_yticklabels(longevities)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
         rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=0, va="center",
     rotation_mode="anchor")





# sort by combo metric
longevities = metrics[:,4,2].round()
speeds = metrics[:,4,3]
combo = longevities / speeds
idxs = np.argsort(combo)
combo = combo[idxs]
metrics = metrics[idxs,:,:]

# generate plot
fig, ax = plt.subplots(figsize = (10,10))
im = ax.imshow(metrics[:,:,0])  
 
ax.set_xlabel("Detection Step")
ax.set_ylabel("combo")
fig.colorbar(im)
ax.set_title("Accuracy sorted by combo")   

# We want to show all ticks...
ax.set_xticks(np.arange(len(det_steps)))
ax.set_yticks(np.arange(len(combo)))
# ... and label them with the respective list entries
ax.set_xticklabels(det_steps)
ax.set_yticklabels(combo)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
         rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=0, va="center",
     rotation_mode="anchor")