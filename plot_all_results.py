#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:26:49 2020

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

# first set of results

aggregator = {}
easy_aggregator = {}
results_directory = "/home/worklab/Documents/code/detection-by-tracking/results_no_mask"

for file in os.listdir(results_directory):   
    with open(os.path.join(results_directory,file),"rb") as f:
        results = pickle.load(f)
        metrics = results[1]
        det_step = int(file.split("_")[-1].split(".cpkl")[0])
        
        try:
            aggregator[det_step]
        except:
            aggregator[det_step] = {}
            
        try:
            for key in metrics:
                aggregator[det_step][key] += metrics[key][0]
        except:
            for key in metrics:
                aggregator[det_step][key] = metrics[key][0]
        
        track = int(file.split("_")[1])
        #if track in [40712,40774,40773,40772,40711,40771,40792,40775,39361,40901]:
        if track not in [40863,40892,40891,40714,40855,40903,40864,40761,39311,39501]:
            try:
                easy_aggregator[det_step]
            except:
                easy_aggregator[det_step] = {}
                
            try:
                for key in metrics:
                    easy_aggregator[det_step][key] += metrics[key][0]
            except:
                for key in metrics:
                    easy_aggregator[det_step][key] = metrics[key][0]        


det_steps = []
motas = []
framerates = []

easy_det_steps = []
easy_motas = []
easy_framerates = []

for det_step in range(0,50):
    if det_step in aggregator.keys():
        det_steps.append(det_step)
        motas.append(aggregator[det_step]["mota"]/40)
        framerates.append(aggregator[det_step]["framerate"]/40)
        
    if det_step in easy_aggregator.keys():
        easy_det_steps.append(det_step)
        easy_motas.append(easy_aggregator[det_step]["mota"]/30)
        easy_framerates.append(easy_aggregator[det_step]["framerate"]/30)


# second set of results
  
aggregator = {}   
results_directory = "/home/worklab/Documents/code/detection-by-tracking/results_kalman"
for file in os.listdir(results_directory):   
    with open(os.path.join(results_directory,file),"rb") as f:
        results = pickle.load(f)
        metrics = results[1]
        det_step = int(file.split("_")[-1].split(".cpkl")[0])
        
        try:
            aggregator[det_step]
        except:
            aggregator[det_step] = {}
            
        try:
            for key in metrics:
                aggregator[det_step][key] += metrics[key][0]
        except:
            for key in metrics:
                aggregator[det_step][key] = metrics[key][0]
        
       
det_steps_2 = []
motas_2 = []
framerates_2 = []

for det_step in range(0,50):
    if det_step in aggregator.keys():
        det_steps_2.append(det_step)
        motas_2.append(aggregator[det_step]["mota"]/40)
        framerates_2.append(aggregator[det_step]["framerate"]/40)    
 
# third set of results 
   
aggregator = {}   
results_directory = "/home/worklab/Documents/code/detection-by-tracking/results_skip_and_localize"
for file in os.listdir(results_directory):   
    with open(os.path.join(results_directory,file),"rb") as f:
        results = pickle.load(f)
        metrics = results[1]
        det_step = int(file.split("_")[-1].split(".cpkl")[0])
        
        try:
            aggregator[det_step]
        except:
            aggregator[det_step] = {}
            
        try:
            for key in metrics:
                aggregator[det_step][key] += metrics[key][0]
        except:
            for key in metrics:
                aggregator[det_step][key] = metrics[key][0]
        
       
det_steps_3 = []
motas_3 = []
framerates_3 = []

for det_step in range(0,50):
    if det_step in aggregator.keys():
        det_steps_3.append(det_step)
        motas_3.append(aggregator[det_step]["mota"]/40)
        framerates_3.append(aggregator[det_step]["framerate"]/40)    



# fourth set of results 
   
aggregator = {}   
results_directory = "/home/worklab/Documents/code/detection-by-tracking/results_adaptive"
for file in os.listdir(results_directory):   
    with open(os.path.join(results_directory,file),"rb") as f:
        results = pickle.load(f)
        metrics = results[1]
        det_step = float(file.split("_")[-1].split(".cpkl")[0])
        
        try:
            aggregator[det_step]
        except:
            aggregator[det_step] = {}
            
        try:
            for key in metrics:
                aggregator[det_step][key] += metrics[key][0]
        except:
            for key in metrics:
                aggregator[det_step][key] = metrics[key][0]
        
       
det_steps_4 = []
motas_4 = []
framerates_4 = []

for det_step in range(0,50):
    if det_step in aggregator.keys():
        det_steps_4.append(det_step)
        motas_4.append(aggregator[det_step]["mota"]/40)
        framerates_4.append(aggregator[det_step]["framerate"]/40)      
    
    
####################### PLOT ####################################    
plt.figure(figsize = (5,5))    
plt.plot(framerates,motas)
for i in range(len(det_steps)):
    plt.annotate(det_steps[i],(framerates[i],motas[i]),fontsize = 10)

# plt.plot(easy_framerates,easy_motas)
# for i in range(len(easy_det_steps)):
#     plt.annotate(easy_det_steps[i],(easy_framerates[i],easy_motas[i]),fontsize = 5)

plt.plot(framerates_2,motas_2)
for i in range(len(det_steps_2)):
    plt.annotate(det_steps_2[i],(framerates_2[i],motas_2[i]),fontsize = 10)

plt.plot(framerates_3,motas_3)
for i in range(len(det_steps_3)):
    plt.annotate(det_steps_3[i],(framerates_3[i],motas_3[i]),fontsize = 10)

plt.plot(framerates_4,motas_4)
for i in range(len(det_steps_4)):
    plt.annotate(det_steps_4[i],(framerates_4[i],motas_4[i]),fontsize = 10)


plt.legend(["Localize Only","Filter Only","Alternate Filter and Localize","Adaptive"],fontsize = 20)
plt.xlim([0,55])
plt.ylim([0.1,0.6])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Framerate (fps)",fontsize = 20)
plt.ylabel("Accuracy (MOTA)",fontsize = 20)
plt.title("Accuracy versus Framerate", fontsize = 24)
