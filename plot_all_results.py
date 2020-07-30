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
medium_aggregator = {}
hard_aggregator = {}
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
        #if track in :
        if track in [40712,40774,40773,40772,40711,40771,40792,40775,39361,40901]:
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

        elif track in [40863,40892,40891,40714,40855,40903,40864,40761,39311,39501]:
            try:
                hard_aggregator[det_step]
            except:
                hard_aggregator[det_step] = {}
                
            try:
                for key in metrics:
                    hard_aggregator[det_step][key] += metrics[key][0]
            except:
                for key in metrics:
                    hard_aggregator[det_step][key] = metrics[key][0] 
                    
        else:
            try:
                medium_aggregator[det_step]
            except:
                medium_aggregator[det_step] = {}
                
            try:
                for key in metrics:
                    medium_aggregator[det_step][key] += metrics[key][0]
            except:
                for key in metrics:
                    medium_aggregator[det_step][key] = metrics[key][0] 

det_steps = []
motas = []
framerates = []

easy_det_steps = []
easy_motas = []
easy_framerates = []
medium_det_steps = []
medium_motas = []
medium_framerates = []
hard_det_steps = []
hard_motas = []
hard_framerates = []

for det_step in range(0,50):
    if det_step in aggregator.keys():
        det_steps.append(det_step)
        motas.append(aggregator[det_step]["mota"]/40)
        framerates.append(aggregator[det_step]["framerate"]/40)
        
    if det_step in easy_aggregator.keys():
        easy_det_steps.append(det_step)
        easy_motas.append(easy_aggregator[det_step]["mota"]/10)
        easy_framerates.append(easy_aggregator[det_step]["framerate"]/10)
        
    if det_step in medium_aggregator.keys():
        medium_det_steps.append(det_step)
        medium_motas.append(medium_aggregator[det_step]["mota"]/20)
        medium_framerates.append(medium_aggregator[det_step]["framerate"]/20)
        
    if det_step in hard_aggregator.keys():
        hard_det_steps.append(det_step)
        hard_motas.append(hard_aggregator[det_step]["mota"]/10)
        hard_framerates.append(hard_aggregator[det_step]["framerate"]/10)

metrics_list = aggregator

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
plt.plot(framerates,motas,marker = 'o')
for i in range(len(det_steps)):
    plt.annotate(det_steps[i],(framerates[i]+0.25,motas[i]),fontsize = 25)

# plt.plot(easy_framerates,easy_motas)
# for i in range(len(easy_det_steps)):
#     plt.annotate(easy_det_steps[i],(easy_framerates[i],easy_motas[i]),fontsize = 5)

plt.plot(framerates_2,motas_2,marker = 'o')
for i in range(len(det_steps_2)):
    plt.annotate(det_steps_2[i],(framerates_2[i]+0.25,motas_2[i]),fontsize = 25)

plt.plot(framerates_3,motas_3,marker = 'o')
for i in range(len(det_steps_3)):
    plt.annotate(det_steps_3[i],(framerates_3[i]+0.25,motas_3[i]),fontsize = 25)

#plt.plot(framerates_4,motas_4)
#for i in range(len(det_steps_4)):
#    plt.annotate(det_steps_4[i],(framerates_4[i],motas_4[i]),fontsize = 10)


plt.legend(["Tracking By Localization","Frame Skipping","Alternate localization and Frame Skipping","Adaptive"],fontsize = 30)
plt.xlim([0,50])
plt.ylim([0.0,0.5])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Framerate (fps)",fontsize = 40)
plt.ylabel("Accuracy (MOTA)",fontsize = 40)
#plt.title("Accuracy versus Framerate", fontsize =)



