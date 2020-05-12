#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:33:03 2020

@author: worklab
"""


import os
import numpy as np
import random 
import time
random.seed = 0

import cv2
from PIL import Image
import torch

from torchvision.transforms import functional as F
import torch.multiprocessing as mp


class FrameLoader():
    
    def __init__(self,track_directory,device, det_step, init_frames, PRELOAD = False):
        
        """
        Parameters
        ----------
        track_directory : str
            Path to frames 
        device : torch.device
            specifies where frames should be loaded to , memory or GPU
        det_step : int
            How often full detection is performed, necessary for image pre-processing
        init_frames : int
            Number of frames used to initialize Kalman filter every det_step frames
        cutoff : int, optional
            If not None, only this many frames are loaded into memory. The default is None.
    
        """
        
        files = []
        for item in [os.path.join(track_directory,im) for im in os.listdir(track_directory)]:
            files.append(item)
            files.sort()    
        
        self.files = files
        self.det_step = det_step
        self.init_frames = init_frames
        self.device = device
    
        # create shared queue
        #mp.set_start_method('spawn')
        ctx = mp.get_context('spawn')
        self.queue = ctx.Queue()
        
        self.frame_idx = 0
        
        self.worker = ctx.Process(target=load_to_queue, args=(self.queue,files,det_step,init_frames,device,))
        self.worker.start()
        time.sleep(5)
        
    def __len__(self):
        return len(self.files)
    
    def __next__(self):
        if self.frame_idx < len(self):
            frame_num = self.frame_idx
            self.frame_idx += 1
        
            frame = self.queue.get(timeout = 0)
            return frame_num, frame
        
        else:
            self.worker.terminate()
            self.worker.join()
            return -1,None

def load_to_queue(image_queue,files,det_step,init_frames,device,queue_size = 5):
    
    frame_idx = 0    
    while frame_idx < len(files):
        
        if image_queue.qsize() < queue_size:
            
            # load next image
            with Image.open(files[frame_idx]) as im:
             
              if frame_idx % det_step < init_frames:   
                  # convert to CV2 style image
                  open_cv_image = np.array(im) 
                  im = open_cv_image.copy() 
                  original_im = im[:,:,[2,1,0]].copy()
                  # new stuff
                  dim = (im.shape[1], im.shape[0])
                  im = cv2.resize(im, (1024,1024))
                  im = im.transpose((2,0,1)).copy()
                  im = torch.from_numpy(im).float().div(255.0).unsqueeze(0)
                  dim = torch.FloatTensor(dim).repeat(1,2)
              else:
                  # keep as tensor
                  original_im = np.array(im)[:,:,[2,1,0]].copy()
                  im = F.to_tensor(im)
                  im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                  dim = None
                 
              # store preprocessed image, dimensions and original image
              im = im.to(device)
              frame = (im,dim,original_im)
             
              # append to queue
              image_queue.put(frame)
             
            frame_idx += 1
    
    while True:  
           time.sleep(5)
        
if __name__ == "__main__":
    
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
        
    path = track_dict[40962]['frames']     
    test = FrameLoader(path,torch.device("cuda:0"),det_step = 10, init_frames = 3)

    all_time = 0
    print(test.queue.qsize())
    count = 0
    while True:
        start = time.time()
        num, frame = next(test)
        
        if num > 0:
            all_time += (time.time() - start)
        
        time.sleep(0.03)
       
        if frame is not None:
            out = frame[0] + 1
        
        if num == -1:
            break
        count += 1
        print(count, all_time/count)