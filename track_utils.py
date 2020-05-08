"""
Created on Sat Mar  7 15:45:48 2020

@author: derek
"""

#%% 0. Imports 
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

from torchvision.transforms import functional as F
from torchvision.ops import roi_align
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
    
    prune_output = []
    for i in range(len(output)):
        if int(output[i,4]) in [2,3,5,7]:
            prune_output.append(i)
    
    output = output[prune_output,:]
    
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
 
def match_greedy(first,second,threshold = 10):
    """
    performs  greedy best-first matching of objects between frames
    inputs - N x 2 arrays of object x and y coordinates from different frames
    output - M x 1 array where index i corresponds to the second frame object 
    matched to the first frame object i
    """

    # find distances between first and second
    dist = np.zeros([len(first),len(second)])
    for i in range(0,len(first)):
        for j in range(0,len(second)):
            dist[i,j] = np.sqrt((first[i,0]-second[j,0])**2 + (first[i,1]-second[j,1])**2)
    
    # select closest pair
    matchings = np.zeros(len(first))-1
    unflat = lambda x: (x//len(second), x %len(second))
    while np.min(dist) < threshold:
        min_f, min_s = unflat(np.argmin(dist))
        #print(min_f,min_s,len(first),len(second),len(matchings),np.argmin(dist))
        matchings[min_f] = min_s
        dist[:,min_s] = np.inf
        dist[min_f,:] = np.inf
        
    return np.ndarray.astype(matchings,int)    
      
def test_outputs(bboxes,crops):
    # define figure subplot grid
    batch_size = len(crops)
    row_size = min(batch_size,8)
    fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True)
    
    for i in range(0,len(crops)):
        
        # get image
        im   = crops[i].data.cpu().numpy().transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        im   = std * im + mean
        im   = np.clip(im, 0, 1)
        
        # get predictions
        
        bbox = bboxes[i].data.cpu().numpy()
        
        wer = 3
        imsize = 224
        
        # transform bbox coords back into im pixel coords
        bbox = (bbox* 224*wer - 224*(wer-1)/2).astype(int)
        # plot pred bbox
        im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.1,0.6,0.9),2)
        im = im.get()
        
        # title with class preds and gt
        label = "{} -> ({})".format(" ", " ")

        if batch_size <= 8:
            axs[i].imshow(im)
            axs[i].set_title(label)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i//row_size,i%row_size].imshow(im)
            axs[i//row_size,i%row_size].set_title(label)
            axs[i//row_size,i%row_size].set_xticks([])
            axs[i//row_size,i%row_size].set_yticks([])
        plt.pause(.001)    
    
def load_models(device):
    yolo_checkpoint =   "/home/worklab/Desktop/checkpoints/yolo/yolov3.weights"
    resnet_checkpoint = "/home/worklab/Desktop/checkpoints/detrac_localizer/CPU_resnet18_epoch4.pt"
    resnet_checkpoint = "/home/worklab/Desktop/checkpoints/detrac_localizer_retrain2/cpu_resnet18_epoch14.pt"

    detector = Darknet_Detector(
                'pytorch_yolo_v3/cfg/yolov3.cfg',
                yolo_checkpoint,
                'pytorch_yolo_v3/data/coco.names',
                'pytorch_yolo_v3/pallete',
                resolution = 1024
                )
            
    localizer = ResNet_Localizer()
    cp = torch.load(resnet_checkpoint)
    localizer.load_state_dict(cp['model_state_dict']) 
    localizer = localizer.to(device)
    
    print("Detector and Localizer on {}.".format(device))
    return detector,localizer
    
def load_all_frames(track_directory,det_step,init_frames,cutoff = None): 
    print("Loading frames into memory.")
    files = []
    frames = []
    for item in [os.path.join(track_directory,im) for im in os.listdir(track_directory)]:
        files.append(item)
        files.sort()
        
    # open and parse images    
    for num, f in enumerate(files):
         with Image.open(f) as im:
             
             if num % det_step < init_frames:   
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
             frames.append((im,dim,original_im))
             if cutoff is not None and num > cutoff:
                 break
             
    n_frames = len(frames)
     
    print("All frames loaded into memory")
    return frames,n_frames
        
def plot(im,detections,post_locations,all_classes,class_dict,frame = None):
    im = im.copy()/255.0

    for det in detections:
        bbox = det[:4]
        color = (0.4,0.4,0.7) #colors[int(obj.cls)]
        c1 =  (int(bbox[0]-bbox[2]/2),int(bbox[1]-bbox[2]*bbox[3]/2))
        c2 =  (int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[2]*bbox[3]/2))
        cv2.rectangle(im,c1,c2,color,1)
        
    for id in post_locations:
        # plot bbox
        try:
            most_common = np.argmax(all_classes[id])
            cls = class_dict[most_common]
        except:
            cls = ""
        label = "{} {}".format(cls,id)
        bbox = post_locations[id][:4]
        if sum(bbox) != 0:

            color = (0.7,0.7,0.4) #colors[int(obj.cls)]
            c1 =  (int(bbox[0]-bbox[2]/2),int(bbox[1]-bbox[3]*bbox[2]/2))
            c2 =  (int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]*bbox[2]/2))
            cv2.rectangle(im,c1,c2,color,1)
            
            # plot label
            text_size = 0.8
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,text_size , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(im, c1, c2,color, -1)
            cv2.putText(im, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,text_size, [225,255,255], 1);
    
    
    if im.shape[0] > 1920:
        im = cv2.resize(im, (1920,1080))
    cv2.imshow("window",im)
    cv2.waitKey(1)
    
    if frame is not None:
        cv2.imwrite("output/{}.png".format(str(frame).zfill(4)),im*255)

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
    
    area_a = a[2] * a[2] * a[3]
    area_b = b[2] * b[2] * b[3]
    
    minx = max(a[0]-a[2]/2, b[0]-b[2]/2)
    maxx = min(a[0]+a[2]/2, b[0]+b[2]/2)
    miny = max(a[1]-a[2]*a[3]/2, b[1]-b[2]*b[3]/2)
    maxy = min(a[1]+a[2]*a[3]/2, b[1]+b[2]*b[3]/2)
    
    intersection = max(0, maxx-minx) * max(0,maxy-miny)
    union = area_a + area_b - intersection
    iou = intersection/union
    
    return iou
    
    
def skip_track(track_path, tracker, det_step = 1, srr = 0, ber = 1, PLOT = True):
        
    init_frames = 3
    
    fsld_max = det_step
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache() 
            
    # get CNNs
    try:
        detector
        localizer
    except:
        detector,localizer = load_models(device)
        
    
    localizer.eval()
         
    # Loop Setup
    frames,n_frames = load_all_frames(track_path,det_step,init_frames,cutoff = None)
    
    frame_num = 0               # iteration counter   
    next_obj_id = 0             # next id for a new object (incremented during tracking)
    fsld = {}                   # fsld[id] stores frames since last detected for object id
    
    all_tracks = {}             # stores states for each object
    all_classes = {}            # stores class evidence for each object
    
    # for keeping track of what's using up time
    time_metrics = {            
        "gpu_load":0,
        "predict":0,
        "pre_localize and align":0,
        "localize":0,
        "post_localize":0,
        "detect":0,
        "parse":0,
        "match":0,
        "match2":0,
        "update":0,
        "add and remove":0,
        "store":0,
        "plot":0
        }
                
                
    # 3. Main Loop

    for (frame,dim,original_im) in frames:
        
        # 1. Move image to GPU
        start = time.time()
        frame = frame.to(device,non_blocking = True)
        if frame_num % det_step < init_frames: #if frame_num % det_step == 0:
            dim = dim.to(device,non_blocking = True)                      
        time_metrics['gpu_load'] += time.time() - start
        

        # 2. Predict next object locations
        start = time.time()
        try: # in the case that there are no active objects will throw exception

            tracker.predict()
            pre_locations = tracker.objs()
        except:
            pre_locations = []    
        time_metrics['predict'] += time.time() - start
    
       
        if frame_num % det_step < init_frames: #Use YOLO
            # 3a. YOLO detect                            
            detections = detector.detect2(frame,dim)
            torch.cuda.synchronize(device)
            time_metrics['detect'] += time.time() - start
            
            start = time.time()
            detections = detections.cpu()
            time_metrics['gpu_load'] += time.time() - start

            
            # postprocess detections
            start = time.time()
            detections = parse_detections(detections)
            time_metrics['parse'] += time.time() - start
         
            # 4a. Match, using Hungarian Algorithm        
            start = time.time()
            
            pre_ids = []
            pre_loc = []
            for id in pre_locations:
                pre_ids.append(id)
                pre_loc.append(pre_locations[id])
            pre_loc = np.array(pre_loc)
            
            # matchings[i] = [a,b] where a is index of pre_loc and b is index of detection
            matchings = match_hungarian(pre_loc,detections[:,:4],iou_cutoff = 0.05)
            time_metrics['match'] += time.time() - start
            
            # try:
            #     start = time.time()
            #     matchings2 = match_greedy(pre_loc,detections[:,:4], threshold = 200)
            #     time_metrics['match2'] += time.time() - start
            # except:
            #     print("failed")
            
            # 5a. Update tracked objects
            start = time.time()
    
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
              
                time_metrics['update'] += time.time() - start
                  
            
            # 6a. For each detection not in matchings, add a new object
            start = time.time()
            
            new_array = np.zeros([len(detections) - len(matchings),4])
            new_ids = []
            cur_row = 0
            for i in range(len(detections)):
                if len(matchings) == 0 or i not in matchings[:,1]:
                    
                    new_ids.append(next_obj_id)
                    new_array[cur_row,:] = detections[i,:4]
    
                    fsld[next_obj_id] = 0
                    all_tracks[next_obj_id] = np.zeros([n_frames,7])
                    all_classes[next_obj_id] = np.zeros(13)
                    
                    next_obj_id += 1
                    cur_row += 1
           
            if len(new_array) > 0:        
                tracker.add(new_array,new_ids)
            
            
            # 7a. For each untracked object, increment fsld        
            for i in range(len(pre_ids)):
                try:
                    if i not in matchings[:,0]:
                        fsld[pre_ids[i]] += 1
                except:
                    fsld[pre_ids[i]] += 1
            
            # 8a. remove lost objects
            removals = []
            for id in pre_ids:
                if fsld[id] > fsld_max:
                    removals.append(id)
           
            if len(removals) > 0:
                tracker.remove(removals)    
            
            time_metrics['add and remove'] += time.time() - start


            
        elif True: # use Resnet  
            # 3b. crop tracked objects from image
            start = time.time()
            # use predicted states to crop relevant portions of frame 
            box_ids = []
            box_list = []
            
            # convert to array
            for id in pre_locations:
                box_ids.append(id)
                box_list.append(pre_locations[id][:4])
            boxes = np.array(box_list)
            
            # convert xysr boxes into xmin xmax ymin ymax
            # first row of zeros is batch index (batch is size 0) for ROI align
            new_boxes = np.zeros([len(boxes),5]) 

            # use either s or s x r for both dimensions, whichever is larger,so crop is square
            #box_scales = np.max(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1)
            box_scales = np.min(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
                
            #expand box slightly
            box_scales = box_scales * ber# box expansion ratio
            
            new_boxes[:,1] = boxes[:,0] - box_scales/2
            new_boxes[:,3] = boxes[:,0] + box_scales/2 
            new_boxes[:,2] = boxes[:,1] - box_scales/2 
            new_boxes[:,4] = boxes[:,1] + box_scales/2 
            torch_boxes = torch.from_numpy(new_boxes).float().to(device)
            
            if False: # mask other bboxes
                # these boxes are not square
                rect_boxes = np.zeros([len(boxes),4])
                rect_boxes[:,0] = boxes[:,0] - boxes[:,2] / 2.0
                rect_boxes[:,1] = boxes[:,1] - boxes[:,2] * boxes[:,3] / 2.0 
                rect_boxes[:,2] = boxes[:,0] + boxes[:,2] / 2.0
                rect_boxes[:,3] = boxes[:,1] + boxes[:,2] * boxes[:,3] / 2.0 
                rect_boxes = rect_boxes.astype(int)
                frame_copy = frame.clone()
                for rec in rect_boxes:
                    frame_copy[:,rec[1]:rec[3],rec[0]:rec[2]] = 0
                frame_copy = frame_copy.unsqueeze(0).repeat(len(boxes),1,1,1)
                
                # in each crop, replace active box with correct pixels
                for i in range(len(rect_boxes)):
                    torch_boxes[i,0] = i # so images are indexed correctly
                    rec = rect_boxes[i]
                    frame_copy[i,:,rec[1]:rec[3],rec[0]:rec[2]] = frame[:,rec[1]:rec[3],rec[0]:rec[2]]
            
            else:
                frame_copy = frame.unsqueeze(0)
            # crop using roi align 
            crops = roi_align(frame_copy,torch_boxes,(224,224))
            time_metrics['pre_localize and align'] += time.time() - start
            
            
            
            # 4b. Localize objects using localizer
            start= time.time()
            cls_out,reg_out = localizer(crops)
            torch.cuda.synchronize()
            time_metrics['localize'] += time.time() - start
            
            start = time.time()
            if  False:
                test_outputs(reg_out,crops)
            
            # store class predictions
            highest_conf,cls_preds = torch.max(cls_out,1)
            for i in range(len(cls_preds)):
                all_classes[box_ids[i]][cls_preds[i].item()] += 1
            
            
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
            
            
            #lastly, replace scale and ratio with original values 
            ## NOTE this is kind of a cludgey fix and ideally localizer should be better
            output[:,2:4] = srr*output[:,2:4] + (1-srr)*boxes[:,2:4] 
            time_metrics['post_localize'] += time.time() - start
            detections = output

            # 6b. Update tracker
            start = time.time()
            # map regressed bboxes directly to objects for update step
            tracker.update(output,box_ids)
            time_metrics['update'] += time.time() - start
            
            # 7b. increment all fslds
            for i in range(len(pre_ids)):
                    fsld[pre_ids[i]] += 1
        
        
            # Low confidence removals
            if True:
                removals = []
                locations = tracker.objs()
                for i in range(len(box_ids)):
                    if highest_conf[i] < 3 and box_ids[i] in locations:
                        removals.append(box_ids[i])
                        print("Removed low confidence object")
                tracker.remove(removals)
        
        # IOU suppression on overlapping bounding boxes
        if True:
            removals = []
            locations = tracker.objs()
            for i in locations:
                for j in locations:
                    if i != j:
                        iou_metric = iou(locations[i],locations[j])
                        if iou_metric > 0.5:
                            # determine which object has been around longer
                            if len(all_classes[i]) > len(all_classes[j]):
                                removals.append(j)
                            else:
                                removals.append(i)
            removals = list(set(removals))
            tracker.remove(removals)
            
            
        # 9. Get all object locations and store in output dict
        start = time.time()
        post_locations = tracker.objs()
        for id in post_locations:
            all_tracks[id][frame_num,:] = post_locations[id][:7]        
        time_metrics['store'] += time.time() - start  
        
        
        # 10. Plot
        start = time.time()
        if PLOT:
            plot(original_im,detections,post_locations,all_classes,class_dict,frame = frame_num)
        time_metrics['plot'] += time.time() - start
   
            
        # increment frame counter 
        if frame_num % 1000 == 0:
            print("Finished frame {}".format(frame_num))
        frame_num += 1
        torch.cuda.empty_cache()
            
    
    cv2.destroyAllWindows()
    
    del frames
    
    total_time = 0
    for key in time_metrics:
        total_time += time_metrics[key]
    
    if False:
        print("Finished file {} for det_step {}".format(track_path,det_step))
        print("\n\nTotal Framerate: {:.2f} fps".format(n_frames/total_time))
        print("---------- per operation ----------")
        for key in time_metrics:
            print("{:.3f}s ({:.2f}%) on {}".format(time_metrics[key],time_metrics[key]/total_time*100,key))


    #write final output 
        
    final_output = []
    for frame in range(n_frames):
        frame_objs = []
        
        for id in all_tracks:
            bbox = all_tracks[id][frame]
            if bbox[0] != 0:
                obj_dict = {}
                obj_dict["id"] = id
                obj_dict["class_num"] = np.argmax(all_classes[id])
                x0 = bbox[0] - bbox[2]/2.0
                x1 = bbox[0] + bbox[2]/2.0
                y0 = bbox[1] - bbox[2]*bbox[3]/2.0
                y1 = bbox[1] + bbox[2]*bbox[3]/2.0
                obj_dict["bbox"] = np.array([x0,y0,x1,y1])
                
                frame_objs.append(obj_dict)
        
        final_output.append(frame_objs)
        
    return final_output, n_frames/total_time, time_metrics



if __name__ == "__main__":
    tracker = Torch_KF("cpu",mod_err = 10, meas_err = 1, state_err = 1)
    final_output,frame_rate,time_metrics = skip_track("/home/worklab/Desktop/detrac/DETRAC-all-data/MVI_20011",tracker,det_step = 5)