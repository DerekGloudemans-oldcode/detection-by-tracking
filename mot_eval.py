"""
Description:
-----------
    Text Here
    
Parameters:
----------
    Text Here
    
Returns:
-------
    Text Here
"""
    
import sys
sys.path.append("py_motmetrics")
import motmetrics

import numpy as np
import xml.etree.ElementTree as ET
import _pickle as pickle
import matplotlib.pyplot as plt

def parse_labels(label_file):
     """
     Returns a set of metadata (1 per track) and a list of labels (1 item per
     frame, where an item is a list of dictionaries (one dictionary per object
     with fields id, class, truncation, orientation, and bbox
     """
     
     class_dict = {
         'Sedan':0,
         'Hatchback':1,
         'Suv':2,
         'Van':3,
         'Police':4,
         'Taxi':5,
         'Bus':6,
         'Truck-Box-Large':7,
         'MiniVan':8,
         'Truck-Box-Med':9,
         'Truck-Util':10,
         'Truck-Pickup':11,
         'Truck-Flatbed':12,
         
         0:'Sedan',
         1:'Hatchback',
         2:'Suv',
         3:'Van',
         4:'Police',
         5:'Taxi',
         6:'Bus',
         7:'Truck-Box-Large',
         8:'MiniVan',
         9:'Truck-Box-Med',
         10:'Truck-Util',
         11:'Truck-Pickup',
         12:'Truck-Flatbed'
         }
     
     
     tree = ET.parse(label_file)
     root = tree.getroot()
     
     # get sequence attributes
     seq_name = root.attrib['name']
     
     # get list of all frame elements
     #frames = root.getchildren()
     frames = list(root)
     # first child is sequence attributes
     seq_attrs = frames[0].attrib
     
     # second child is ignored regions
     ignored_regions = []
     for region in frames[1]:
         coords = region.attrib
         box = np.array([float(coords['left']),
                         float(coords['top']),
                         float(coords['left']) + float(coords['width']),
                         float(coords['top'])  + float(coords['height'])])
         ignored_regions.append(box)
     frames = frames[2:]
     
     # rest are bboxes
     all_boxes = []
     frame_counter = 1
     for frame in frames:
         while frame_counter < int(frame.attrib['num']):
             # this means that there were some frames with no detections
             all_boxes.append([])
             frame_counter += 1
         
         frame_counter += 1
         frame_boxes = []
         #boxids = frame.getchildren()[0].getchildren()
         boxids = list(list(frame)[0])
         for boxid in boxids:
             #data = boxid.getchildren()
             data = list(boxid)
             coords = data[0].attrib
             stats = data[1].attrib
             bbox = np.array([float(coords['left']),
                             float(coords['top']),
                             float(coords['left']) + float(coords['width']),
                             float(coords['top'])  + float(coords['height'])])
             det_dict = {
                     'id':int(boxid.attrib['id']),
                     'class':stats['vehicle_type'],
                     'class_num':class_dict[stats['vehicle_type']],
                     'color':stats['color'],
                     'orientation':float(stats['orientation']),
                     'truncation':float(stats['truncation_ratio']),
                     'bbox':bbox
                     }
             
             frame_boxes.append(det_dict)
         all_boxes.append(frame_boxes)
     
     sequence_metadata = {
             'sequence':seq_name,
             'seq_attributes':seq_attrs,
             'ignored_regions':ignored_regions
             }
     return all_boxes, sequence_metadata

def test_regions(regions,x,y):
    """
    Determines whether point (x,y) falls within any of regions
    """
    
    for region in regions:
        if x > region[0] and x < region[2] and y > region[1] and y < region[3]:
            return True
    return False

def evaluate_mot(preds,gts,ignored_regions = [],threshold = 100):
    """
    Description:
    -----------
        Evaluates a set of multiple object tracks and returns a variety of MOT
        metrics for these tracks.
        
    Parameters:
    ----------
        preds - the predicted object locations and ids
        gts   - the ground-truth object locations and ids
        ** both preds and gts are expected to be of the form list[list[dict]]:
            - one outer list item per frame
            - one inner list item per object
            - dict must contain fields id, bbox (x0,y0,x1,y1), and class
            
    Returns:
    -------
        metrics - list of MOT metrics for the tracks
    """
    
    acc = motmetrics.MOTAccumulator(auto_id = True)
    
    assert len(preds) == len(gts) , "Length of predictions and ground truths are not equal: {},{}".format(len(preds),len(gts))
         

    for frame in range(len(gts)):
        # get gts in desired format
        gt = gts[frame]
        gt_ids = [] # object ids for each object in this frame
        for obj in gt:
            
            # gx = (obj["bbox"][0] + obj['bbox'][2]) /2.0
            # gy = (obj["bbox"][1] + obj['bbox'][3]) /2.0
            # exclude = test_regions(ignored_regions,gx,gy)
            
            # if not exclude:
                gt_ids.append(obj["id"])
        gt_ids = np.array(gt_ids)
        
        
        # get preds in desired format
        pred = preds[frame]
        pred_ids = [] # object ids for each object in this frame
        for obj in pred:
            
            #pred object center
            px = (obj["bbox"][0] + obj['bbox'][2]) /2.0
            py = (obj["bbox"][1] + obj['bbox'][3]) /2.0
            exclude = test_regions(ignored_regions,px,py)
            
            if not exclude:
                pred_ids.append(obj["id"])
        pred_ids = np.array(pred_ids)
        
        # get distance matrix in desired format
        
        if False: # use distance for matching
            dist = np.zeros([len(gt_ids),len(pred_ids)])
            for i in range(len(gt_ids)):
                for j in range(len(pred_ids)):
                    # ground truth object center
                    gx = (gt[i]["bbox"][0] + gt[i]['bbox'][2]) /2.0
                    gy = (gt[i]["bbox"][1] + gt[i]['bbox'][3]) /2.0
                    
                    # pred object center
                    px = (pred[j]["bbox"][0] + pred[j]['bbox'][2]) /2.0
                    py = (pred[j]["bbox"][1] + pred[j]['bbox'][3]) /2.0
                    
                    d = np.sqrt((px-gx)**2 + (py-gy)**2)
                    dist[i,j] = d
        
        else: # use iou for matching
            dist = np.ones([len(gt_ids),len(pred_ids)])
            for i in range(len(gt_ids)):
                for j in range(len(pred_ids)):
                    minx = max(gt[i]["bbox"][0],pred[j]["bbox"][0])
                    maxx = min(gt[i]["bbox"][2],pred[j]["bbox"][2])
                    miny = max(gt[i]["bbox"][1],pred[j]["bbox"][1])
                    maxy = min(gt[i]["bbox"][3],pred[j]["bbox"][3])
                    
                    intersection = max(0,maxx-minx) * max(0,maxy-miny)
                    a1 = (gt[i]["bbox"][2] - gt[i]['bbox'][0]) * (gt[i]["bbox"][3] - gt[i]['bbox'][1])
                    a2 = (pred[j]["bbox"][2] - pred[j]['bbox'][0]) * (pred[j]["bbox"][3] - pred[j]['bbox'][1])
                    
                    union = a1+a2-intersection
                    iou = intersection / union
                    dist[i,j] = 1-iou
            
        # if detection isn't close to any object (> threshold), remove
        # this is a cludgey fix since the detrac dataset doesn't have all of the vehicles labeled
        # get columnwise min
        if False:
            mins = np.min(dist,axis = 0)
            idxs = np.where(mins < threshold)
            
            pred_ids = pred_ids[idxs]
            dist = dist[:,idxs]
        
        # update accumulator
        acc.update(gt_ids,pred_ids,dist)
        
        
    metric_module = motmetrics.metrics.create()
    summary = metric_module.compute(acc,metrics = ["num_frames",
                                                   "num_unique_objects",
                                                   "mota",
                                                   "motp",
                                                   "precision",
                                                   "recall",
                                                   "num_switches",
                                                   "mostly_tracked",
                                                   "partially_tracked",
                                                   "mostly_lost",
                                                   "num_fragmentations",
                                                   "num_false_positives",
                                                   "num_misses",
                                                   "num_switches"])
    
    return summary,acc
        

if __name__ == "__main__":
    biggest_array = np.zeros([9,18])
    
    for num in [63525,63552]:
        all_step_metrics = {}
        for det_step in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,23,26,29]:
    
            label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
            gt_file = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3/MVI_{}_v3.xml".format(num)
            pred_file = "/home/worklab/Documents/code/detection-by-tracking/preds_temp/preds_MVI_{}_{}.cpkl".format(num,det_step)
            # get gt labels
            gts,metadata = parse_labels(gt_file)
            ignored_regions = metadata['ignored_regions']
            
            # get pred labels
            try:
                with open(pred_file,"rb") as f:
                    preds,frame_rate = pickle.load(f)
            except:
                with open(pred_file,"rb") as f:
                    preds,frame_rate,time_met = pickle.load(f)
                
            metrics,acc = evaluate_mot(preds,gts,ignored_regions,threshold = 40)
            metrics = metrics.to_dict()
            metrics["framerate"] = frame_rate
            all_step_metrics[det_step] = metrics
            
        
        
        # aggregate and plot all 
        n_objs = metrics["num_unique_objects"][0]
        det_step = []
        mota = []
        motp = []
        mostly_tracked = []
        mostly_lost = []
        num_fragmentations = []
        num_switches = []
        num_fp = []
        num_fn = []
        framerate = []
        
        for d in all_step_metrics:
            det_step.append(d)
            mota.append(all_step_metrics[d]["mota"][0])
            motp.append(1.0 - ( all_step_metrics[d]["motp"][0]/ 1100))
            mostly_tracked.append(all_step_metrics[d]["mostly_tracked"][0]/n_objs)
            mostly_lost.append(all_step_metrics[d]["mostly_lost"][0]/n_objs)
            num_fragmentations.append(all_step_metrics[d]["num_fragmentations"][0])
            num_switches.append(all_step_metrics[d]["num_switches"][0])
            num_fp.append(all_step_metrics[d]["num_misses"][0])
            num_fn.append(all_step_metrics[d]["num_switches"][0])
            framerate.append(all_step_metrics[d]["framerate"])
        
        metrics_np= np.array([mota,motp,mostly_tracked,mostly_lost,num_fragmentations,num_switches,num_fp,num_fn,framerate])
        biggest_array+= metrics_np
        if False:
            plt.figure()
            plt.plot(det_step,mota)
            plt.plot(det_step,motp) #1100 = 1-(960**2 + 540**2 )**0.5, so normalize by image size
            plt.plot(det_step,mostly_tracked)
            plt.plot(det_step,mostly_lost)
            plt.plot(det_step,framerate)
            
            plt.legend(["MOTA","MOTP","MT","ML","100 Hz"])
        
    biggest_array = biggest_array / 2.0