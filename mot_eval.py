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
        frames = root.getchildren()
        
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
        for frame in frames:
            frame_boxes = []
            boxids = frame.getchildren()[0].getchildren()
            for boxid in boxids:
                data = boxid.getchildren()
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

def evaluate_mot(preds,gts):
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
        gt_ids = [] # object ideas for each object in this frame
        for obj in gt:
            gt_ids.append(obj["id"])
        gt_ids = np.array(gt_ids)
        
        
        # get preds in desired format
        pred = preds[frame]
        pred_ids = [] # object ids for each object in this frame
        for obj in pred:
            pred_ids.append(obj["id"])
        pred_ids = np.array(pred_ids)
        
        # get distance matrix in desired format
        dist = np.zeros([len(gt_ids),len(pred_ids)])
        for i in range(len(gt_ids)):
            for j in range(len(pred_ids)):
                # ground truth object center
                gx = (gt[i]["bbox"][0] + gt[i]['bbox'][2]) /2.0
                gy = (gt[i]["bbox"][1] + gt[i]['bbox'][3]) /2.0
                
                # pred object center
                px = (pred[j]["bbox"][0] + pred[j]['bbox'][2]) /2.0
                py = (pred[j]["bbox"][1] + pred[j]['bbox'][3]) /2.0
            
                dist[i,j] = np.sqrt((px-gx)**2 + (py-gy)**2)
                
        # update accumulator
        acc.update(gt_ids,pred_ids,dist)
        
        
    metric_module = motmetrics.metrics.create()
    summary = metric_module.compute(acc,metrics = ["num_frames","mota","motp","precision","recall","num_switches","mostly_tracked","partially_tracked","mostly_lost","num_fragmentations"])
    
    return summary,acc
        

if __name__ == "__main__":
    
    label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
    gt_file = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3/MVI_20011_v3.xml"
    pred_file = "/home/worklab/Documents/code/detection-by-tracking/preds_temp/MVI_20011_preds.cpkl"
    # get gt labels
    gts,metadata = parse_labels(gt_file)
    
    # get pred labels
    with open(pred_file,"rb") as f:
        preds = pickle.load(f)
        
    metrics,acc = evaluate_mot(preds,gts)