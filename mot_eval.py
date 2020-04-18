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
    
    acc = motmetrics.MOTAccumulator()
    
    assert len(preds) == len(gts) , "Length of predictions and ground truths are not equal: {},{}".format(len(preds),len(gts))
         

    for frame in range(len(gts)):
        # get gts in desired format
        gt = gts[frame]
        gt_ids = [] # object ideas for each object in this frame
        for obj in gt:
            gt_ids.append(obj[id])
        gt_ids = np.array(gt_ids)
        
        
        # get preds in desired format
        pred = preds[frame]
        pred_ids = [] # object ids for each object in this frame
        for obj in pred:
            pred_ids.append(obj[id])
        pred_ids = np.array(pred_ids)
        
        # get distance matrix in desired format
        dist = np.zeros([len(gt_ids),len(pred_ids)])
        for i in range(len(gt_ids)):
            for j in range(len(pred_ids)):
                # ground truth object center
                gx = (gt[i]["bbox"][0] + gt[i]['bbox'[2]]) /2.0
                gy = (gt[i]["bbox"][1] + gt[i]['bbox'[3]]) /2.0
                
                # pred object center
                px = (pred[i]["bbox"][0] + pred[i]['bbox'[2]]) /2.0
                py = (pred[i]["bbox"][1] + pred[i]['bbox'[3]]) /2.0
            
                dist[i,j] = np.sqrt((px-gx)**2 + (py-gy)**2)
                
        # update accumulator
        acc.udpate(gt_ids,pred_ids,dist)
        