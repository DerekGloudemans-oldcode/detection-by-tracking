"""
This file provides a dataset class for working with the UA-detrac tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - training/testing loader mode (random images from across all tracks) using __getitem__()
    - track mode - returns a single image, in order, using __next__()
"""

import os
import numpy as np
import random 
import math
random.seed = 0

import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

try:
    from detrac_files.detrac_plot_utils_copy import pil_to_cv, plot_bboxes_2d
except:
    from detrac_plot_utils_copy import pil_to_cv, plot_bboxes_2d


class Track_Dataset(data.Dataset):
    """
    Creates an object for referencing the UA-Detrac 2D object tracking dataset
    and returning single object images for localization. Note that this dataset
    does not automatically separate training and validation data, so you'll 
    need to partition data manually by separate directories
    """
    
    def __init__(self, label_dir):
        """ initializes object
        image dir - (string) - a directory containing a subdirectory for each track sequence
        label dir - (string) - a directory containing a label file per sequence
        """

        
        # parse labels and store in dict keyed by track name
        label_list = []
        for item in os.listdir(label_dir):
            name = item.split("_v3.xml")[0].split("MVI_")[-1]
            if int(name) in  [20012,20034,63525,63544,63552,63553,63554,63561,63562,63563]:
                print("Removed Validation tracks, gotta maintain data separation!")
                continue

            detections = self.parse_labels(os.path.join(label_dir,item))[0]
            
            objects = {}
            
            
            for frame in detections:
                for item in frame:
                    id = item['id']
                    bbox = item['bbox']
                    if id in objects.keys():
                        objects[id].append(bbox)
                    else:
                        objects[id] = [bbox]
                        
            # get rid of object ids and just keep list of bboxes
            for id in objects:
                label_list.append(np.array(objects[id]))
                
        self.label_list = label_list
        
        # parse_labels returns a list (one frame per index) of lists, where 
        # each item in the sublist is one object
        # so we need to go through and keep a running record of all objects, indexed by id
            
        
        
        


    def __len__(self):
        """ returns total number of frames in all tracks"""
        return len (self.label_list)

    def __getitem__(self, index, n = 8):
        
        data = self.label_list[index]
        
        # if track is too short, just use the next index instead
        while len(data) <= n:
            index = (index + 1) % len(self.label_list)
            data = self.label_list[index]
        
        start = np.random.randint(0,len(data)-n)
        return data[start:start+n,:]
        
        
    def parse_labels(self,label_file):
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
        

if __name__ == "__main__":
    #### Test script here
    try:
        test
    except:
        try:
            label_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\DETRAC-Train-Annotations-XML-v3"
            test = Track_Dataset(label_dir)
    
        except:
            label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
            test = Track_Dataset(label_dir)
    idx = np.random.randint(0,len(test))
    
    cv2.destroyAllWindows()