# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:45:48 2020

@author: derek
"""
import os
import numpy as np
import random 
import math
random.seed = 0

import cv2
from PIL import Image
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms,models
from torchvision.transforms import functional as F

from detrac_localization_dataset import Localize_Dataset


# define ResNet18 based network structure
class ResNet_Localizer(nn.Module):
    
    """
    Defines a new network structure with vgg19 feature extraction and two parallel 
    fully connected layer sequences, one for classification and one for regression
    """
    
    def __init__(self):
        """
        In the constructor we instantiate some nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_Localizer, self).__init__()
        
        # remove last layers of vgg19 model, save first fc layer and maxpool layer
        self.feat = models.resnet18(pretrained=True)

        # get size of some layers
        start_num = self.feat.fc.out_features
        mid_num = int(np.sqrt(start_num))
        
        cls_out_num = 13 # car or non-car (for now)
        reg_out_num = 4 # bounding box coords
        embed_out_num = 128
        
        # define classifier
        self.classifier = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,cls_out_num,bias = True),
                          nn.Softmax(dim = 1)
                          )
        
        # define regressor
        self.regressor = nn.Sequential(
                          nn.Linear(start_num,mid_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(mid_num,reg_out_num,bias = True),
                          nn.ReLU()
                          )
        
        self.embedder = nn.Sequential(
                  nn.Linear(start_num,start_num // 3,bias=True),
                  nn.ReLU(),
                  nn.Linear(start_num // 3,embed_out_num,bias = True),
                  nn.ReLU()
                  )

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        features = self.feat(x)
        cls_out = self.classifier(features)
        reg_out = self.regressor(features)
        #embed_out = self.embedder(features)
        #out = torch.cat((cls_out, reg_out), 0) # might be the wrong dimension
        
        return cls_out,reg_out

def train_model(model, optimizer, scheduler,losses,
                    dataloaders,device, patience= 10, start_epoch = 0,
                    all_metrics = None):
        """
        Alternates between a training step and a validation step at each epoch. 
        Validation results are reported but don't impact model weights
        """
        max_epochs = 500
        
        # for storing all metrics
        if all_metrics == None:
          all_metrics = {
                  'train_loss':[],
                  'val_loss':[],
                  "train_acc":[],
                  "val_acc":[]
                  }
        
        # for early stopping
        best_loss = np.inf
        epochs_since_improvement = 0

        for epoch in range(start_epoch,max_epochs):
            for phase in ["train","val"]:
                if phase == 'train':
                    scheduler.step() # adjust learning rate after so many epochs
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                # Iterate over data.
                count = 0
                total_loss = 0
                total_acc = 0
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        cls_out,reg_out = model(inputs)
                        
                        # apply each loss function
                        loss = 0
                        for loss_fn in losses:
                            loss += loss_fn(reg_out.float(),targets[:,:4].float())
                        acc = 0
                        
                        # backpropogate loss and adjust model weights
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
        
                    # verbose update
                    count += 1
                    total_acc += acc
                    total_loss += loss.item()
                    if count % 20 == 0:
                      print("{} epoch {} batch {} -- Loss: {:03f}".format(phase,epoch,count,loss.item()))
            
            # report and record metrics
            avg_acc = total_acc/count
            avg_loss = total_loss/count
            if epoch % 1 == 0:
                print("Epoch {} avg {} loss: {:05f}  acc: {}".format(epoch, phase,avg_loss,avg_acc))
                all_metrics["{}_loss".format(phase)].append(total_loss)
                all_metrics["{}_acc".format(phase)].append(avg_acc)

                if avg_loss < best_loss:
                    # save a checkpoint
                    PATH = "/home/worklab/Desktop/checkpoints/detrac_localizer/resnet18_epoch{}.pt".format(epoch)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "metrics": all_metrics
                        }, PATH)
                
                torch.cuda.empty_cache()
                
            # stop training when there is no further improvement
            if avg_loss < best_loss:
                epochs_since_improvement = 0
                best_loss = avg_loss
            else:
                epochs_since_improvement +=1
            
            print("{} epochs since last improvement.".format(epochs_since_improvement))
            if epochs_since_improvement >= patience:
                break
                
        return model , all_metrics

def load_model(checkpoint_file,model,optimizer):
    """
    Reloads a checkpoint, loading the model and optimizer state_dicts and 
    setting the start epoch
    """
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    all_metrics = checkpoint['metrics']

    return model,optimizer,epoch,all_metrics

class Box_Loss(nn.Module):        
    def __init__(self):
        super(Box_Loss,self).__init__()
        
    def forward(self,output,target,epsilon = 1e-07):
        """ Compute the bbox iou loss for target vs output using tensors to preserve
        gradients for efficient backpropogation"""
        
        # minx miny maxx maxy
        minx,_ = torch.max(torch.cat((output[:,0].unsqueeze(1),target[:,0].unsqueeze(1)),1),1)
        miny,_ = torch.max(torch.cat((output[:,1].unsqueeze(1),target[:,1].unsqueeze(1)),1),1)
        maxx,_ = torch.min(torch.cat((output[:,2].unsqueeze(1),target[:,2].unsqueeze(1)),1),1)
        maxy,_ = torch.min(torch.cat((output[:,3].unsqueeze(1),target[:,3].unsqueeze(1)),1),1)

        zeros = torch.zeros(minx.shape).unsqueeze(1).to(device)
        delx,_ = torch.max(torch.cat(((maxx-minx).unsqueeze(1),zeros),1),1)
        dely,_ = torch.max(torch.cat(((maxy-miny).unsqueeze(1),zeros),1),1)
        intersection = torch.mul(delx,dely)
        a1 = torch.mul(output[:,2]-output[:,0],output[:,3]-output[:,1])
        a2 = torch.mul(target[:,2]-target[:,0],target[:,3]-target[:,1])
        #a1,_ = torch.max(torch.cat((a1.unsqueeze(1),zeros),1),1)
        #a2,_ = torch.max(torch.cat((a2.unsqueeze(1),zeros),1),1)
        union = a1 + a2 - intersection 
        iou = intersection / (union + epsilon)
        #iou = torch.clamp(iou,0)
        return 1- iou.sum()/(len(iou)+epsilon)
    
#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    
    checkpoint_file = None
    patience = 10

    label_dir       = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
    train_image_dir = "/home/worklab/Desktop/detrac/DETRAC-train-data"
    test_image_dir  = "/home/worklab/Desktop/detrac/DETRAC-test-data"
    
    
    # 1. CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        MULTI = True
    else:
        MULTI = False
    torch.cuda.empty_cache()   
    
    # 2. load model
    try:
        model
    except:
        model = ResNet_Localizer()
        if MULTI:
            model = nn.DataParallel(model)
    model = model.to(device)
    print("Loaded model.")
    
    
    # 3. create training params
    params = {'batch_size' : 32,
              'shuffle'    : True,
              'num_workers': 0,
              'drop_last'  : True
              }
    
    # 4. create dataloaders
    try:   
        len(train_data)
        len(test_data)
    except:   
        pos_path = "/media/worklab/data_HDD/cv_data/images/data_stanford_cars"
        neg_path = "/media/worklab/data_HDD/cv_data/images/data_imagenet_loader"
        train_data = Localize_Dataset(train_image_dir, label_dir)
        test_data =  Localize_Dataset(test_image_dir,label_dir)
        
    trainloader = data.DataLoader(train_data, **params)
    testloader = data.DataLoader(test_data, **params)
    
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
    print("Got dataloaders.")
    
    # 5. define stochastic gradient descent optimizer    
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum = 0.9)
    
    # 6. decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    # 7. define start epoch for consistent labeling if checkpoint is reloaded
    start_epoch = 0
    all_metrics = None

    # 8. if checkpoint specified, load model and optimizer weights from checkpoint
    if checkpoint_file != None:
        model,optimizer,start_epoch,all_metrics = load_model(checkpoint_file, model, optimizer)
        #model,_,start_epoch = load_model(checkpoint_file, model, optimizer) # optimizer restarts from scratch
        print("Checkpoint loaded.")
     
    # 9. define losses
    losses = [Box_Loss()]#, nn.MSELoss()]
    
    if True:    
    # train model
        print("Beginning training.")
        model = train_model(model,
                            optimizer, 
                            exp_lr_scheduler,
                            losses,
                            dataloaders,
                            device,
                            patience = patience,
                            start_epoch = start_epoch,
                            all_metrics = all_metrics)
        
    

   