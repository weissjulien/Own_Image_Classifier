# Imports here
#---------------------------------------------------------------#
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import json

from workspace_utils import active_session

import PIL
from PIL import Image
from collections import OrderedDict

import numpy as np
import random
import os
import time


# Function for command line arguments
#---------------------------------------------------------------#    
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network options")
    
    parser.add_argument('input_image',
                        type=str, 
                        help='Select a path to an image as str.')
    
    parser.add_argument('checkpoint',
                        type=str, 
                        help='Select a path to a trained model checkpoint as str.')
    
    parser.add_argument('--top_k',
                        type=int,
                        default = 5,
                        help='Provide an number for most likely classes as int')  
    
    parser.add_argument('--category_names',
                        type=str, 
                        help='Use a mapping of categories to real names as str.')
    
    parser.add_argument('--gpu',
                        action="store_true", 
                        help='Put device to Cuda and use GPU instead of CPU')

    args = parser.parse_args()
    return args


# Function to load a checkpoint of a trained model.
#---------------------------------------------------------------#
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])#, strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    #epochs = checkpoint['epochs']
    for param in model.parameters():
            param.requires_grad = False
  
    #print('model loaded')    
    return model, checkpoint['class_to_idx']

 
# Function to process an input image so that the model can make a prediction on it.
#---------------------------------------------------------------#      
def process_image(input_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    im = Image.open(input_image)
    
    width, height = im.size
    im_ratio = width / height
    if im_ratio >= 1:
        new_height = 256
        new_width = round(256 * im_ratio)
        im.thumbnail((new_width, new_height))
    else:
        new_width = 256
        new_height = round(256 / im_ratio)
        im.thumbnail((new_width, new_height))
        
    width, height = im.size    
    new_width = 224
    new_height = 224
    left = round((width - new_width)/2)
    upper = round((height -new_height)/2)
    right = round((width + new_width)/2)
    lower = round((height + new_height)/2)
    im = im.crop((left, upper, right, lower))
    
    np_im = np.array(im)
    np_im = np_im / 255
    
    np_mean = np.array([0.485, 0.456, 0.406])
    np_std = np.array([0.229, 0.224, 0.225])
    
    norm_im = (np_im - np_mean) / np_std

    norm_im = norm_im.transpose((2, 0, 1)) 
    return norm_im
    

# Function to predict the top K categories of an input image.
#---------------------------------------------------------------#  
def predict(torch_image, model, topk, use_gpu, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    image = torch_image
    image.unsqueeze_(0)
    
    model.eval()
    
    if use_gpu and torch.cuda.is_available():
        print("Predicting with GPU...")
        image = image.cuda()
        model.cuda()
        
    with torch.no_grad():
        log_output = model.forward(image)
        output = torch.exp(log_output)
    
    top_p, top_class = output.topk(topk, dim=1)
    top_p = top_p[0].cpu().numpy().tolist()
    top_class_idx = top_class[0].cpu().numpy().tolist()
       
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    top_class = [idx_to_class[i] for i in top_class_idx]
    
    if cat_to_name == None:
        return top_p, top_class, cat_to_name
    else:
        top_flower = [cat_to_name[i] for i in top_class]
        return top_p, top_class, top_flower

   
# 'flowers/test/46/image_00996.jpg'

    
# Main function
#---------------------------------------------------------------#     
def main():
    args = arg_parser()
   
    if args.gpu:
        use_gpu=True
    if not args.gpu:
        use_gpu=False
      
    flower_class = args.input_image.split('/')[2]
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            flower_name = cat_to_name[flower_class]
            print('flower name: {}'.format(flower_name))
    if not args.category_names:
        cat_to_name = None
        print('flower class: {}'.format(flower_class))
    
    model, class_to_idx = load_checkpoint(args.checkpoint)
    proc_image = process_image(args.input_image)
    #torch_img = torch.from_numpy(proc_image).type(torch.FloatTensor)
    torch_img = torch.from_numpy(proc_image).float()
    top_p, top_class, top_flower = predict(torch_img, model, top_k, use_gpu, cat_to_name)
    
    print('Predicted class probabilities:')
    print(top_p)
    print('Predicted flower class:')
    print(top_class)
    print('Predicted flower name:')
    print(top_flower)
    print('finished providing my predicition!')
    
    
# Run predict.py
#---------------------------------------------------------------#     
if __name__ == '__main__': main()      