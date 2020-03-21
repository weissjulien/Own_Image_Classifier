# Imports here
#---------------------------------------------------------------#
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

from workspace_utils import active_session

import PIL
from PIL import Image
from collections import OrderedDict

import numpy as np
import seaborn as sns
import random
import os
import time
import sys

# Function for command line arguments
#---------------------------------------------------------------#
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network options")
    
    parser.add_argument('data_dir',
                        type=str, 
                        help='Select a dataset directory to train the model as str.')
    
    parser.add_argument('--save_dir',
                        type=str,
                        help='Select a directory to save the trained model as str.')
    
    parser.add_argument('--arch',
                        type=str,
                        default = 'vgg16',
                        help='Select a Neural Netwrok model from torchvision.models as str.')
    
    parser.add_argument('--learning_rate',
                        type=float, 
                        default = 0.001,
                        help='Provide a learning rate for your model.classifier to be trained with as float.')
    
    parser.add_argument('--epochs',
                        type=int,
                        default = 3,
                        help='Provide a number of epochs to train your model.classifier as int.')
    
    parser.add_argument('--hidden_units',
                        type=int,
                        help='number of of hidden units for a single hidden layer architecture as int.')
    
    parser.add_argument('--gpu',
                        action="store_true", 
                        help='Put device to Cuda and use GPU instead of CPU')

    args = parser.parse_args()
    return args
                        
                        
# Function to set the data directories for training, validation and testing.
# Definition of transforms and dataloaders.
#---------------------------------------------------------------#
def prepare_data(data_directory='flowers'):
    data_dir = data_directory #'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return train_data, trainloader, validloader, testloader, cat_to_name


# Function to setup a pretrained network, freeze parameters and
# replace classifier parameters with untrained classifier network.
#---------------------------------------------------------------#    
def create_model(arch, hidden_units):
    
    exec("model = models.{}(pretrained=True)".format(arch), globals())
    try:
        input_dim = model.classifier.in_features
    except:
        input_dim = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False
    
  
    hidden_layers = len(hidden_units)
    classifier= []

    classifier.append('nn.Linear({},{}),'.format(input_dim, hidden_units[0]))                             
    classifier.append('nn.ReLU(),')
    classifier.append('nn.Dropout(p=0.2),')
                    
    if hidden_layers>1:
        for i in range(hidden_layers-1):
            classifier.append('nn.Linear({},{}),'.format(hidden_units[i], hidden_units[i+1]))    
            classifier.append('nn.ReLU(),')
            classifier.append('nn.Dropout(p=0.2),')
                            
    classifier.append('nn.Linear({},102),'.format(hidden_units[hidden_layers-1]))                 
    classifier.append('nn.LogSoftmax(dim=1))')                          
    
    classifier_string = 'nn.Sequential('
    for i in classifier:
        classifier_string +=  i
    #print(classifier_string) 
    
    exec("model.classifier = {}".format(classifier_string), globals())

    #print(model.classifier)
    #model.classifier = nn.Sequential(nn.Linear(25088, 1024),
     #                                nn.ReLU(),
      #                               nn.Dropout(p=0.2),
       #                              nn.Linear(1024, 256),
        #                             nn.ReLU(),
         #                            nn.Dropout(p=0.2),
          #                           nn.Linear(256, 102),
           #                          nn.LogSoftmax(dim=1))

    print('Model created')
    print(model.classifier)

    return model
    

# Function to train the model classifier.
#---------------------------------------------------------------#   
def train_network(epochs, device, learning_rate, trainloader, validloader, model):
    model.to(device)
  
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    epochs = epochs
    steps = 0
    running_loss = 0
    print_frequency = 5
    start_time = time.time()
    with active_session():
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
        
                optimizer.zero_grad()
        
                log_outputs = model.forward(inputs)
                loss = criterion(log_outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
                if steps % print_frequency == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            log_outputs = model.forward(inputs)
                            batch_loss = criterion(log_outputs, labels)
                    
                            test_loss += batch_loss.item()
                    
                            # Calculate accuracy
                            outputs = torch.exp(log_outputs)
                            top_p, top_class = outputs.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_frequency:.3f}.. "
                          f"validation loss: {test_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()
    final_time = time.time() - start_time    
    print('finished training in {:.0f}m {:.7f}s'.format(final_time // 60, final_time % 60))
    return optimizer
                
                
# Function to test the model classifier.
#---------------------------------------------------------------# 
def test_network(model,device):
    model.to(device)
    
    accuracy = 0
    model.eval()
    with active_session():
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                log_outputs = model.forward(inputs)
            
                    
                # Calculate accuracy
                outputs = torch.exp(log_outputs)
                top_p, top_class = outputs.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
        print(f"Test accuracy: {accuracy/len(testloader):.3f}..")
        model.train()
        return 
    
    
    
# Function to save the model as a checkpoint.
#---------------------------------------------------------------# 
def save_model(save_dir, model, optimizer, train_data):
    
    model.to('cpu')
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'batch_size':64,
                  'model':model,
                  'classifier':model.classifier,
                  'optimizer':optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx':model.class_to_idx}

    torch.save(checkpoint, save_dir)
    print('Model saved')
    return


# Main function
#---------------------------------------------------------------# 
def main():
    args = arg_parser()
    
    accepted_arch = ['vgg13', 'vgg16', 'vgg19', 'densenet121' ,'densenet161', 'densenet169']
    arch = args.arch
    if not arch in accepted_arch:
        sys.exit('Please choose among one of the following architectures: vgg13, vgg16, vgg19, densenet121, densenet161, densenet169.')
                 
    if args.hidden_units:
        hidden_units = [args.hidden_units]
    if not args.hidden_units:
        hidden_units = [1024,256]
               
    if args.save_dir:
        save_dir = args.save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_dir += '/checkpoint.pth'
    if not args.save_dir:
        save_dir = 'checkpoint.pth'
        
    if args.gpu:        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.gpu:
        device = torch.device('cpu')
        
    train_data, trainloader, validloader, testloader, cat_to_name = prepare_data(args.data_dir)
    model = create_model(arch, hidden_units)
    #optimizer = train_network(epochs, device, learning_rate, trainloader, validloader, model)
    #save_model(save_dir, model, optimizer, train_data)
    print('finished training and saving the model!')
    
# Run train.py
#---------------------------------------------------------------#     
if __name__ == '__main__': main()  