import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from train import load_model

import json
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Image to predict')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--labels', type=str, help='JSON file containing label names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args, _ = parser.parse_known_args()


# Implement the code to predict the class from an image file
def predict(image, checkpoint, topk=5, labels='', gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use command line values when specified
    if args.image:
        image = args.image     
        
    if args.checkpoint:
        checkpoint = args.checkpoint

    if args.topk:
        topk = args.topk
            
    if args.labels:
        labels = args.labels

    if args.gpu:
        gpu = args.gpu
    
    # Load the checkpoint
    checkpoint_dict = torch.load(checkpoint)

    arch = checkpoint_dict['arch']
    num_labels = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']
        
    model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        model.cuda()
        
    was_training = model.training    
    model.eval()
    
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np.transpose(image, (1, 2, 0)) - mean)/std    
    image = np.transpose(image, (2, 0, 1))
    
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0) # this is for VGG
    
    if gpu and torch.cuda.is_available():
        image = image.cuda()
            
    result = model(image).topk(topk)

    if gpu and torch.cuda.is_available():
        # Added softmax here as per described here:
        # https://github.com/pytorch/vision/issues/432#issuecomment-368330817
        probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        classes = result[1].data.cpu().numpy()[0]
    else:       
        probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        classes = result[1].data.numpy()[0]


    if labels:
        with open(labels, 'r') as f:
            cat_to_name = json.load(f)

        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]
        
    model.train(mode=was_training)

    # Print only when invoked by command line 
    if args.image:
        print('Predictions and probabilities:', list(zip(classes, probs)))
    
    return probs, classes

# Perform predictions if invoked from command line
if args.image and args.checkpoint:
predict(args.image, args.checkpoint)