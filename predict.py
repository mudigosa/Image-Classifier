import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import futils

#Command Line Arguments

argumentparser = argparse.ArgumentParser(
    description='predict-file')
argumentparser.add_argument('input_img', default='mudigosa/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
argumentparser.add_argument('checkpoint', default='/home/workspace/mudigosa/checkpoint.pth', nargs='*', action="store",type = str)
argumentparser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
argumentparser.add_argument('--category_names', dest="category_names", action="store", default='mudigosa/mapping/cat_to_name.json')
argumentparser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

parser = argumentparser.parse_args()
path_images = parser.input_img
number_of_outputs = parser.top_k
power = parser.gpu
input_img = parser.input_img
path = parser.checkpoint
category_names = parser.category_names



training_loader, testing_loader, validation_loader = futils.load_data()


futils.load_checkpoint(path)


with open('category_names', 'r') as json_file:
    cat_to_name = json.load(json_file)


probability = futils.predict(path_images, model, number_of_outputs, power)


labels = [cat_to_name[str(index + 1)] for index in np.array(probability[1][0])]
probabilities = np.array(probability[0][0])


i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probabilities[i]))
    i += 1

print("Here you are")
