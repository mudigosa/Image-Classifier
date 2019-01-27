import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import futils

argumentparser = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

argumentparser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
argumentparser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
argumentparser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
argumentparser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
argumentparser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
argumentparser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
argumentparser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
argumentparser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

parser = argumentparser.parse_args()
where = parser.data_dir
paths = parser.save_dir
lr = parser.learning_rate
structures = parser.arch
dropout = parser.dropout
hidden_layer = parser.hidden_units
power = parser.gpu
epochs = parser.epochs


trainloader, v_loader, testloader = futils.load_data(where)


model, optimizer, criterion = futils.nn_setup(structures,dropout,hidden_layer,lr,power)


futils.train_network(model, optimizer, criterion, epochs, 20, trainloader, power)


futils.save_checkpoint(paths,structures,hidden_layer,dropout,lr)


print("All Set and Done. The Model is trained") # Coffee timeee
