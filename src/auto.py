#Autoencoder for the first part of picture generation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

print("hello Uj !!!")

DATA_DIR = '../DataBase/'

flowerDataset = ImageFolder(DATA_DIR)
img = flowerDataset.__getitem__(0)
print(type(img[0]))
plt.imshow(img[0])
plt.show()
