#Autoencoder for the first part of picture generation
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("hello Uj !!!")

DATA_DIR = '../DataBase/scaledFlowers'

def showImagesBatch(img):
    print(type(img), img.size())
    img =img / 2 + 0.5
    npImg = img.numpy()
    plt.imshow(np.transpose(npImg, (1, 2, 0)))
    plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

flowerDataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)
flowerLoader = torch.utils.data.DataLoader(flowerDataset, batch_size=4, shuffle=True, num_workers=2)

testIter = iter(flowerLoader)
images, indexes = testIter.next()
showImagesBatch(torchvision.utils.make_grid(images))