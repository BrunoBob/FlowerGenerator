#Autoencoder for the first part of picture generation
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("hello Uj !!!")

DATA_DIR = '../DataBase/scaledFlowersVerySmall'
SAVE_PATH = '../SaveModel/auto.pth'

def showImagesBatch(img):
    print(type(img), img.size())
    img =img / 2 + 0.5
    npImg = img.numpy()
    plt.imshow(np.transpose(npImg, (1, 2, 0)))
    plt.show()

class FlowerNet(nn.Module):
    def __init__(self):
        super(FlowerNet, self).__init__()
        self.fc1 = nn.Linear(25 * 25 * 3, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 1000)
        self.fc4 = nn.Linear(1000, 25 * 25 * 3)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        return x

    def getFlatDim(self,x):
        size = x.size()[1:]
        flatDim = 1
        for s in size:
            flatDim *= s
        return flatDim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

flowerDataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)
flowerLoader = torch.utils.data.DataLoader(flowerDataset, batch_size=4, shuffle=True, num_workers=2)

net = FlowerNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)

for epoch in range(1):
    runningLoss = 0.0
    for i, data in enumerate(flowerLoader,0):
        inputs, indexes = data
        inputs = inputs.view(-1,net.getFlatDim(inputs)).to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()
        if( i % 100 == 99):
            print("Epoch: ", epoch + 1, " - loss: ", runningLoss/100)
            runningLoss = 0.0

print("Finished training")

torch.save(net.state_dict(), SAVE_PATH)

#Show a random batch of image from dataset
#testIter = iter(flowerLoader)
#images, indexes = testIter.next()
#showImagesBatch(torchvision.utils.make_grid(images))
#images = images.view(-1,net.getFlatDim(images)).to(device)
#testOutput = net(images)
#testOutput = testOutput.view(4,3,25,25).cpu()
#showImagesBatch(torchvision.utils.make_grid(testOutput))
