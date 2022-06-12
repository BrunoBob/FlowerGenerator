import autoencoder
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def main():
    print('hello world')
    
    gpu = torch.device("cuda")
    print(gpu)
    
    model = autoencoder.AE()
    model = model.to(gpu)
    
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    print(mem)
    
    loss_function = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)
    
    transform = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.Grayscale(),
    transforms.ToTensor()])

    data = datasets.ImageFolder('../DataBase', transform=transform)

    validation_ratio = .2
    test_ratio = .1

    validation_len = int(len(data) * validation_ratio)
    test_len = int(len(data) * test_ratio)
    train_len = len(data) - validation_len - test_len 

    train_set, val_set, test_set = torch.utils.data.random_split(data, [8, 24, 8157])
    #train_set, val_set, test_set = torch.utils.data.random_split(data, [train_len, validation_len, test_len])

    train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = 1, shuffle = True)
    val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = 8, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = 8, shuffle = True)
    
    epochs = 1
    outputs = []
    losses = []
    
    for epoch in range(epochs):
        for (image, _) in train_loader:
            
            image = image.reshape(-1, 128*128)
            image = image.to(gpu, non_blocking=True)

            output = model(image)
            
            loss = loss_function(output, image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
    for current_loss in losses:
        print(current_loss)
    
if __name__ == '__main__':
    main()