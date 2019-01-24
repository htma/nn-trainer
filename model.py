import torch
import numpy as np
import matplotlib.pyplot as plt
import  torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms

#from hidden_filter import hid_filter, build_RGB
from data_loader import MyCustomDataset

dtype = torch.float
device = torch.device("cpu")

class FullyConnectedNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(FullyConnectedNet, self).__init__()
        self.H1 = nn.Linear(D_in, H1)
        self.H2 = nn.Linear(H1, H2)
        self.H3 = nn.Linear(H2, H3)
        self.Output = nn.Linear(H3, D_out)        

    def forward(self, x):
        h1 = F.relu(self.H1(x))
        h2 = F.relu(self.H2(h1))
        h3 = F.relu(self.H3(h2))
        y_pred = self.Output(h3)
        return F.log_softmax(y_pred)

def main():
    train_loader = torch.utils.data.DataLoader(
    MyCustomDataset('./data/dataset.csv',
                    transform=transforms.Compose([
                        transforms.ToTensor()])),
        batch_size=200,
        shuffle=False)

#    train_loader = torch.utils.data.DataLoader(
 #       datasets.MNIST('../data', train=True, download=True,
  #                     transform=transforms.Compose([
   #                        transforms.ToTensor(),
    #                       transforms.Normalize((0.1307,),(0.3081,))])),
        #batch_size=200, shuffle=True)

    D_in, D_out = 2,2
    H1, H2, H3 = 4, 16, 2
    model = FullyConnectedNet(D_in, H1, H2, H3, D_out)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.NLLLoss()

    # run the main training loop
    for epoch in range(50):
        for batch_idx, (data,target) in enumerate(train_loader):
   #         print(batch_idx)
            data, target = Variable(data), Variable(target)
#            print(len(data), len(train_loader), len(train_loader.dataset))
#            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data),
                    len(train_loader.dataset),
                    100. * batch_idx/len(train_loader),
                    loss.data[0]))
    
if __name__ == '__main__':
    main()
