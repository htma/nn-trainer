import torch
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from data_loader import MyCustomDataset

dtype = torch.float
device = torch.device("cpu")

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        N, D_in, H, D_out = 2, 2, 2, 2
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        #self.w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
        #self.w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

def train(batch_size=10, learning_rate=1e-2, epochs=10, log_interval=100):
    #    x = x.type(torch.FloatTensor)
    transformations = transforms.Compose([transforms.ToTensor()])
    custom_dataset = MyCustomDataset('./data/dataset.csv', transformations)

    # Define data loader
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                 batch_size=10,
                                                 shuffle=False)

    nnet = NNet()
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(nnet.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.NLLLoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
          #  data = data.view(-1, 1*2)
             
            optimizer.zero_grad()
            net_out = nnet(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))    
if __name__ == '__main__':
    train()

