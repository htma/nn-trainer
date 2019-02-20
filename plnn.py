import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms

from data_loader import MyCustomDataset

from hidden_filter import hid_filter, build_RGB
#from generate_data import generate_batches

dtype = torch.float
device = torch.device("cpu")

class PLNN(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(PLNN, self).__init__()
        self.hidden1 = nn.Linear(D_in, H1)
        self.hidden2 = nn.Linear(H1, H2)
        self.hidden3 = nn.Linear(H2, H3)
        self.output = nn.Linear(H3, D_out)
        # Create random Tensors for  weights.
        # Setting requires_grad=True indicates that we want to compute
        # gradients w.r.t. these Tensors during the backward pass.
        # self.w1 = torch.randn(D_in, H1, device=device, dtype=dtype, requires_grad=True)
        # self.w2 = torch.randn(H1, H2, device=device, dtype=dtype, requires_grad=True)
        # self.w3 = torch.randn(H2, H3, device=device, dtype=dtype, requires_grad=True)
        # self.w4 = torch.randn(H3, D_out, device=device, dtype=dtype, requires_grad=True)
        

    def forward(self, x):
        state = [] # the states of all hidden layers
        # Forward pass
        h1 = self.hidden1(x)
        h1 = hid_filter(h1)
        state.extend(h1.detach().numpy()[0])
        h1_relu = h1.clamp(min=0)
        h2 = self.hidden2(h1_relu)
        h2 = hid_filter(h2)
        state.extend(h2.detach().numpy()[0])
        h2_relu = h2.clamp(min=0)
        h3 = self.hidden3(h2_relu)
        h3 = hid_filter(h3)
        state.extend(h3.detach().numpy()[0])
        h3_relu = h3.clamp(min=0)
        y_pred = self.output(h3_relu)
        state = list(map(int, state))
        
        return state, F.log_softmax(y_pred)

def main():
    N, D_in, D_out = 10, 2, 2
    H1, H2, H3 = 4, 16, 2
    model = PLNN(D_in, H1, H2, H3, D_out)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.NLLLoss()
    
    print(model)
    #batches = generate_batches(N)

    train_loader = torch.utils.data.DataLoader(
        MyCustomDataset('./data/dataset.csv',
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size=200,shuffle=False)    

    # define  an unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    a, b = 1 * np.cos(theta), 1 * np.sin(theta)
  
    # training
    for epoch in range(3):
        fig, ax = plt.subplots()
        for batch_idx, (data, target)  in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            state,predictions = model(data)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()

            # painting training  results
            ndata = data.numpy()
            print('ndata is ', ndata)
            for x in ndata:
                rect = plt.Rectangle(x, 0.1, 0.1, fc=build_RGB(state))
                ax.add_patch(rect)

            plt.plot(a, b, linestyle='-', linewidth=2, label='Circle')
            ax.set(xlabel='$x_1$', ylabel='$x_2$', title='Train data')
            ax.xaxis.set_ticks([-1.5, -1.2, -0.9, -0.6,-0.3, 0, 0.3,0.6,0.9,1.2,1.5])
            ax.yaxis.set_ticks([-1.5, -1.2, -0.9, -0.6,-0.3, 0, 0.3,0.6,0.9,1.2,1.5])
    
            ax.grid(True)
            plt.show()

            # printing training results
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data),
                    len(train_loader.dataset),
                    100. * batch_idx/len(train_loader),
                    loss.data[0]))


if __name__ == '__main__':
    main()
