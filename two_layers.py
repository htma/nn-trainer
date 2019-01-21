import torch
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

dtype = torch.float
device = torch.device("cpu")

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        N, D_in, H, D_out = 2, 2, 2, 1
        self.w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
        self.w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    def forward(self, x):
        x = x.mm(self.w1)
        h_relu = F.relu(x)
        y_pred = h_relu.mm(self.w2)

        return y_pred

def train(N=2, D_in=2, H=2, D_out=1, learning_rate=1e-2, epoches=10):
    # N is batch size;D_in is input dimension;
    # H is hidden dimension;  D_out is output dimentsion.

    x = -1.5 + np.random.rand(N, D_in)*3
    y = np.random.rand(N, D_out)
    x = torch.from_numpy(x)
    x = x.type(torch.FloatTensor)
    y = torch.from_numpy(y)
    y = y.type(torch.FloatTensor)

    nnet = NNet()
    
    for t in range(epoches):
        y_pred = nnet(x)
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())

    # Use autograd to compute the backward pass.
        loss.backward()

    # Update weigths using gradient descent.
        with torch.no_grad():
            nnet.w1 -= learning_rate * nnet.w1.grad
            nnet.w2 -= learning_rate * nnet.w2.grad

        # Manully zero the gradients after updating weights
            nnet.w1.grad.zero_()
            nnet.w2.grad.zero_()
        

if __name__ == '__main__':
    train()

