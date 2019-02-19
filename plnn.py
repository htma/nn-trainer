import torch
import numpy as np
import matplotlib.pyplot as plt

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
        self.hidden1 = torch.nn.Linear(D_in, H1)
        self.hidden2 = torch.nn.Linear(H1, H2)
        self.hidden3 = torch.nn.Linear(H2, H3)
        self.output = torch.nn.Linear(H3, D_out)
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
        return state

def main():
    N, D_in, D_out = 10, 2, 2
    H1, H2, H3 = 4, 16, 2
    model = PLNN(D_in, H1, H2, H3, D_out)
    print(model)
    #batches = generate_batches(N)

    train_loader = torch.utils.data.DataLoader(
        MyCustomDataset('./data/dataset.csv',
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size=200,shuffle=False)    
 
    # paint configures
    fig, ax = plt.subplots()
    for batch_idx, (data, target)  in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        state = model(data)

        ndata = data.numpy()
        for x in ndata:
            rect = plt.Rectangle(x, 0.3, 0.3, fc=build_RGB(state))
            ax.add_patch(rect)

    ax.set(xlabel='$x_1$', ylabel='$x_2$',
           title='Train data')
    ax.xaxis.set_ticks([-1.5, -1.2, -0.9, -0.6,-0.3, 0, 0.3,0.6,0.9,1.2,1.5])
    ax.yaxis.set_ticks([-1.5, -1.2, -0.9, -0.6,-0.3, 0, 0.3,0.6,0.9,1.2,1.5])
    
    ax.grid(True)
    plt.show()
    

if __name__ == '__main__':
    main()
