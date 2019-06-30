import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms

from data_loader import MyCustomDataset

from hidden_neuron_status import neuron_active, build_RGB
#from generate_data import generate_batches

dtype = torch.float
batch_size = 10
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
        # Forward pass
        h1 = self.hidden1(x)
        h1_state = neuron_active(h1)
    #    print('state_h1 shape is ', h1_state.shape)
        h1_relu = h1.clamp(min=0)
        h2 = self.hidden2(h1_relu)
        h2_state = neuron_active(h2)
        # states: hidden layer active states, i.e., configurations
        states = np.concatenate((h1_state, h2_state), axis=1)
   #     print('state_h2 shape is ', states.shape)
        
        h2_relu = h2.clamp(min=0)
        h3 = self.hidden3(h2_relu)
        h3_state = neuron_active(h3)
        states = np.concatenate((states, h3_state), axis=1)
  #      print('state_h3 shape is ', states.shape)

        h3_relu = h3.clamp(min=0)
        y_pred = self.output(h3_relu)
        
        return states, F.log_softmax(y_pred, dim=1)

def store_states(states):
    ''' Store states into a dict.'''
    result = dict()
    for i in range(states.shape[0]):
        state = int(''.join(list(map(str, states[0, :]))), 2)
        if state not in result.values():
            result[i] = state

    return result
        
def main():
    D_in, D_out = 2, 2
    H1, H2, H3 = 4, 16, 2
    model = PLNN(D_in, H1, H2, H3, D_out)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
   
    
    print(model)
    #batches = generate_batches(N)

    train_loader = torch.utils.data.DataLoader(
        MyCustomDataset('./data/dataset.csv',
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size=10,shuffle=False)    

    # state_dict: keep the active state values, such as {state_id, state_value}
    state_dict = dict()
    state_id = 0 

    for batch_idx, (data, target)  in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        states,predictions = model(data)
        for i in range(states.shape[0]):
            state = int(''.join(list(map(str, states[0,:]))), 2)
            if state not in state_dict.values():
                id += 1
                state_dict[id] = state

        print('state dict is:  ', state_dict)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()

