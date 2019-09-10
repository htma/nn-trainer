# A new version 2.0 of plnn
# created on Sep 10, 2019, Teacher's Day!

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def active_state(x):
    '''To  calculate hidden neurons' active states.
      - x: hidden neurons outputs
      - Return a  list of hidden neuron states
    '''
    x = x.detach().numpy()
    states = x.copy()
    states[x>0] = 1
    return states

#np.random.seed(1)
# 1. load data: from numpy to tensor
X = np.random.rand(5,2)
X = torch.tensor(X, dtype=torch.float32)
X = X.view(-1, 2)
y = torch.ones((5), dtype=torch.float32)
y = y.view(-1)
print(X)
print(y.size())

train_dataset = torch.utils.data.TensorDataset(X,y)
print(len(train_dataset))
print(train_dataset[0][1])

# 2. Define model: feedforward nn(2, 4, 16, 2, 2)
class FourLayers(nn.Module):
    def __init__(self, D_in, D_H, D_out):
        super(FourLayers, self).__init__()
        self.H1 = nn.Linear(D_in, D_H)
        self.output = nn.Linear(D_H, D_out)

    def forward(self, x):
        act_states = []
        h1 = F.relu(self.H1(x))
#        print('Hidden Layer1 outputs:', h1)
        act_states = active_state(h1)
        print('Hidden 1 active states: ', act_states)
        out = self.output(h1)

        return out
    
D_in, D_out = 2, 2
D_H = 4
model = FourLayers(D_in, D_H, D_out)
print(model(train_dataset[1][0]))

# 3. Train model
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True)
print(len(train_loader.dataset))

for data, label in train_loader:
    data = data.view(-1, 2)
    model(data)
                                           

