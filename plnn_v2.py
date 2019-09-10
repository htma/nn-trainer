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
    x = x.reshape(x.shape[1])
    states = x.copy()
    states[x>0] = 1
    return states.astype(int)

np.random.seed(1)
torch.manual_seed(1)
# 1. load data: from numpy to tensor
X = np.random.rand(8,28*28)
X = torch.tensor(X, dtype=torch.float32)
X = X.view(-1, 28*28)
y_1 = torch.ones((4), dtype=torch.int64)
y_2 = torch.zeros((4), dtype=torch.int64)
y = torch.stack((y_1, y_2))
y = y.view(-1)
print(X)
print(y)

train_dataset = torch.utils.data.TensorDataset(X,y)
print(len(train_dataset))
print(train_dataset[0][1])

# 2. Define model: feedforward nn(2, 4, 16, 2, 2)
class FourLayers(nn.Module):
    def __init__(self, D_in, D_H1, D_H2, D_out):
        super(FourLayers, self).__init__()
        self.H1 = nn.Linear(D_in, D_H1)
        self.H2 = nn.Linear(D_H1, D_H2)
        self.output = nn.Linear(D_H2, D_out)

    def forward(self, x):
        act_states = []
        h1 = F.relu(self.H1(x))
        h1_states = active_state(h1)
#        print('Hidden 1 active states: ', h1_states)
        h2 = F.relu(self.H2(h1))
        h2_states = active_state(h2)
 #       print('Hidden 2 active states: ', h2_states)
        act_states = np.concatenate((h1_states, h2_states), axis=0)
        print('act_states: ', act_states)
        out = self.output(h2)

        return act_states, out
    
D_in, D_out = 28*28, 2
D_H1, D_H2 = 8, 2
model = FourLayers(D_in, D_H1, D_H2, D_out)
#print(model(train_dataset[1][0]))

# 3. Train model
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True)
print(len(train_loader.dataset))

epochs = 1
learning_rate  = 0.1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# state_dict: keep the active state values, such as {state_id, state_value}
state_dict = dict()
state_id = 0 

for epoch in range(epochs):
    for data, labels in train_loader:
        optimizer.zero_grad()
        states, predictions = model(data)

        for i in range(states.shape[0]):
            state = int(''.join(list(map(str, states))), 2)
            if state not in state_dict.values():
                state_id += 1
                state_dict[state_id] = state

        print(state_dict)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
#        print('epoch : %d, loss is %.3f' % (epoch, loss.item()))
                                           

