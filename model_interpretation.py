import torch
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

torch.manual_seed(1)

def active_state(x):
    '''To  calculate hidden neurons' active states.
      - x: hidden neurons outputs
      - Return a  list of hidden neuron states
    '''
    x = x.detach().numpy()
    states = x.copy()
    states[x>0] = 1
    return states.astype(int)

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.H1 = nn.Linear(2, 8)
        self.H2 = nn.Linear(8, 2)
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        states = {}
        h1 = F.relu(self.H1(x))
        h1_states = active_state(h1)
        h2 = F.relu(self.H2(h1))
        h2_states = active_state(h2)
        states['h1.state'] = h1_states
        states['h2.state'] = h2_states
        out = self.fc(h2)
        return states, out

def main():
    x = torch.randn(2)
    model = NNet()
    print(model)
    states, output = model(x)
 #   for name, param in model.state_dict().items():
  #      print(name, param.data)

    w1 = model.state_dict()['H1.weight']
    b1 = model.state_dict()['H1.bias']
    w2 = model.state_dict()['H2.weight']
    b2 = model.state_dict()['H2.bias']
    w3 = model.state_dict()['fc.weight']
    b3 = model.state_dict()['fc.bias']
    
    print('w2: ', w2.size())
    print('b2: ', b2.size())

    diag_s1 = torch.diag(torch.tensor((states['h1.state']), dtype=torch.float32))
    diag_s2 = torch.diag(torch.tensor((states['h2.state']), dtype=torch.float32))

    w1_hat, b1_hat = w1, b1
    s1w1 = torch.matmul(diag_s1, w1)
    s1b1 = torch.matmul(diag_s1, b1)
    w2_hat = torch.matmul(w2, s1w1)
    b2_hat = torch.matmul(w2, s1b1)+b2

    s2w2hat = torch.matmul(diag_s2, w2_hat)
    s2b2hat = torch.matmul(diag_s2, b2_hat)
    w3_hat = torch.matmul(w3, s2w2hat)
    b3_hat = torch.matmul(w3, s2b2hat)+b3

    print('w2_hat: ', w2_hat.size())
    print('b2_hat: ', b2_hat.size())

    weights = torch.cat((w1, w2_hat)).numpy()
    bias = torch.cat((b1, b2_hat)).numpy()
    bias = bias.reshape((10, 1))
    active_states = np.hstack((states['h1.state'], states['h2.state'])).astype(int)
    active_states = active_states.reshape((10, 1))

    weights_bias = np.append(weights, bias, axis=1)
    weights_bias_states  = np.append(weights_bias, active_states, axis=1)
    print(weights_bias_states.shape)
   # print('w3_hat: ', w3_hat)
   # print('b3_hat: ', b3_hat)

    output_file = open('./output.txt', 'ab')
    np.savetxt(output_file, weights_bias_states, delimiter=',')
    output_file.close()
if __name__ == '__main__':
    main()

