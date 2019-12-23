import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt
import numpy as np

def active_state(x):
    '''To  calculate hidden neurons' active states.
      - x: hidden neurons outputs
      - Return a  list of hidden neuron states
    '''
    x = x.detach().numpy()
    states = x.copy()
    states[x>0] = 1
    return states.astype(int)

class ThreeLayerFNN(nn.Module):
    def __init__(self, D_in, D_out):
        super(ThreeLayerFNN, self).__init__()
        self.H1 = nn.Linear(D_in, 8)
        self.H2 = nn.Linear(8, 2)
        self.fc = nn.Linear(2, D_out)

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

def load_data():
    train_dataset = datasets.FashionMNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
    test_dataset = datasets.FashionMNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())
    train_filter = np.where((train_dataset.train_labels == 8) |
                        (train_dataset.train_labels == 9))
    test_filter = np.where((test_dataset.test_labels == 8) |
                        (test_dataset.test_labels == 9))
    train_dataset.train_data = train_dataset.train_data[train_filter]
    train_dataset.train_labels = train_dataset.train_labels[train_filter]
    train_dataset.train_labels[train_dataset.train_labels == 8] = 0
    train_dataset.train_labels[train_dataset.train_labels == 9] = 1

    test_dataset.test_data = test_dataset.test_data[test_filter]
    test_dataset.test_labels = test_dataset.test_labels[test_filter]
    test_dataset.test_labels[test_dataset.test_labels == 8] = 0
    test_dataset.test_labels[test_dataset.test_labels == 9] = 1

    train_loader = torch.utils.data.DataLoader(
               dataset=train_dataset,
               batch_size=100, 
               shuffle=True)

    test_loader = torch.utils.data.DataLoader(
              dataset=test_dataset,
               batch_size=1, 
               shuffle=False)

    return train_loader, test_loader

def train(model, epoch, train_loader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # state_dict: keep the active state values: {state_id, state_int_value}
    state_dict = dict()
    state_id = 0
    
    for batch_id, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28).requires_grad_()
        optimizer.zero_grad()
        states, outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
#        print('epoch: {} [{}/{} ({:.0f}%)]\t . Loss: {}'.format(
 #           epoch, batch_id*len(images), len(train_loader.dataset),
  #          100.*batch_id/len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28).requires_grad_()
            _, outputs = model(images)
            test_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    D_in, D_out = 28*28, 2    
    model = ThreeLayerFNN(D_in, D_out)
    train_loader, test_loader = load_data()
    for epoch in range(1, 6):
        train(model, epoch, train_loader)
        test(model, test_loader)
#    torch.save(model.state_dict(), 'fmnist4_model.pkl')

def imshow(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    trimg = np.transpose(npimg, (1,2,0))
    plt.imshow(trimg)
    plt.show()


def check_states(model):
#    print(model_new.named_parameters())

  #  image = np.random.rand(1,28*28)
   # image = torch.tensor(image, dtype=torch.float32)
    #image = image.view(-1, 28*28)

    train_loader, test_loader = load_data()
    # dataiter = iter(test_loader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))

    image = test_loader.dataset[200][0].view(-1, 28*28)
    show_img = image.numpy().reshape(28, 28)
    plt.imshow(show_img, cmap='gray')
    output_file = open('./image_200.txt', 'wb')    
    np.savetxt(output_file, show_img, delimiter=',')
    output_file.close()
    
#    plt.show()

    with torch.no_grad():
        active_states, output = model(image)
        print('active states: ', active_states)
        print('outputs: ', output)

def interpret_instance(model):
    train_loader, test_loader = load_data()
    image = test_loader.dataset[200][0].view(-1, 28*28)
    show_img = image.numpy().reshape(28, 28)
    plt.imshow(show_img, cmap='gray')
    plt.show()
    states, output = model(image)

    w1 = model.state_dict()['H1.weight']
    b1 = model.state_dict()['H1.bias']
    w2 = model.state_dict()['H2.weight']
    b2 = model.state_dict()['H2.bias']
    w3 = model.state_dict()['fc.weight']
    b3 = model.state_dict()['fc.bias']

    diag_s1 = torch.diag(torch.tensor((states['h1.state'][0]), dtype=torch.float32))
    diag_s2 = torch.diag(torch.tensor((states['h2.state'][0]), dtype=torch.float32))

    w1_hat, b1_hat = w1, b1
    s1w1 = torch.matmul(diag_s1, w1)
    s1b1 = torch.matmul(diag_s1, b1)
    w2_hat = torch.matmul(w2, s1w1)
    b2_hat = torch.matmul(w2, s1b1)+b2

    s2w2hat = torch.matmul(diag_s2, w2_hat)
    s2b2hat = torch.matmul(diag_s2, b2_hat)
    w3_hat = torch.matmul(w3, s2w2hat)
    b3_hat = torch.matmul(w3, s2b2hat)+b3

    weights = torch.cat((w1, w2_hat)).numpy()
    bias = torch.cat((b1, b2_hat)).numpy()
    bias = bias.reshape((10, 1))
    active_states = np.hstack((states['h1.state'], states['h2.state'])).astype(int)
    active_states = active_states.reshape(10, 1)

    weights_bias = np.append(weights, bias, axis=1)
    weights_bias_states  = np.append(weights_bias, active_states, axis=1)
    print(weights_bias_states)
    
 #   output_file = open('./hidden_neurons.txt', 'wb')    
  #  np.savetxt(output_file, weights_bias_states, delimiter=',')
   # output_file.close()

if  __name__ == '__main__':
#    main()
    D_in, D_out = 28*28, 2    
    model = ThreeLayerFNN(D_in, D_out)
    model.load_state_dict(torch.load('fmnist4_model.pkl'))
    print(len(list(model.parameters())))
 #   check_states(model)
    interpret_instance(model)




     



    


