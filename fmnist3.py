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

class ThreeLayerFNN(nn.Module):
    def __init__(self, D_in, D_out):
        super(ThreeLayerFNN, self).__init__()
        self.fc1 = nn.Linear(D_in, 8)
        self.fc2 = nn.Linear(8, 2)
        self.fc3 = nn.Linear(2, D_out)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

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
    
    for batch_id, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28).requires_grad_()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('epoch: {} [{}/{} ({:.0f}%)]\t . Loss: {}'.format(
            epoch, batch_id*len(images), len(train_loader.dataset),
            100.*batch_id/len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28).requires_grad_()
            outputs = model(images)
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
    model.load_state_dict(torch.load('fmnist2_model.pkl'))
    train_loader, test_loader = load_data()
    for epoch in range(1, 6):
        train(model, epoch, train_loader)
        test(model, test_loader)
#    torch.save(model.state_dict(), 'fmnist2_model.pkl')

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

    image = test_loader.dataset[0][0].view(-1, 28*28)
    show_img = image.numpy().reshape(28, 28)
    plt.imshow(show_img, cmap='gray')
    plt.show()

    with torch.no_grad():
        output = model(image)
        print(output)

if  __name__ == '__main__':
    #main()
    D_in, D_out = 28*28, 2    
    model = ThreeLayerFNN(D_in, D_out)
    model.load_state_dict(torch.load('fmnist2_model.pkl'))
    check_states(model)




     



    


