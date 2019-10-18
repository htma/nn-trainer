import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data.dataset import Dataset

train_dataset = datasets.FashionMNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = datasets.FashionMNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

train_filter = np.where((train_dataset.train_labels == 0) |
                        (train_dataset.train_labels == 1))
test_filter = np.where((test_dataset.test_labels == 0) |
                        (test_dataset.test_labels == 1))

train_dataset.train_data = train_dataset.train_data[train_filter]
train_dataset.train_labels = train_dataset.train_labels[train_filter]
test_dataset.test_data = test_dataset.test_data[test_filter]
test_dataset.test_labels = test_dataset.test_labels[test_filter]

print(train_dataset)
print(test_dataset)

batch_size = 100
n_iters = 6000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(
               dataset=train_dataset,
               batch_size=batch_size, 
               shuffle=True)

test_loader = torch.utils.data.DataLoader(
               dataset=test_dataset,
               batch_size=batch_size, 
               shuffle=False)

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    trimg = np.transpose(npimg, (1,2,0))
    plt.imshow(trimg)
    plt.show()

# print(fmnist_loader)
dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

class ThreeLayerFNN(nn.Module):
    def __init__(self, D_in, D_out):
        super(ThreeLayerFNN, self).__init__()
        self.fc1 = nn.Linear(D_in, 8)
#        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)
 #       self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, D_out)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

D_in, D_out = 28*28, 2
model = ThreeLayerFNN(D_in, D_out)
print(model)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Train model
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28).requires_grad_()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # Updating parameters
        optimizer.step()
        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images to a Torch Variable
                images = images.view(-1, 28*28).requires_grad_()

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
         #       print('predicted: ', predicted)
                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


