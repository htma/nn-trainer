import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_loader import MyCustomDataset

    # loading training data
train_loader = torch.utils.data.DataLoader(
    MyCustomDataset('./data/dataset.csv',
                    transform=transforms.Compose([
                        transforms.ToTensor()])),
    batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    MyCustomDataset('./data/dataset.csv',
                    transform=transforms.Compose([
                        transforms.ToTensor()])),
    batch_size=100, shuffle=False)


print(train_loader.dataset[0][0])
print(len(test_loader.dataset))

class FourLayerFNN(nn.Module):
    def __init__(self):
        super(FourLayerFNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 16)
        self.fc3 = nn.Linear(16, 2)
        self.fc4 = nn.Linear(2, 2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

model = FourLayerFNN()
#print(model)

num_epochs = 500
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Train model
iter = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 2).requires_grad_()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # Updating parameters
        optimizer.step()
        iter += 1
        if iter % 100 == 0:
            correct, total = 0, 0
            for images, labels in train_loader:
                images = images.view(-1, 2).requires_grad_()
                outputs = model(images)
                _, prediction = torch.max(outputs.data, 1)
 #               print('prediction: ', prediction)
                correct += (prediction == labels).sum()
                total += labels.size(0)
            accuracy = 100.* correct/total
                
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


# Save model
save_model = True
if save_model is True:
    # Saves only parameters
    torch.save(model.state_dict(), 'syn_model.pkl')

        
