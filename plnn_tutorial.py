from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from data_loader import MyCustomDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 16)
        self.fc3 = nn.Linear(16, 2)
        self.fc4 = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('########', batch_idx, len(data), len(train_loader.dataset), len(train_loader))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        metavar='LR', help='learning rate default(: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True, help='desables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50,
                        metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For saving the current model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_avaiable()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # loading training data
    train_loader = torch.utils.data.DataLoader(
        MyCustomDataset('./data/dataset.csv',
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # training the model
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    for epoch in range(1, args.epochs+1):
        train(args, model, device, train_loader, optimizer, epoch)

    if (args.save_model):
        torch.save(model.state_dict(), 'mnist_cnn.pt')

if __name__ == '__main__':
    main()
        
    
                        
    
    
    