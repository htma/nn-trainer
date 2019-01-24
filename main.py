import argparse
import operator
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.autograd import Variable
from data_loader import MyCustomDataset

from model import FullyConnectedNet

# parsing arguments
parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument('--batch_size', type=int, default=200,
                    help='batch size (default: 200)')
parser.add_argument('--learning_rate', type=float, default=1e-2, help="learning rate (default:0.01)")
parser.add_argument('--max_epoch', type=int, default=1,
                    help="number of epoch (default: 1)")
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Desiables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_avaiable()

kwargs = {'num_workers':1, 'pin_memory': True} if args.cuda else {}

# loading data
train_loader = torch.utils.data.DataLoader(
    MyCustomDataset('./data/dataset.csv',
                    transform=transforms.Compose([
                        transforms.ToTensor()])),
                    batch_size=args.batch_size,
                    shuffle=False, **kwargs)

def main():
    train_loader = torch.utils.data.DataLoader(
    MyCustomDataset('./data/dataset.csv',
                    transform=transforms.Compose([
                        transforms.ToTensor()])),
        batch_size=200,
        shuffle=False)

#    train_loader = torch.utils.data.DataLoader(
 #       datasets.MNIST('../data', train=True, download=True,
  #                     transform=transforms.Compose([
   #                        transforms.ToTensor(),
    #                       transforms.Normalize((0.1307,),(0.3081,))])),
        #batch_size=200, shuffle=True)

    D_in, D_out = 2,2
    H1, H2, H3 = 4, 16, 2
    model = FullyConnectedNet(D_in, H1, H2, H3, D_out)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()

    # run the main training loop
    for epoch in range(args.max_epoch):
        for batch_idx, (data,labels) in enumerate(train_loader):
   #         print(batch_idx)
            data, labels = Variable(data), Variable(labels)
            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data),
                    len(train_loader.dataset),
                    100. * batch_idx/len(train_loader),
                    loss.data[0]))
    
if __name__ == '__main__':
    main()

