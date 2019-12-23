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

def load_data():
    train_dataset = datasets.FashionMNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
    test_dataset = datasets.FashionMNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
               dataset=train_dataset,
               batch_size=100, 
               shuffle=False)

    test_loader = torch.utils.data.DataLoader(
              dataset=test_dataset,
               batch_size=1, 
               shuffle=False)

    return train_loader, test_loader


def sort_by_first(lst):
    ''' 
     Sort a list lst of  triples by the first element x of it, such as (x, i, j).
     Return a set of index tuple (i, j)  in the triple lst.
    
    '''
    sort_lst = lst.sort()
    result = []
    for x in sort_lst:
        result.append(x[0]
    return result

def main():
    train_loader, test_loader = load_data()
    image = train_loader.dataset[200][0]
    mask = torch.zeros(image.size())
    mask[:,20,:] = 1
    mask[:,21,:] = 1
    mask[:,22,:] = 1
    print(mask.size())

    image = image*mask
    show_img = image.numpy().reshape(28, 28)
    plt.imshow(show_img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    # main()
