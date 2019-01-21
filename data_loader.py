# Custom Dataset
# created at 8:24 on Jan 17, 2019
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class MyCustomDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        """
        Args:
            csv_path: path to csv file
            transforms: pytorch transforms for transforms and tensor conversion
        """
        # reading a csv
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])

        #assigning transforms
        self.transforms = transforms

    def __getitem__(self, index):
        # retrun data and label
        # index is the nth data/image(as tensor) you are going to return
        label = self.labels[index].astype('uint8')

        data = np.asarray(self.data.iloc[index][1:])
  
        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)
        print('data dtype', data.dtype)
#        if self.transforms is not None:
 #           data = self.transforms(data)
        return (data, label)

    def __len__(self):
        return len(self.data.index)

if __name__ == '__main__':
    # define transorms and custom dataset
    transformations = transforms.Compose([transforms.ToTensor()])
    custom_dataset = MyCustomDataset('./data/dataset.csv', transformations)

    # Define data loader
    dataset_loader = DataLoader(dataset=custom_dataset,
                                                 batch_size=10,
                                                 shuffle=False)
    for images, labels in dataset_loader:
        print(images, labels)
    
    

