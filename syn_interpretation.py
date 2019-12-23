import torch
import torch.nn as nn
import torchvision.transforms as transforms
from syn import FourLayerFNN
from data_loader import MyCustomDataset
from generate_data import painting
import  matplotlib.pyplot as plt

# model test 
def test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 2).requires_grad_()
            states, outputs = model(images)
            test_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def check_states(model, image):
    '''print the active states of every hidden layer for a given input image.'''
    with torch.no_grad():
        active_states, output = model(image)
        print(active_states)

def interpret_instance(model, image):
    states, output = model(image)
    w1, b1 = model.state_dict()['H1.weight'], model.state_dict()['H1.bias']
    w2, b2 = model.state_dict()['H2.weight'], model.state_dict()['H2.bias']
    w3, b3 = model.state_dict()['H3.weight'], model.state_dict()['H3.bias']
    w4, b4 = model.state_dict()['fc.weight'], model.state_dict()['fc.bias']
    print(w1, b1)
    print(w2.size(), b2.size())
    print(w3.size(), b3.size())
    print(w4.size(), b4.size())

if __name__ == '__main__':
    model = FourLayerFNN()
    model.load_state_dict(torch.load('syn_model.pkl'))

    # load test data, here is same to train_data
    test_loader = torch.utils.data.DataLoader(
        MyCustomDataset('./data/dataset.csv',
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size=1, shuffle=True)
#    test(model, test_loader)
    image  = test_loader.dataset.images[0]
    image = image.view(-1, 2)
#    _, outputs = model(image)
 #   _, prediction = torch.max(outputs.data, 1)
  #  print(prediction)
  #    check_states(model, image)
    interpret_instance(model, image)


