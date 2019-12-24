import numpy as np
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

def calculate_ineuqality_coefficients(model, image):
    states, output = model(image)
    w1, b1 = model.state_dict()['H1.weight'], model.state_dict()['H1.bias']
    w2, b2 = model.state_dict()['H2.weight'], model.state_dict()['H2.bias']
    w3, b3 = model.state_dict()['H3.weight'], model.state_dict()['H3.bias']
    w4, b4 = model.state_dict()['fc.weight'], model.state_dict()['fc.bias']
    
    diag_s1 = torch.diag(torch.tensor((states['h1.state'][0]),
                                      dtype=torch.float32))
    w2_hat = torch.matmul(w2, torch.matmul(diag_s1, w1))
    b2_hat = torch.matmul(w2, torch.matmul(diag_s1, b1)) + b2

    diag_s2 = torch.diag(torch.tensor((states['h2.state'][0]),
                                      dtype=torch.float32))

    w3_hat = torch.matmul(w3, torch.matmul(diag_s2, w2_hat))
    b3_hat = torch.matmul(w3, torch.matmul(diag_s2, b2_hat)) + b3
    print(w3_hat.size(), b3_hat.size())

    weights = torch.cat((w1, w2_hat, w3_hat)).numpy()
    bias = torch.cat((b1, b2_hat, b3_hat)).numpy()
    bias = bias.reshape(22, 1)
    active_states = np.hstack((states['h1.state'], states['h2.state'],
                               states['h3.state'])).astype(int)
    active_states = active_states.reshape(22, 1)

    weight_bias = np.append(weights, bias, axis=1)
    weight_bias_states = np.append(weights_bias, active_states, axis=1)

    output_file = open('./syn_weight_bias_states.txt', 'wb')
    np.savetxt(output_file, weight_bias_states, delimiter=',')
    output_file.close()

def calculate_feasible_range(file_name):
    '''calculate a feasible range for inequalities such as : ax + by + c <= 0 or ax+by+c >0.'''
    weight_bias_states = np.loadtxt(file_name, delimiter=',')

    positive_states = weight_bias_states[weight_bias_states[:,3]>0]
    negative_states = weight_bias_states[weight_bias_states[:,3]==0]
    net_positive_states = positive_states[positive_states[:,1]>0]
    net_negative_states = negative_states[negative_states[:,1]>0]

    negative_y_states = negative_states[negative_states[:,1] < 0]
    negative_y_states = -1*negative_y_states
    positive_y_states = np.column_stack((negative_y_states[:,:3], np.ones(negative_y_states.shape[0])))
    net_positive_states = np.concatenate((net_positive_states,positive_y_states), axis=0)
    print(net_positive_states)

    x = np.linspace(-1.5, 1.5, 1000)
    ny, py = [], [] # netgative state ys, positive state ys
    for row in net_negative_states:
        a, b, c, _ = row
        ny.extend([(a*x + c ) / b])
    min_y = np.minimum.reduce([ny[0], ny[1], ny[2], ny[3], ny[4], ny[5]])

    for row in net_positive_states:
        a, b, c,_ = row
        py.extend([(a*x + c) / b])

    max_y = np.maximum.reduce([py[0], py[1],py[2],py[3],py[4],
                               py[5],py[6],py[7], py[8], py[9],
                               py[10], py[11],py[12],py[13],py[14]])
        
    plt.plot(x, min_y)
    plt.plot(x, max_y)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    plt.show()
    
    
def interpret_instance(file_name):
    weight_bias_states = np.loadtxt(file_name, delimiter=',')
    plot_lines(weight_bias_states)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()

def plot_lines(weight_bias_states):
    for row in weight_bias_states[:3,:]:
        a, b, c, _ = row
        print('a: ', a, '; b:', b, '; c:', c)
        plot_line(a, b, c)

def plot_line(a, b, c):
    x = np.linspace(-1.5, 1.5, 1000)
    y = -(a*x + c) / b
    plt.plot(x, y)

    
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
    
#    interpret_instance('./syn_weight_bias_states.txt')
    calculate_feasible_range('./syn_weight_bias_states.txt')

