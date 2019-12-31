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
    file_name = './ycy_weight_bias_states.txt'
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
#    print(w3_hat.size(), b3_hat.size())

    weights = torch.cat((w1, w2_hat, w3_hat)).numpy()
    bias = torch.cat((b1, b2_hat, b3_hat)).numpy()
    bias = bias.reshape(22, 1)
    active_states = np.hstack((states['h1.state'], states['h2.state'],
                               states['h3.state'])).astype(int)
    active_states = active_states.reshape(22, 1)

    weight_bias = np.append(weights, bias, axis=1)
    weight_bias_states = np.append(weight_bias, active_states, axis=1)

    output_file = open(file_name, 'wb')
    np.savetxt(output_file, weight_bias_states, delimiter=',')
    output_file.close()
    return file_name

def calculate_feasible_range(x, file_name):
    '''
       First devide  inequality cofficients into two classes, zero-states and one-states.(that is ax + by + c <= 0 and ax+by+c > 0).
       Then calculate a feasible linear range.
       '''
    weight_bias_states = np.loadtxt(file_name, delimiter=',')
   
    # First, change the direction of inequalities with negative second column
    negative_y_states = weight_bias_states[weight_bias_states[:,1]<=0]
    processed_negative_y_states =  process_negative_ys(negative_y_states)

    positive_y_states = weight_bias_states[weight_bias_states[:,1]>0]
    new_states = np.concatenate((positive_y_states, processed_negative_y_states), axis=0)
#    print('new states: ', new_states)

    one_states = new_states[new_states[:,3]>0]
    zero_states = new_states[new_states[:,3]<=0]
    return calculate_linear_range(x, one_states, zero_states)

def process_negative_ys(negative_y_states):
    states = -1*negative_y_states
    zero_states = states[states[:,3] == 0]
    one_states = states[states[:,3] == -1]
    zero_to_one_states = np.column_stack((zero_states[:,:3], np.ones(zero_states.shape[0])))
    one_to_zero_states = np.column_stack((one_states[:,:3], np.zeros(one_states.shape[0])))
    return np.concatenate((zero_to_one_states, one_to_zero_states), axis=0)

def calculate_linear_range(x, one_states, zero_states):
    ny, py = [], [] # netgative state ys, positive state ys
    for row in zero_states:
         a, b, c, _ = row
         ny.extend([(-a*x  -c ) / b])
    npny = np.array(ny)
#    min_y = npny.min(axis=0)
    min_y = np.minimum.reduce(npny)
    
    for row in one_states:
        a, b, c,_ = row
        py.extend([(-a*x - c) / b])
    nppy = np.array(py)
    max_y = nppy.max(axis=0)

    return min_y, max_y
    
def plot_linear_range(ax, x, min_y, max_y):
    ax.fill_between(x, min_y, max_y, where=min_y>max_y, color=np.random.rand(3,), alpha=0.5)

def plot_unit_circle(ax):
    # make a simple unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    a, b = 1 * np.cos(theta), 1 * np.sin(theta)
    ax.plot(a, b, linestyle='-', linewidth=2,
             color='black', label='Unit Circle')

def plot_one_model(ax, model, x, images):
    for image in images:
        image = image.view(-1, 2)
        coefficients_file_name = calculate_ineuqality_coefficients(model, image)
        min_y, max_y = calculate_feasible_range(x, coefficients_file_name)
        plot_linear_range(ax, x, min_y, max_y)
        plot_unit_circle(ax)

def plot_multiple_models(x, images):
    
    model_name = 'syn_model'
    models = []
    fig, axes = plt.subplots(3,3)
    fig.suptitle('Linear ranges for multiple models')
    for i, ax in enumerate(axes.flat):
        model = FourLayerFNN()
        model.load_state_dict(torch.load(model_name+str(i)+'.pkl'))
        plot_one_model(ax, model, x, images)
        ax.set(xlabel='x', ylabel='y')
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        ax.set_title('model'+str(i))
        ax.label_outer()
    plt.show()

if __name__ == '__main__':
    model = FourLayerFNN()
    model.load_state_dict(torch.load('syn_model6.pkl'))
    x = np.linspace(-1.5, 1.5, 2000)

    # load test data, here is same to train_data
    test_loader = torch.utils.data.DataLoader(
        MyCustomDataset('./data/dataset.csv',
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size=1, shuffle=True)
#    test(model, test_loader)
    images  = test_loader.dataset.images
    # for image in images:
    #     image = image.view(-1,2)
    #     coefficients_file_name = calculate_ineuqality_coefficients(model, image)
    #     min_y, max_y = calculate_feasible_range(x, coefficients_file_name)
    #     plt.fill_between(x, min_y, max_y, where=min_y>max_y, color=np.random.rand(3,), alpha=0.5)
    # plt.show()
    

#    plot_one_model(model, images)
    plot_multiple_models(x, images)








    
