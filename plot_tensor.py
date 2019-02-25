# Given a tensor, plot its data.

import torch
import matplotlib.pyplot as plt

def paint_tensor(data):
    '''
       Paint a tensor by translating it into a numpy array.
       Args:
         -data: a tensor.
    '''       
    npdata = data.numpy()
    x, y = npdata[:, 0], npdata[:,1]
    plt.plot(x, y, 'ob')
    plt.show()

def write_tensor(data, output_file):
    '''
      Writing a tensor into a file.
    '''
    npdata = data.numpy()
    x, y = npdata[:, 0], npdata[:, 1]
    with open(output_file, 'w') as output:
        for i in range(len(x)):
            output.write(str(x[i]) + ',')
            output.write(str(y[i]) + '\n')

    output.close()

def main():
    torch.manual_seed(1)
    data = torch.rand(4,2)
    print(data)
    paint_tensor(data)    
    write_tensor(data, './data/output.csv')

if __name__ == '__main__':
    main()
