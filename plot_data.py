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

def write_tensor_to_file(data, labels, output_file):
    '''
      Writing a tensor into a file.
    '''
    npdata = data.numpy()
    nplabels = labels.numpy()
    x, y = npdata[:, 0], npdata[:, 1]
    with open(output_file, 'a') as output_data:
        for i in range(len(x)):
            output_data.write(str(int(nplabels[i])) + ',')
            output_data.write(str(x[i]) + ',')
            output_data.write(str(y[i]) + '\n')

    output_data.close()

def paint_from_file(file_name, title):
    '''
    Painting a picture from a csv file.
    '''
    fig, ax = plt.subplots()
    xc, yc = [], [] # points in the unit circle
    xr, yr = [], [] # points not in the unit circle
    with open(file_name, 'r') as input_data:
        points = [line.strip() for line  in input_data.readlines()]
        for point in points:
            label, x, y = point.split(',')
            if label == '1':
                xc.append(float(x))
                yc.append(float(y))
            else:
                xr.append(float(x))
                yr.append(float(y))

    plt.scatter(xc, yc, color='red', label='Positive Point')
    plt.scatter(xr, yr, color='blue', label='Negative Point')
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    ax.set(xlabel='$x_1$', ylabel='$x_2$', title=title)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show(block=True)
    input_data.close()

def paint_from_tensors(data, labels, output_file, title):
    'paint the testing results with data and label.'
    write_tensor_to_file(data, labels, output_file)
    paint_from_file(output_file, title)
    
def main():
    torch.manual_seed(1)
    data = torch.rand(10,2)
    labels = torch.ones(10)
#    print(data, labels)
  #  paint_tensor(data)    
#    write_tensor_to_file(data, labels, './data/output.csv')
    paint_from_file('./data/predicted_result.csv', title='Predicted Results')
 #   paint_from_file('./data/dataset.csv', title='Train Data')
#    paint_from_tensors(data, labels, './data/output.csv')
if __name__ == '__main__':
    main()
