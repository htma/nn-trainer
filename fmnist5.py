import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def main():
    w = np.loadtxt('./hidden_neurons.txt', delimiter=',')
    w= w[:,0:-2]
    print(w.shape)
    row_min = np.amin(w, axis=1)
    row_max = np.amax(w, axis=1)
    w0 = (w[0] -row_min[0])/row_max[0] * 255
    weights = [0]*10
    for i in range(10):
        weights[i] = ((w[i]-row_min[i])/row_max[i]* 255).reshape(28,28)
        plt.subplot(4,4,i+1)
        plt.imshow(weights[i], cmap='gray',extent=[0,30,0,30])
        plt.title('weight ' + str(i+1))
        plt.axis('off')
      #  plt.grid()
        
    plt.show()

    

if  __name__ == '__main__':
    main()
