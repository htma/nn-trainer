import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

import torch

def produce_data():
    xs, ys = [], []
    for i in range(1, 11):
        xs.append((i-6)*0.3)
    for j in range(1, 11):
        ys.append((j-6)*0.3)

    data = [(i, j) for i in xs for j in ys]

    return data
    

def neuron_active(t):
    ct = t.clone()
    ct[t < 0] = 0    
    ct[t >= 0] = 1

    return ct.data.numpy()[0].astype(int)

def build_RGB(state):
    '''return a triple from a list of ints states, which represent the valuses of RGB'''
    R, G, B = state[:8], state[8:16], state[16:]
    R = int(''.join(list(map(str, R))), 2)
    G = int(''.join(list(map(str, G))), 2)
    B = int(''.join(list(map(str, B))), 2)
    RGB = '#%02x%02x%02x' % (R, G, B)
    return RGB

def paint_states(states):
    rgb = build_RGB(states)
    fig, ax = plt.subplots()
    rect = plt.Rectangle((-1.2, -1.5), 0.3, 0.3, fc=rgb)
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    ax.add_patch(rect)
    ax.set(xlabel='$x_1$', ylabel='$x_2$',
           title='Train data')
    ax.xaxis.set_ticks([-1.5, -1.2, -0.9, -0.6,-0.3, 0, 0.3,0.6,0.9,1.2,1.5])
    ax.yaxis.set_ticks([-1.5, -1.2, -0.9, -0.6,-0.3, 0, 0.3,0.6,0.9,1.2,1.5])
    ax.grid(True)
    plt.show()

    
if __name__ == '__main__':
    t = torch.tensor([[-0.5731625, -0.26016295, -1.2452601, 0.23687911]])
  #  t = torch.randn(4)
#    print(t)
 #   print(hid_filter(t))

#    states = [1]*22
#    paint_states(states)


 #   print(produce_data())




    
    
