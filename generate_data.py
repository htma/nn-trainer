# generate random points within a range, such as circle or rectangle.
# created on Dec 25, 2018 by mht

import math
import numpy as np
import matplotlib.pyplot as plt 


num_samples = 10  # number of points 

def generate_points(num_samples):
    # generate the num_samples points within a rectangle[-1.5, 1.5, -1.5, 1.5]
    points = -1.5+ np.random.rand(num_samples,2)*3
    px, py = list(points[:,0]), list(points[:,1])

    # divide points into two classes: in the unit circle and not.
    xc, yc = [], [] # points in the circle
    xr, yr = [], [] # points not in the circle
    for i in range(len(px)):
        if math.sqrt(px[i]**2 + py[i]**2) <= 1:
            xc.append(px[i])
            yc.append(py[i])
        else:
            xr.append(px[i])
            yr.append(py[i])

    return xc, yc, xr, yr

def write_to_file(xc, yc, xr, yr):
    with open('data/dataset.csv','w') as output_file:
        for i in range(len(xc)):
            output_file.write(str(1)+',')
            output_file.write(str(xc[i])+ ',')
            output_file.write(str(yc[i])+ '\n')

        for i in range(len(xr)):
            output_file.write(str(0)+',')
            output_file.write(str(xr[i])+',')
            output_file.write(str(yr[i])+'\n')
    output_file.close()

def painting(xc, yc, xr, yr):
    fig, ax = plt.subplots()
    
    # make a simple unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    a, b = 1 * np.cos(theta), 1 * np.sin(theta)
    plt.plot(a, b, linestyle='-', linewidth=2,
             color='black', label='Unit Circle')

    # painting the points
    plt.scatter(xc, yc, color='red', label='Positive Points')
    plt.scatter(xr, yr, color='blue', label='Negative Points')
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    ax.set(xlabel='$x_1$', ylabel='$x_2$', title='Train data')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show(block=True)

def main():
    xc, yc, xr, yr = generate_points(num_samples)
    painting(xc, yc, xr, yr)
    write_to_file(xc,yc,xr,yr)

if __name__ == '__main__':
    main()
