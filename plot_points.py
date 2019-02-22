import math
import numpy as np
import matplotlib.pyplot as plt

# number of points
num_samples = 500

def painting(xc, yc, xr, yr):
   fig, ax = plt.subplots()

   # make a simple unit circle
   theta = np.linspace(0, 2*np.pi, num_samples)
   a, b = 1*np.cos(theta), 1*np.sin(theta)
   plt.plot(a, b, linestyle='-', linewidth=2, color='black', label='Circle')

   # painting the points
   plt.scatter(xr, yr, color='blue')
   plt.scatter(xc, yc, color='red')
   plt.axis([-1.5, 1.5, -1.5, 1.5])
   ax.set(xlabel='$x_1$', ylabel='$x_2$', title='Train data')
   plt.show()

def generate_points(num_samples):
   # generate the num_samples points within a rectangle [-1,5,1.5, -1.5, 1.5]
   point = -1.5+np.random.rand(num_samples, 2)*3
   px, py = list(point[:,0]), list(point[:,1])

   # divide points into two classes: in the circle and not.
   xc, yc = [], [] # points in the unit circle
   xr, yr = [], [] # points not in the unit  circle
   for i in range(len(px)):
      if math.sqrt(px[i]**2+py[i]**2) <= 1:
         xc.append(px[i])
         yc.append(py[i])
      else:
         xr.append(px[i])
         yr.append(py[i])

   return xc, yc, xr, yr

def main():
   xc, yc, xr, yr = generate_points(num_samples)
   painting(xc, yc, xr, yr)

if __name__ == '__main__':
   main()
