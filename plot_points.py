import math
import numpy as np
import matplotlib.pyplot as plt

#x1 = [1, 1.15, 1.23, 0.92, 1.31, 1.18, 1.27, 1.07]
#x2 = [3, 3.2]

#y1 = [1.17, 0.95, 1.04, 1.32, 1, 1.22, 1.28, 0.99]
#y2 = [1, 1.2]

# number of points
num_samples = 500
# make a simple unit circle
theta = np.linspace(0, 2*np.pi, num_samples)
a, b = 1*np.cos(theta), 1*np.sin(theta)
plt.plot(a, b, linestyle='-', linewidth=2, color='green', label='Circle')

# generate the num_samples points within a circle
#r = np.random.rand((num_samples))
#xc, yc = r*np.cos(theta), r*np.sin(theta)
#plt.scatter(xc, yc, color='red')

# generate the num_samples points within a rectangle [-1,5,1.5] 
point = -1.5+np.random.rand(num_samples, 2)*3
x, y = list(point[:,0]), list(point[:,1])
xc, yc = [], [] # point in the circle
xr, yr = [], [] # points not in the circle

for i in range(len(x)):
   if math.sqrt(x[i]**2+y[i]**2) <= 1:
        xc.append(x[i])
        yc.append(y[i])
   else:
       xr.append(x[i])
       yr.append(y[i])                  

plt.scatter(xr, yr, color='blue')
plt.scatter(xc, yc, color='red')
#plt.scatter(x1, y1, color='red')
#plt.scatter(x2, y2, color='blue')
#circle=plt.Circle((2.0, 2.0), 1, color='r', fill=False)
#fig = plt.gcf()
#fig.gca().add_artist(circle)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.show()
