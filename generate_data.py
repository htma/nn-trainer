# generate random points within a range, such as circle or rectangle.
# created on Dec 25, 2018 by mht
# revised on Jan 2, 2019

import numpy as np 
import matplotlib.pyplot as plt 


num_samples = 100

# make a simple unit circle 
theta = np.linspace(0, 2*np.pi, num_samples)
a, b = 1 * np.cos(theta), 1 * np.sin(theta)

# generate the points within a circle
r = np.random.rand((num_samples))
xc, yc = r * np.cos(theta), r * np.sin(theta)


# generate the points within a rectangle
xr = -1.5+ np.random.rand(1,1000)*3
yr = -1.5+ np.random.rand(1,1000)*3

# plots
plt.figure(figsize=(7,6))
plt.plot(a, b, linestyle='-', linewidth=2, label='Circle')
plt.plot(xr, yr, marker='o', linestyle='')
#plt.plot(xc, yc, marker='o', linestyle='')
plt.ylim([-1.5,1.5])
plt.xlim([-1.5,1.5])
plt.grid()
#plt.legend(loc='upper right')
plt.show(block=True)

