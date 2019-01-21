# generate random points within a range, such as circle or rectangle.
# created on Dec 25, 2018 by mht

import math
import numpy as np
import matplotlib.pyplot as plt 


num_samples = 10 # number of points 

# make a simple unit circle 
theta = np.linspace(0, 2*np.pi, 100)
a, b = 1 * np.cos(theta), 1 * np.sin(theta)

# generate the points within a circle
#r = np.random.rand((num_samples))
#xc, yc = r * np.cos(theta), r * np.sin(theta)


# generate the num_samples points within a rectangle
x = -1.5+ np.random.rand(num_samples,2)*3
xr, yr = x[:,0], x[:,1]
points = list(zip(xr, yr))


#        print('The label of the point {} is {:}'.format((point[0], point[1]), 1))
 

# write the data to a file
with open('data/dataset.csv','w') as output_file:
    for point in points:
        if math.sqrt(point[0]**2+point[1]**2) <= 1:
            output_file.write(str(1)+",")
            output_file.write(str(point[0])+ ",")
            output_file.write(str(point[1])+ "\n")
           
        else:
            output_file.write(str(0)+",")
            output_file.write(str(point[0])+ ",")
            output_file.write(str(point[1])+ "\n")
 

output_file.close()
        
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

