import matplotlib.pyplot as plt
import numpy as np
import itertools

def cartesian_product(x,y):
    '''
      x: array of x corordinates (x1, x2, x3,...).
      y:  array of y corordinates (y1, y2,y3, ...).
      Return the array of points ((x1,y1), (x2,y2), ...)).     
    '''
    points = np.array(list(itertools.product(*[x,y])))

    return points

def write_to_file(x, y):
    with open('data/zcurve_points.csv','w') as output_file:
        for i in range(len(x)):
            output_file.write(str(x[i])+ ',')
            output_file.write(str(y[i])+ '\n')

    output_file.close()

k = 6 # number of points every axis
x = np.linspace(-1.5+3/2**(k+1), 1.5-3/2**(k+1), 2**k)
y = np.linspace(-1.5+3/2**(k+1), 1.5-3/2**(k+1), 2**k)


#x_left, y_left = plot_zcurve(x[:2],y[:2])
#x_left, y_right = plot_zcurve(x[:2],y[2:])
#x_right, y_left = plot_zcurve(x[2:],y[:2])
#x_right, y_right = plot_zcurve(x[2:],y[2:])

#x = np.concatenate((x_left, x_left, x_right, x_right), axis=None)
#y = np.concatenate((y_left, y_right, y_left, y_right), axis=None)


points = cartesian_product(x,y)
xx = [item[0] for item in points]
yy = [item[1] for item in points]
write_to_file(xx,yy)
# the left-top point of 8x8=64 points
#x1 = np.array((-1.4375,-0.8125))
#y1 = np.array((0.8125,1.4375))
#px1, py1 = plot_zcurve(x1, y1)

# Lines on top of scatter
fig, ax = plt.subplots()
#plt.subplot(211)

#plt.plot(x, y, 'r', lw=3)
plt.scatter(xx, yy, s=20)

plt.plot([-1.5, 1.5], [0, 0], '--', lw=1)
plt.plot([0,0],[-1.5,1.5], '--',  lw=1)
plt.axis([-1.5, 1.5, -1.5, 1.5])
ax.set(xlabel='$x$', ylabel='$y$', title='Data Z-curve Distribution')
#plt.legend(loc='upper right')

plt.show()
