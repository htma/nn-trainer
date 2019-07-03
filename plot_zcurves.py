import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_zcurve(x,y):
    '''x: tuple of x corordinate (x1, x2).
       y: tuple of y corordinate (y1, y2).
    '''
    x = np.concatenate((x,x), axis=None)
    y = list(zip(y[::-1], y[::-1]))
    y = [item for tup in y for item in tup]
    return x, y
    
#x = np.array((-0.75,0.75,-0.75,0.75))
#y = np.array((0.75,0.75,-0.75,-0.75))
k = 2 # number of points every axis
x = np.linspace(-1.5+3/2**(k+1), 1.5-3/2**(k+1), k**2)
y = np.linspace(-1.5+3/2**(k+1), 1.5-3/2**(k+1), k**2)
x_left, y_left = plot_zcurve(x[:2],y[:2])
x_left, y_right = plot_zcurve(x[:2],y[2:])
x_right, y_left = plot_zcurve(x[2:],y[:2])
x_right, y_right = plot_zcurve(x[2:],y[2:])


# Lines on top of scatter
fig, ax = plt.subplots()
#plt.subplot(211)
plt.plot(x_left, y_left, 'r', lw=3)
plt.scatter(x_left, y_left, s=120)

plt.plot([x_left[-1], x_right[0]], [y_right[-1], y_right[0]], 'black', lw=3)
plt.plot(x_left, y_right, 'r', lw=3)
plt.scatter(x_left, y_right, s=120)
plt.plot([x_right[-1], x_left[0]], [y_right[-1], y_left[0]], 'black', lw=3)
plt.plot(x_right, y_left, 'r', lw=3)
plt.scatter(x_right, y_left, s=120)
plt.plot([x_left[-1], x_right[0]], [y_left[-1], y_left[0]], 'black', lw=3)
plt.plot(x_right, y_right, 'r', lw=3)
plt.scatter(x_right, y_right, s=120)

plt.plot([-1.5, 1.5], [0, 0], '--', lw=1)
plt.plot([0,0],[-1.5,1.5], '--',  lw=1)
plt.axis([-1.5, 1.5, -1.5, 1.5])
ax.set(xlabel='$x$', ylabel='$y$', title='Data Z-curve Distribution')
#plt.grid()
#plt.legend(loc='upper right')

#plt.title('Lines on top of dots')

# # Scatter plot on top of lines
# plt.subplot(212)
# plt.plot(x, y, 'r', zorder=1, lw=3)
# plt.scatter(x, y, s=120, zorder=2)
# plt.title('Dots on top of lines')


plt.show()
