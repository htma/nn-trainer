import matplotlib.pyplot as plt
import numpy as np
import itertools

#x = np.array((-0.75,0.75,-0.75,0.75))
#y = np.array((0.75,0.75,-0.75,-0.75))
k = 16 # number of points every axis
x = np.linspace(-1.5+3/2**k, 1.5-3/2**k, k)
y = np.linspace(-1.5+3/2**k, 1.5-3/2**k, k)
#x = np.concatenate((x,x), axis=None)
#y = list(zip(y[::-1],y[::-1]))
#y = [item for t in y for item in t]
xy = list(zip(*itertools.product(x,y)))
x, y = xy[0], xy[1]
print(x)
print(y)

# Lines on top of scatter
fig, ax = plt.subplots()
#plt.subplot(211)
plt.plot(x, y, 'r', lw=3)
plt.scatter(x, y, s=120)
plt.axis([-1.5, 1.5, -1.5, 1.5])
ax.set(xlabel='$x_1$', ylabel='$x_2$', title='Z curve')
plt.grid()
plt.legend(loc='upper right')

plt.title('Lines on top of dots')

# # Scatter plot on top of lines
# plt.subplot(212)
# plt.plot(x, y, 'r', zorder=1, lw=3)
# plt.scatter(x, y, s=120, zorder=2)
# plt.title('Dots on top of lines')

# A new figure, with individually ordered items
# x = np.linspace(0, 2*np.pi, 100)
# plt.figure()
# plt.plot(x, np.sin(x), linewidth=10, color='black', label='zorder=10', zorder=10)  # on top
# plt.plot(x, np.cos(1.3*x), linewidth=10, color='red', label='zorder=1', zorder=1)  # bottom
# plt.plot(x, np.sin(2.1*x), linewidth=10, color='green', label='zorder=3', zorder=3)
# plt.axhline(0, linewidth=10, color='blue', label='zorder=2', zorder=2)
# plt.title('Custom order of elements')
# l = plt.legend()
# l.set_zorder(20)  # put the legend on top
plt.show()
