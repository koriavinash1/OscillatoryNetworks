import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
from skimage.restoration import unwrap_phase

dt = 0.01
T  = 10
omega = 2.*np.pi/ 25.
phi = 37.5*np.pi/180.
mu  = 1.
eps = 1.

from sympy.solvers import solve
from sympy import Symbol
y = Symbol('y')

x = list(np.array(range(0, int(T/dt)))*dt)

# y1 = np.array([solve(mu*xi - y - (xi + 3./2. * y)*(xi**2 + y**2), y) for xi in x], dtype='complex64')
# y2 = np.array([solve(xi + mu*y + (3./2. * xi - y)*(xi**2 + y**2), y) for xi in x], dtype='complex64')

y1 = [solve(xi* (mu - (xi**2 + y**2)) - omega*y + eps*(np.cos((2.*np.pi * 4/float(T/dt))*int(xi/dt))), y) for xi in x]
y2 = [solve(y * (mu - (xi**2 + y**2)) + omega*xi, y) for xi in x]

def roots(x, y):
	plot1 = {'x':[], 'y' :[]}
	plot2 = {'x':[], 'y' :[]}
	plot3 = {'x':[], 'y' :[]}
	for xx, yy in zip(x, y):
		yy = np.array(yy, dtype = 'complex64')
		try:
			if yy[0] == np.conjugate(yy[0]): 
				plot1['x'].append(xx)
				plot1['y'].append(np.float(yy[0]))
		except: pass

		try:
			if yy[1] == np.conjugate(yy[1]): 
				plot2['x'].append(xx)
				plot2['y'].append(np.float(yy[1]))
		except: pass

		try:
			if yy[2] == np.conjugate(yy[2]): 
				plot3['x'].append(xx)
				plot3['y'].append(np.float(yy[2]))
		except: pass

	return plot1, plot2, plot3

p11, p12, p13 = roots(x, y1)
p21, p22, p23 = roots(x, y2)

plt.figure('plot1')
# plt.title(str(len(p11['x'])) +"=========" + str(len(p12['y'])))
plt.plot(p11['x'], p11['y'], 'r')
plt.plot(p21['x'], p21['y'], 'b')


# plt.figure('plot2')
# plt.title(str(len(p12['x'])) +"=========" + str(len(p12['y'])))
plt.plot(p12['x'], p12['y'], 'r')
plt.plot(p22['x'], p22['y'], 'b')

# plt.figure('plot3')
# plt.title(str(len(p13['x'])))
plt.plot(p13['x'], p13['y'], 'r')
plt.plot(p23['x'], p23['y'], 'b')

plt.xlabel('x')
plt.ylabel('y')
plt.legend(['red: xi*(1-(xi**2+y**2))-omega*y+F = 0', 'blue: y*(mu-(xi**2+y**2))+omega*xi = 0 '])
plt.show()