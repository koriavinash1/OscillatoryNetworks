import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from scipy.signal import hilbert

theta = 90*np.pi/180.

np.random.seed(218)
wt   = np.random.randn(2)

def update_weights(v, w, phi = 90.):

	phi = phi * np.pi/180.0
	assert len(v) == len(w)

	rv = np.abs(hilbert(v))	
	av = np.angle(hilbert(v))

	deltaw = np.prod(rv)*(1. + np.array([np.cos(av[0] - av[1] + phi), np.sin(av[1] - av[0] + phi)]))*ita
	# print deltaw
	return w + deltaw

a     = 0.5
b     = 0.1
gamma = 0.1
freq  = 0.1
ita   = 1
T     = 50
dt    = 0.01

v1, v2 = np.random.randn(), np.random.randn()
u1, u2 = 0.0, 0.0

phi = 90 * np.pi/180.0



for ii in range(25):
	v1s, v2s = [], []
	plt.ion()
	plt.clf()
	for i in range(int(T/dt)):
		v1s.append(v1)
		v2s.append(v2)

		I1 = 0.5 # + 0.5*np.sin(0.1*np.pi*i/T) 
		I2 = 0.5 # + 0.5*np.sin(0.1*np.pi*i/T + phi)
		
		cp1 = 0.1*(wt[0] * v2 - v1 * wt[1])
		cp2 = -0.1*(wt[1] * v1 - v2 * wt[0])

		v1dot = (v1*(a - v1)*(v1 - 1) - u1 + I1 + cp1)/freq
		u1dot = b*v1 - gamma*u1

		v2dot = (v2*(a - v2)*(v2 - 1) - u2 + I2 + cp2)/(freq)
		u2dot = b*v2 - gamma*u2

		v1 = v1 + v1dot*dt
		u1 = u1 + u1dot*dt
		
		v2 = v2 + v2dot*dt
		u2 = u2 + u2dot*dt

		wt = update_weights(np.array([v1, v2]), wt)
		wt = wt/np.sum(np.abs(wt))

	ph1 = np.mean(np.unwrap(np.angle(hilbert(v1s))))*180./np.pi
	ph2 = np.mean(np.unwrap(np.angle(hilbert(v2s))))*180./np.pi

	plt.plot(v1s)
	plt.plot(v2s)
	plt.title("Epoch: {}".format(ii) + " Estimated phase difference: {}".format(abs(ph1 - ph2)))
	plt.pause(0.5)
