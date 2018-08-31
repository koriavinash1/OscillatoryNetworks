import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from scipy.signal import hilbert


np.random.seed(218)
wt   = 0.005*np.random.randn(2)

a     = 0.5
b     = 0.1
gamma = 0.1
freq  = 0.1


ita   = 1e-3
cf    = 0.05
cfu    = 0.005


T     = 50
dt    = 0.01
phi   = 90 * np.pi/180.0

epochs = 1000
train_epochs = 30



def update_weights(Z1, Z2, W):
	deltaw = ita*(Z1*np.conjugate(Z2) - W)
	W = W + deltaw
	return np.array([np.real(W),np.imag(W)]), np.array([np.real(deltaw), np.imag(deltaw)])

def get_roots(I):
	x = [np.abs(aa) for aa in np.roots([-1, (a+1), -(a+b/gamma), I]) if np.conjugate(aa) == aa]
	return x[0], b/gamma * x[0]


dws = []
angle = []

for ep in range(epochs):
	v1s, v2s = [], []
	plt.ion()

	v1 =  v2 = 0.05*np.random.randn() #, np.random.randn()
	u1 =  u2 = 0.05*np.random.randn() #, np.random.randn()


	for i in range(int(T/dt)):
		v1s.append(v1)
		v2s.append(v2)

		I1 = 0.3 + 0.2*np.cos((2.*np.pi * 6/float(T/dt))*i)
		I2 = 0.3 + 0.2*np.cos((2.*np.pi * 6/float(T/dt))*i + phi)

		v10, u10 = get_roots(I1)
		v20, u20 = get_roots(I2)

		if ep > train_epochs:
			I2 = I1 = 0.3 #*np.cos((2.*np.pi * 6/float(T/dt))*i)
			cf = 0.05


		v1, v2 = v1 - v10, v2 - v20
		u1, u2 = u1 - u10, u2 - u20


		cpv1 = wt[0]*v2 - wt[1]*u2
		cpu1 = wt[1]*v2 + wt[0]*u2

		cpv2 = wt[0]*v1 + wt[1]*u1
		cpu2 = -wt[1]*v1 + wt[0]*u1


		v1dot = (v1*(a - v1)*(v1 - 1) - u1 + I1 + cf*cpv1)/freq
		u1dot = b*v1 - gamma*u1 + cfu*cpu1

		v2dot = (v2*(a - v2)*(v2 - 1) - u2 + I2 + cf*cpv2)/freq
		u2dot = b*v2 - gamma*u2 + cfu*cpu2

		v1 = v1 + v1dot*dt
		u1 = u1 + u1dot*dt
		
		v2 = v2 + v2dot*dt
		u2 = u2 + u2dot*dt

		if ep < train_epochs:
			wt, dw = update_weights(v1 + u1*1j, v2 + u2*1j, wt[0] + 1j*wt[1])
			wt     = wt/np.max(np.abs(wt))

		angle.append(180./np.pi *np.arctan2(wt[1], wt[0])) 
		dws.append(dw)

	ph1 = np.mean(np.unwrap(np.angle(hilbert(v1s[3000:]))))*180./np.pi
	ph2 = np.mean(np.unwrap(np.angle(hilbert(v2s[3000:]))))*180./np.pi


	plt.figure('v')
	plt.clf()
	plt.plot(v1s)
	plt.plot(np.array(v2s))
	plt.title("Epoch: {}".format(ep) + " Estimated phase difference: {}".format(abs(ph1 - ph2)))
	
	plt.figure('dw')
	plt.clf()
	plt.plot(np.array(dws)[:,0])

	plt.figure('dw2')
	plt.clf()
	plt.plot(np.array(dws)[:,1])
	
	plt.figure('ang')
	plt.clf()
	plt.plot(np.array(angle))
	plt.pause(0.5)
