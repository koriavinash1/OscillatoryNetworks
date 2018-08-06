import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 


for ii in range(0, 180, 5):
	theta = ii*np.pi/180.

	wt1   = np.array([np.cos(theta), -1*np.sin(theta)])
	wt2   = np.array([np.cos(theta), -1*np.sin(theta)])


	a     = 0.5
	b     = 0.1
	gamma = 0.1
	freq  = 0.1
	T     = 500
	dt    = 0.01

	v1, v2 = np.random.randn(), np.random.randn()
	u1, u2 = 0.0, 0.0

	phi = 90* np.pi/180.0

	v1s, v2s = [], []
	
	plt.ion()
	plt.clf()
	for i in range(int(T/dt)):
		v1s.append(v1)
		v2s.append(v2)

		I1 = 0.5 # + 0.5*np.sin(0.1*np.pi*i/T) 
		I2 = 0.5 # + 0.5*np.sin(0.1*np.pi*i/T + phi)
		
		cp1 = 0.1*(wt1[0] * v2 - v1 * wt1[1])
		cp2 = 0.1*(wt2[0] * v1 - v2 * wt2[1])

		v1dot = (v1*(a - v1)*(v1 - 1) - u1 + I1 + cp1)/freq
		u1dot = b*v1 - gamma*u1

		v2dot = (v2*(a - v2)*(v2 - 1) - u2 + I2 + cp2)/(freq*2)
		u2dot = b*v2 - gamma*u2

		v1 = v1 + v1dot*dt
		u1 = u1 + u1dot*dt
		
		v2 = v2 + v2dot*dt
		u2 = u2 + u2dot*dt

	plt.plot(v1s)
	plt.title(str(ii))
	plt.plot(v2s)
	plt.pause(0.5)