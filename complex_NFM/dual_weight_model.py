import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

theta = 45.*np.pi/180.

wt1   = 1*np.array([np.sin(theta), np.cos(theta)])
wt2   = 1*np.array([np.sin(theta), np.cos(theta)])


a     =  0.139
gamma = 2.54
eps   = 0.008
freq  = 0.08

T     = 500
dt    = 0.01


v1, v2 = 0.05, 0.05
u1, u2 = 0.05, 0.05

phi = 90* np.pi/180.0

v1s, v2s = [], []

for i in range(int(T/dt)):
	v1s.append(v1)
	v2s.append(v2)

	I1 = 0.05 + 0.5*np.sin(0.1*np.pi*i/T) 
	I2 = 0.05 + 0.5*np.sin(0.1*np.pi*i/T + phi)
	
	cp1 = wt1[0] * (v2 - v1) + wt1[1] * (v2 - u1)  
	cp2 = wt2[0] * (v1 - v2) + wt2[1] * (v1 - u2)

	v1dot = (v1*(a - v1)*(v1 - 1) - u1 + I1 - cp1)/freq
	u1dot = eps*(v1 - gamma*u1)

	v2dot = (v2*(a - v2)*(v2 - 1) - u2 + I2 - cp2)/freq
	u2dot = eps*(v2 - gamma*u2)

	v1 = v1 + v1dot*dt
	u1 = u1 + u1dot*dt
	
	v2 = v2 + v2dot*dt
	u2 = u2 + u2dot*dt

plt.plot(v1s)
plt.plot(v2s)
plt.show()