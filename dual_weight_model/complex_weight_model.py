import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

a     = 0.5
alpha = 0.1
beta  = 0.1
gamma = 0.05

C1 = ((a+1)+gamma*(beta - alpha)*1j)
C2 = ((1-a)+gamma*(beta + alpha)*1j)

ita   = 1e-4
cf    = 0.05
cfu   = 0.05


T     = 50
dt    = 0.01
phi   = 30 * np.pi/180.0

epochs = 1000
train_epochs = 1000

dws = []
angle = []



Z1, Z2 = 0.05 + 0.01j, 0.05 + 0.01j

Z1s, Z2s = [], []

for i in range(int(T/dt)):
	I1 = 0.005#*np.cos((2.*np.pi * 6/float(T/dt))*i)
	I2 = 0.005#*np.cos((2.*np.pi * 6/float(T/dt))*i + phi)

	Z1s.append(Z1)
	Z2s.append(Z2)

	Z1star = np.conjugate(Z1)
	Z2star = np.conjugate(Z2)


	Z1dot = (2*(a+1)*np.abs(Z1)**2 - (Z1 + Z1star)*np.abs(Z1)**2 - C1*Z1 + C2*Z1star + I1)/(2*gamma)
	Z2dot = (2*(a+1)*np.abs(Z2)**2 - (Z2 + Z2star)*np.abs(Z2)**2 - C1*Z2 + C2*Z2star + I2)/(2*gamma)
	
	Z1 = Z1 + Z1dot*dt
	Z2 = Z2 + Z2dot*dt


# plt.subplot(2, 1, 1)
plt.plot(np.real(Z1s))
# plt.subplot(2, 1, 2)
plt.plot(np.real(Z2s))
plt.show()

# plt.subplot(2, 1, 1)
plt.plot(np.imag(Z1s))
# plt.subplot(2, 1, 2)
plt.plot(np.imag(Z2s))
plt.show()

# plt.subplot(2, 1, 1)
plt.plot(np.real(Z1s), np.imag(Z1s))
plt.plot(np.real(Z1s[0]), np.imag(Z1s[0]), '*r')
# plt.subplot(2, 1, 2)
plt.plot(np.real(Z2s), np.imag(Z2s))
plt.plot(np.real(Z2s[0]), np.imag(Z2s[0]), '*r')
plt.show()