import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.01
T = 100

omega = 2.*np.pi/ 25
phi = np.pi/2
mu  = 1
eps = 01

ita = 1

c1 = 0. + 0.j
cf = 0.001

epochs = 25
train_epochs = 10

plt.ion()
for ep in range(epochs):
	Z1s, Z2s = [], []

	Z1 = 0.1 + 0.2*1j
	Z2 = 0.1 + 0.2*1j

	for i in range(int(T/dt)):

		if ep < train_epochs:
			F2 = np.cos((2.*np.pi * 3.5/float(T/dt)) *i)
			F1 = np.cos((2.*np.pi * 3.5/float(T/dt))*i + phi)
		else :
			F2 = F1 = 0
			cf = 0.01

		Z1s.append(Z1)
		Z2s.append(Z2)

		cp1 = c1 *(Z2 - Z1)
		cp2 = np.conjugate(c1) *(Z1 - Z2)

		Z1dot = Z1*(mu - np.abs(Z1)**2) + omega*Z1 *1j + eps*F1 #+ cf*np.real(cp1)
		Z2dot = Z2*(mu - np.abs(Z2)**2) + omega*Z2 *1j + eps*F2 #+ cf*np.imag(cp2)

		Z1 = Z1 + Z1dot*dt
		Z2 = Z2 + Z2dot*dt

	if ep < train_epochs:
		deltaw = ita*(np.exp(1j*phi) * np.mean(Z1s) * np.mean(np.conjugate(Z2)) + (1 + 1j)*np.mean(np.abs(Z1*Z2)))
		c1 += deltaw 
		print (c1)

	plt.figure('Comp')
	plt.clf()
	plt.plot(np.real(Z1s), np.imag(Z1s))
	plt.plot(np.real(Z1s)[0], np.imag(Z1s)[0], '*g')
	plt.plot(np.real(Z2s), np.imag(Z2s))
	plt.plot(np.real(Z2s)[0], np.imag(Z2s)[0], '*r')

	plt.figure('Real')
	plt.clf()
	plt.plot(np.real(Z1s), 'g')
	plt.plot(np.real(Z2s), 'r')
	plt.title("epoch : {}".format(ep))
	plt.pause(0.05)