import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.01
T = 100

rand = lambda x: np.random.randn(x)
conj = lambda x: np.conjugate(x)

omega = 2.*np.pi/ 25
phi = 90.*np.pi/180.
mu  = 1
eps = 01
dws = []

ita = 1e-1

c1 = 0.01 + 0.01j
cf = 0.1

epochs = 20
train_epochs = 10

plt.ion()
for ep in range(epochs):
	Z1s, Z2s = [], []

	Z1 = 0.01 + 0.1j #rand(1) + rand(1)*1j
	Z2 = 0.05 + 0.5j #rand(1)*1j
	
	for i in range(int(T/dt)):

		F2 = 0.01 #*np.cos((2.*np.pi * 4/float(T/dt))*i)
		F1 = 0.01 # *np.cos((2.*np.pi * 4/float(T/dt))*i + phi)

		if ep > train_epochs: F2 = F1 = 0.01

		Z1s.append(Z1)
		Z2s.append(Z2)

		cp1 = cf * (np.real(c1) *np.real(Z2) - np.imag(c1)*np.imag(Z1))
		cp2 = cf * (np.imag(c1) *np.imag(Z2) - np.real(c1)*np.real(Z1))

		# print (c1, cp1, cp2, Z1, Z2)
		# if i == 100: exit()

		Z1dot = Z1*(mu - np.abs(Z1)**2) + omega*Z1 *1j + eps*F1 + cp1
		Z2dot = Z2*(mu - np.abs(Z2)**2) + omega*Z2 *1j + eps*F2 - cp2

		Z1 = Z1 + Z1dot*dt
		Z2 = Z2 + Z2dot*dt

		# deltaw = ita*(np.exp(1j*phi)*Z1*np.conjugate(Z2) + (1 + 1j)*np.abs(Z1*Z2))
		deltaw = ita*(np.exp(1j*phi)*Z1*np.conjugate(Z2)/(np.abs(Z1*Z2)) - (1 + 1j))
		# deltaw = ita*(Z1*np.conjugate(Z2) - c1) 

		# deltaw = deltaw if ep < train_epochs else (0. + 0.j)
		dws.append(deltaw)
		c1 += deltaw 
		c1 = c1/np.abs(c1)
		# print (deltaw/ita)

	plt.figure('Comp')
	plt.clf()
	plt.plot(np.real(Z1s), np.imag(Z1s))
	plt.plot(np.real(Z1s)[0], np.imag(Z1s)[0], '*g')
	plt.plot(np.real(Z2s), np.imag(Z2s))
	plt.plot(np.real(Z2s)[0], np.imag(Z2s)[0], '*r')

	plt.figure('Dws')
	plt.clf()
	plt.plot(np.abs(dws))

	plt.figure('Real')
	plt.clf()
	plt.plot(np.real(Z1s), 'g')
	plt.plot(np.real(Z2s), 'r')
	plt.title("epoch : {}".format(ep))
	plt.pause(0.5)