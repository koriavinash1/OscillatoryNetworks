import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dt = 0.001
T = 10

omega = 20
phi = 90.* np.pi/180.0
train_epoch = 30

ita =1e-3


np.random.seed(2018)
c1 = np.random.randn()+ np.random.randn()*1j

def weight_update(Z1, Z2, w, phi = phi):
	p  = np.exp(-1j*phi)
	dw = ita *(np.abs(Z1*Z2) + 0.5*(Z1*np.conjugate(Z2)*p + np.conjugate(p)*Z2*np.conjugate(Z1)))# + np.abs(Z1) * np.abs(Z2) *(1 + 1j))
	dw = ita *(Z1*np.conjugate(Z2) - w)
	w  = w + dw
	return w, dw

dws = []	



for ep in range(50):
	Z1s, Z2s = [], []
	plt.ion()
	
	np.random.seed(2018)
	Z1, Z2 = (0.1*np.random.randn() + np.random.randn()*0.1j), (0.1*np.random.randn() + np.random.randn()*0.1j)
	Z1 = Z2

	for i in range(int(T/dt)):
		
		Z1star = np.conjugate(Z1)
		Z2star = np.conjugate(Z2)

		if ep < train_epoch:
			F1 = np.cos(omega*i)
			F2 = np.cos(omega*i + phi)
		else:
			F1 = 1. #np.cos(omega*i)
			F2 = 1.

		cp1 = c1 *(Z2 - Z1)
		cp2 = np.conjugate(c1) *(Z1 - Z2)

		Z1s.append(Z1)
		Z2s.append(Z2)

		Z1dot = Z1*(1 - np.abs(Z1)**2) + omega*Z1 *1j + 10*F1 # + 0.01*cp1
		Z2dot = Z2*(1 - np.abs(Z2)**2) + omega*Z2 *1j + 10*F2 # + 0.01*cp2

		Z1 = Z1 + Z1dot*dt
		Z2 = Z2 + Z2dot*dt
		
		if ep < train_epoch:
			c1, dw = weight_update(Z1, Z2, c1) 
			c1     = c1/np.abs(c1)

		dws.append(dw)

	# plt.subplot(2, 1, 1)
	plt.figure('Comp')
	plt.clf()
	plt.plot(np.real(Z1s), np.imag(Z1s))
	plt.plot(np.real(Z1s)[0], np.imag(Z1s)[0], '*g')
	plt.plot(np.real(Z2s), np.imag(Z2s))
	plt.plot(np.real(Z2s)[0], np.imag(Z2s)[0], '*r')
	plt.title("Epoch : {}".format(ep))

	plt.figure('Real')
	plt.clf()
	plt.plot(np.real(Z1s))
	plt.plot(np.real(Z2s))

	plt.figure('dw')
	plt.clf()
	plt.plot(np.real(dws))
	# plt.plot(np.imag(dws))

	plt.pause(0.5)
	# plt.show()
	# # plt.subplot(2, 1, 1)
	# plt.plot(np.imag(Z1s))
	# # plt.subplot(2, 1, 2)
	# plt.plot(np.imag(Z2s))
	# plt.show()

	# # plt.subplot(2, 1, 1)
	# plt.plot(np.real(Z1s), np.imag(Z1s))
	# plt.plot(np.real(Z1s[0]), np.imag(Z1s[0]), '*r')
	# # plt.subplot(2, 1, 2)
	# plt.plot(np.real(Z2s), np.imag(Z2s))
	# plt.plot(np.real(Z2s[0]), np.imag(Z2s[0]), '*r')
	# plt.show()