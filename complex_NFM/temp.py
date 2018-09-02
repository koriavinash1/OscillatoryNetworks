import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
from skimage.restoration import unwrap_phase

dt = 0.01
T = 100

rand = lambda x: np.random.randn(x)
conj = lambda x: np.conjugate(x)

omega = 2.*np.pi/ 25
phi = 37.5*np.pi/180.
mu  = 1
eps = 1
dws = []

ita = 1e-5

c1 = 0.01 + 0.01j
cf = 0.01

def perform_exp(phi, plot=True):
	estimated_ph = []
	dt = 0.01
	T = 100

	rand = lambda x: np.random.randn(x)
	conj = lambda x: np.conjugate(x)

	omega = 2.*np.pi/ 25
	mu  = 1
	eps = 1
	dws = []

	ita = 1e-5

	c1 = 0.01 + 0.01j
	cf = 0.01


	epochs = 50
	train_epochs = 30

	dws = []
	angle = []

	for ep in range(epochs):
		Z1s, Z2s = [], []

		Z1 = rand(1) + rand(1)*1j
		Z2 = rand(1) + rand(1)*1j
		
		for i in range(int(T/dt)):

			F2 = np.cos((2.*np.pi * 4/float(T/dt))*i)
			F1 = np.cos((2.*np.pi * 4/float(T/dt))*i + phi)

			if ep > train_epochs: 
				F2 = F1 = 0.01
				cf = 1

			Z1s.append(Z1)
			Z2s.append(Z2)

			cp1 = cf * c1*Z2
			cp2 = cf * np.conjugate(c1)*Z1


			Z1dot = Z1*(mu - np.abs(Z1)**2) + omega*Z1 *1j + eps*F1 + cp1
			Z2dot = Z2*(mu - np.abs(Z2)**2) + omega*Z2 *1j + eps*F2 + cp2

			Z1 = Z1 + Z1dot*dt
			Z2 = Z2 + Z2dot*dt

			# deltaw = ita*(np.exp(1j*phi)*Z1*np.conjugate(Z2) + (1 + 1j)*np.abs(Z1*Z2))
			# deltaw = ita*(np.exp(1j*phi)*Z1*np.conjugate(Z2)/(np.abs(Z1*Z2)) - (1 + 1j))
			# deltaw = ita*np.abs(np.angle(Z1) - np.angle(Z2))
			deltaw = ita*(Z2*np.conjugate(Z1) - c1) 

			deltaw = deltaw if ep <= train_epochs else (0. + 0.j)
			dws.append(deltaw)
			c1 += deltaw 
			c1 = c1/np.abs(c1)
			# print (deltaw/ita)

		ph1 = np.mean(unwrap_phase(np.angle(Z1s[5000:])))*180./np.pi
		ph2 = np.mean(unwrap_phase(np.angle(Z2s[5000:])))*180./np.pi

		# pdb.set_trace()
		if ep > train_epochs:
			phdif = abs(ph1 - ph2)
			print phdif
			phdif = (phdif / (phdif // 360.)) if phdif > 360. else phdif
			phdif = phdif - 360. if phdif > 360. else phdif
			phdif = phdif if phdif <= 180 else 360.0 - phdif
			estimated_ph.append(phdif)


		if plot:
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

	return np.mean(np.array(estimated_ph))

ph_estph = []
for i in tqdm(range(50, 180, 5)):
	phi   = i * np.pi/180.0
	est   = perform_exp(phi, False)
	print ([phi * 180.0 / np.pi, est])
	ph_estph.append([phi * 180.0 / np.pi, est])

ph_estph = np.array(ph_estph)

plt.figure('Plot Final')
plt.stem(ph_estph[:,0], ph_estph[:,1])
plt.plot(ph_estph[:,0], ph_estph[:,0], 'r')
plt.xlabel("given phase difference")
plt.ylabel("lat. phase difference after training")
plt.title("Hopf")
plt.show()