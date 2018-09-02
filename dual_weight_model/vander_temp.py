import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.restoration import unwrap_phase

def perform_exp(phi, plot=True):
	estimated_ph = []
	dt = 0.01
	T = 100

	rand = lambda x: np.random.randn(x)
	conj = lambda x: np.conjugate(x)

	omega = 2.*np.pi/ 25
	mu  = 4.0
	eps = 10
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

		Z1 = 01*(rand(1) + rand(1)*1j)
		Z2 = 01*(rand(1) + rand(1)*1j)
		
		for i in range(int(T/dt)):

			F2 = np.cos((2.*np.pi * 4/float(T/dt))*i)
			F1 = np.cos((2.*np.pi * 4/float(T/dt))*i + phi)

			if ep > train_epochs: 
				F2 = F1 = 0.01
				cf = 1

			Z1s.append(Z1)
			Z2s.append(Z2)

			cp1 = cf * c1 * Z2
			cp2 = cf * np.conjugate(c1)*Z1

			Z1dot = Z1 * mu/6. *(3. - np.abs(Z1)**2) + 1./mu * Z1 *1j + conj(Z1) * mu/6. *(9. - np.abs(Z1)**2) + 1./mu * conj(Z1)*1j + eps*F1 + cp1
			Z2dot = Z2 * mu/6. *(3. - np.abs(Z2)**2) + 1./mu * Z2 *1j + conj(Z2) * mu/6. *(9. - np.abs(Z2)**2) + 1./mu * conj(Z2)*1j + eps*F2 + cp2

			Z1 = Z1 + Z1dot*dt
			Z2 = Z2 + Z2dot*dt

			deltaw = ita*(Z2*np.conjugate(Z1) - c1) 

			deltaw = deltaw if ep <= train_epochs else (0. + 0.j)
			dws.append(deltaw)
			c1 += deltaw 
			c1 = c1/np.abs(c1)
			# print (deltaw/ita)

			angle.append(180./np.pi *np.arctan2(np.imag(c1), np.real(c1))) 
			dws.append(deltaw)

		ph1 = np.mean(unwrap_phase(np.angle(Z1s[5000:])))*180./np.pi
		ph2 = np.mean(unwrap_phase(np.angle(Z2s[5000:])))*180./np.pi

		if ep > train_epochs:
			phdif = abs(ph1 - ph2)
			phdif = phdif / (phdif // 360.) if phdif > 360. else phdif
			phdif = phdif - 360. if phdif > 360. else phdif
			print phdif
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
for i in tqdm(range(0, 180, 5)):
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
plt.title("Vanderpole")
plt.show()