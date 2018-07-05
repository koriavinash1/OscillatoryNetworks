from Oscillators import *
from configure import Config
from tqdm import tqdm
from GaussianStatistics import *
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import pandas as pd

# load Orientation Bars
OrientationBars = np.load('./Orientation_bars.npy')
print ("Orientation Bars Loaded: Shape {}".format(OrientationBars.shape))

Gstat = GaussianStatistics()
config= Config()

# for 3D plot...
X = np.arange(0, config.N)
Y = np.arange(0, config.N)
X, Y = np.meshgrid(X, Y)

# ----------------------------------------------

# dynamic to dynamic
class NFM(object):
	def __init__(self, ci=0.3):
		self.ci    = ci
		self.ita   = 1
		self.d2dnfm = FreqAdaptiveCoupledNFM_D2D(size=(config.N, config.N),
						exe_rad = config.eRad,
						inhb_rad = config.N, # for global inhabition
						exe_ampli = config.eA,
						inhb_ampli = config.iA)# self.phase is (T/dt, N, N) shape

	def display(self, Z, i, fig,  _type = '3d'):
		plt.clf()
		plt.ion()
		if _type == '3d':
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(X, Y, Z/np.max(Z) , cmap=cm.coolwarm,
			               linewidth=0, antialiased=False)
			ax.set_title(str(i))
			# Customize the z axis.
			ax.set_zlim(-1.01, 1.01)
			ax.zaxis.set_major_locator(LinearLocator(10))
			ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
			fig.colorbar(surf, aspect=10)
		else:
			plt.imshow(Z)

		plt.draw()
		plt.xlabel(str(i))
		plt.pause(0.005)
		pass

	# Statis input to dynamic output
	def staticToDynamic(self, image):
		"Statis to Dynamic needs one to one connections,\
		Between static image and Oscillator"

		s2dnfm = CoupledNFM(size=(config.N, config.N),
						exe_rad = config.eRad,
						inhb_rad = config.N,
						exe_ampli = config.eA,
						inhb_ampli = config.iA,
						aff = image)
		s2d_nsheets = []
		for i in range(0, int(config.T/config.dt)):
			s2dnfm.lateralDynamics(verbose = False, ci = self.ci)
			s2d_nsheets.append(s2dnfm.Z)
			self.display(s2dnfm.Z, i)
		return np.array(s2d_nsheets)

	# calculate phase vector for each neuron
	def instantPhase(self, tensor):
		""
		# tensor shape -> (T/dt, N, N)
		# tensor numpy array
		assert len(tensor.shape) == 3
		phases = [np.angle(hilbert(x)) for x in tensor.reshape(-1, np.prod(tensor.shape[1:]))]
		radius = [np.abs(hilbert(x)) for x in tensor.reshape(-1, np.prod(tensor.shape[1:]))]
		return np.array(phases).reshape(tensor.shape), np.array(radius).reshape(tensor.shape)

	# weight update rule 
	# https://link.springer.com/content/pdf/10.1007%2Fs11571-018-9489-x.pdf
	def deltaweights_polar(self, rI, rNFM, argI, argNFM):
		# rI, rNFM 3D spatio-temporal signal
		# returns (N, N, N, N) 4D weight delta weights
		deltaw = np.empty((config.N, config.N, config.N, config.N)) 
		for ii in range(config.N):
			for jj in range(config.N):
				NFM = rNFM[:, ii, jj].reshape(-1, 1, 1)
				arg = argNFM[:,ii,jj].reshape(-1, 1, 1)
				deltaw[ii, jj, :, :] = self.ita*np.mean(rI*NFM* (1. + np.cos(argI - arg)), axis=0)
		mean = np.mean(deltaw, axis = 0)
		deltaw[deltaw < 0.95* np.max(mean)] = 0
		return deltaw

	def deltaweights_xy(self, s2d, d2d):
		# rI, rNFM 3D spatio-temporal signal
		# returns (N, N, N, N) 4D weight delta weights
		deltaw = np.empty((config.N, config.N, config.N, config.N)) 
		for ii in range(config.N):
			for jj in range(config.N):
				t = d2d[:, ii, jj].reshape(-1, 1, 1)
				deltaw[ii, jj, :, :] = self.ita*np.mean(s2d*t, axis=0)
		return deltaw

	# Dynamic update without SOM layer
	def one_fit_D2D(self, image):
		"fits one image at a time Train NFM end2end"
		binary_image = image.copy()
		binary_image[binary_image > 0.5] = 1
		binary_image[binary_image < 0.5] = -1

		# plt.plot(np.sin(np.arange(0,int(config.T/config.dt))*2*3.14*4 + 0.05))
		# plt.show()
		s2d_nsheets = [c*np.sin(np.arange(0,int(config.T/config.dt))*2*3.14*4 + 0.05) for r in binary_image for c in r ]
		s2d_nsheets = np.array(s2d_nsheets)
		s2d_nsheets = s2d_nsheets.T.reshape(-1, config.N, config.N)

		d2d_nsheets = []
		for i in range(0, int(config.T/config.dt)):
			self.d2dnfm.lateralDynamics(aff = s2d_nsheets[i,:,:], verbose = False, ci = 0.1)
			d2d_nsheets.append(self.d2dnfm.Z)
			# fig1 = plt.figure('s2d')
			# self.display(s2d_nsheets[i, :,:], i, fig1)
			# fig2 = plt.figure('d2d')
			# self.display(self.d2dnfm.Z, i, fig2)

		d2d_nsheets = np.array(d2d_nsheets)
		phiNFM, rNFM = self.instantPhase(d2d_nsheets)

		# for ai in range(10):
		# 	for aj in range(10):
		# 		plt.plot(phiNFM[:, ai, aj])
		# plt.show()

		phiI, rI = self.instantPhase(s2d_nsheets)

		self.deltaw = self.deltaweights_polar(rI, rNFM, phiI, phiNFM)
		# self.deltaw = self.deltaweights_xy(s2d_nsheets, d2d_nsheets)
		return phiNFM, rNFM

	def fit_train_data(self, images, epochs = 40):
		a = []
		assert len(images.shape) == 3
		for i in tqdm(range(epochs)):
			for jj in tqdm(range(0, len(images), 10)):
				self.one_fit_D2D(images[jj])
				# self.d2dnfm.updateWeights(self.deltaw)
				a.append(np.mean(self.deltaw))
		plt.plot(a)
		plt.show()
		pass

	def save(self):
		np.save('./nfm_weghts.npy', self.d2dnfm.Waff)

	def check_resp(self, image, aff_path = './nfm_weights.npy'):
		self.d2dnfm.loadWeights(path=aff_path)
		phase, _ = self.one_fit_D2D(image)
		phase_map = np.mean(phase - phase[:, 0,0].reshape(-1, 1, 1), 0)
		return phase_map



if __name__ == '__main__':
	nfm = NFM()
	images = OrientationBars.reshape(-1, 10, 10)
	# count = 1
	# for i in range(0, 90, 1):
	# 	pm = nfm.check_resp(images[i])
	# 	pm[pm < 0.9*np.max(pm)] = 0
	# 	plt.subplot(10, 9, count)
	# 	plt.imshow(pm)
	# 	count += 1
	# plt.show()
	# nfm.fit_train_data(images, epochs = 1)
	# nfm.save()

	weights = np.load('./SOM_weights.npy').reshape(10,10,10,10)
	a = np.empty((100,100))
	for i in range(10):
		for j in range(10):
			a[i*10:(i+1)*10, j*10:(j+1)*10] = weights[i,j,:,:]
	plt.imshow(a)
	plt.show()
