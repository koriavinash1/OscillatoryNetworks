from Oscillators import *
from configure import Config
from tqdm import tqdm
from GaussianStatistics import *
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import pandas as pd
import pdb

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
	def __init__(self, ci=0.3, test=False):
		self.ci    = ci
		self.ita   = 0.001
		self.d2dnfm = FreqAdaptiveCoupledNFM_D2D(size=(config.N, config.N),
						exe_rad = config.eRad,
						inhb_rad = config.N, # for global inhabition
						exe_ampli = config.eA,
						inhb_ampli = config.iA,
						test = test)# self.phase is (T/dt, N, N) shape

	def display(self, Z, i, fig,  _type = '3d'):
		plt.clf()
		plt.ion()
		if _type == '3d':
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(X, Y, np.array(Z, dtype='float') , cmap=cm.coolwarm,
			               linewidth=0, antialiased=False)
			ax.set_title(np.max(Z))
			# Customize the z axis.
			ax.set_zlim(-1.01, 1.01)
			ax.zaxis.set_major_locator(LinearLocator(10))
			ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
			fig.colorbar(surf, aspect=10)
		else:
			plt.imshow(Z)

		plt.draw()
		plt.xlabel(str(i))
		plt.pause(0.0005)
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
	# def instantPhase(self, tensor):
	# 	""
	# 	# tensor shape -> (T/dt, N, N)
	# 	# tensor numpy array
	# 	print ("in instantanious phase calculation",tensor.shape)
	# 	assert len(tensor.shape) == 3
	# 	phases, radius = [], []
	# 	for x in tensor.reshape(-1, np.prod(tensor.shape[1:])).T:
	# 		# pdb.set_trace()
	# 		h = hilbert(x)
	# 		phases.append(np.angle(h))
	# 		radius.append(np.abs(h))
	# 	return np.array(phases).T.reshape(-1, 10, 10), \
	# 			np.array(radius).T.reshape(-1, 10, 10)


	def instantPhase(self, tensor, wt=None):
		""
		# tensor shape -> (T/dt, N, N)
		# tensor numpy array
		# print ("in instantanious phase calculation",tensor.shape)
		assert len(tensor.shape) == 3
		# if np.any(wt) != None:
		# 	tensor = [np.sum(wt.copy().dot(tensor[i, :, :]), axis=(2,3)) for i in range(len(tensor))]
		# 	tensor = np.array(tensor)

		phases, radius = np.zeros_like(tensor), np.zeros_like(tensor)
		for ii in range(config.N):
			for jj in range(config.N):
				h = hilbert(tensor[:,ii,jj])
				phases[:, ii, jj] = np.angle(h)
				radius[:, ii, jj] = np.abs(h)
		return phases, radius


	# weight update rule 
	# https://link.springer.com/content/pdf/10.1007%2Fs11571-018-9489-x.pdf
	def deltaweights_polar(self, rI, rNFM, argI, argNFM):
		# rI, rNFM 3D spatio-temporal signal
		# returns (N, N, N, N) 4D weight delta weights
		rI, rNFM, argI, argNFM = rI/np.max(rI), rNFM/np.max(rNFM), argI/np.max(argI), argNFM/np.max(argNFM)
		# print (np.max(iterable, key)(rNFM), np.min(rNFM), np.max(rI), np.min(rI))

		deltaw = np.empty((config.N, config.N, config.N, config.N)) 
		for ii in range(config.N):
			for jj in range(config.N):
				NFM = rNFM[:, ii, jj].reshape(-1, 1, 1)
				arg = argNFM[:,ii,jj].reshape(-1, 1, 1)

				relative_phase = abs(argI - arg)
				
				# for ll in range(arg.shape[1]):
				# 	for kk in range(arg.shape[2]):
				# 		ref_phase = abs(argI[:, ll, kk] - arg[:,0,0])
				# 		# print (ref_phase.shape, argI.shape, arg.shape)
				# 		if float(np.sum(abs(ref_phase) < 2))/len(ref_phase) > 0.90:
				# 			ref_phase = np.zeros_like(ref_phase)
				# 		relative_phase[:, ll, kk] = ref_phase

				# deltaw[ii, jj, :, :] = 1*self.ita*np.cos(np.mean(relative_phase, axis=0))
				# deltaw[ii, jj, :, :] = 1*self.ita*np.mean(NFM*rI, axis=0)
				temp = 1*self.ita*np.mean(NFM*rI*(1+np.cos(relative_phase)), axis=0)
				# print (np.max(np.cos(relative_phase)), np.min(np.cos(relative_phase)))
				# temp[temp < 0.90*abs(np.macostoolsax(temp))] = 0
				# plt.imshow(rI[0, :, :])
				# plt.show()
				deltaw[ii, jj, :, :] = temp 

		# Winning statistics..........
		# mean = np.mean(deltaw, axis = 0)
		# deltaw[deltaw < np.max(mean)] = 0
		return deltaw

	def deltaweights_xy(self, s2d, d2d):
		# rI, rNFM 3D spatio-temporal signal
		# returns (N, N, N, N) 4D weight delta weights
		deltaw = np.empty((config.N, config.N, config.N, config.N)) 
		for ii in range(config.N):
			for jj in range(config.N):
				t = d2d[:, ii, jj].reshape(-1, 1, 1)
				deltaw[ii, jj, :, :] = self.ita*np.mean(s2d*t, axis=0)

		# Winning statistics..........
		# mean = np.mean(deltaw, axis = 0)
		# deltaw[deltaw < 0.95* np.max(mean)] = 0
		return deltaw

	def calculate_winning_statistics(self,input_phase, NFM_phase):
		relative_phase = abs(input_phase - NFM_phase)
		
		for ii in range(NFM_phase.shape[1]):
			for jj in range(NFM_phase.shape[2]):
				ref_phase = abs(NFM_phase[:, ii, jj] - input_phase[:, ii, jj])
				if float(np.sum(abs(ref_phase) < 1))/len(ref_phase) > 0.90:
					ref_phase = np.zeros_like(ref_phase)
				relative_phase[:, ii, jj] = ref_phase

		mean_  = np.mean(relative_phase, axis = 0)
		mean_c = np.cos(mean_)

		# mean_c[mean_c < abs(np.max(mean_c))] = 0
		# mean_c[mean_c == np.max(mean_c)]     = 1 
		return  mean_c

	# Dynamic update without SOM layer
	def one_fit_D2D(self, image):
		"fits one image at a time Train NFM end2end"
		binary_image = image.copy()
		binary_image[binary_image > 0.5] = 1.0
		binary_image[binary_image < 0.5] = -1.0

		# plt.plot(np.sin((np.arange(0,5000)/100)*2*3.14*0.02))
		# plt.show()

		# convert static to dynamic input...
		s2d_nsheets = [c*np.sin(np.arange(0,int(config.T/config.dt))*config.dt*2*3.14*0.01) for r in binary_image for c in r]

		s2d_nsheets = np.array(s2d_nsheets)
		s2d_nsheets = s2d_nsheets.T.reshape(-1, config.N, config.N)
		# pdb.set_trace()

		d2d_nsheets = []
		for i in range(0, int(config.T/config.dt)):
			self.d2dnfm.lateralDynamics(aff = s2d_nsheets[i,:,:], verbose = False, ci = 0.1)
			d2d_nsheets.append(self.d2dnfm.Z)
			# if i == 0:
			# 	plt.plot(s2d_nsheets[:,4,2])
			# 	plt.show()
			# fig1 = plt.figure('s2d')
			# self.display(s2d_nsheets[i, :,:], i, fig1)
			# fig2 = plt.figure('d2d')
			# self.display(self.d2dnfm.Z, i, fig2)

		d2d_nsheets  = np.array(d2d_nsheets)
		phiNFM, rNFM = self.instantPhase(d2d_nsheets)
		phiI,   rI   = self.instantPhase(s2d_nsheets, self.d2dnfm.Waff)

		phs_diff   = np.zeros_like(phiNFM)
		for ii in range(phiNFM.shape[1]):
			for jj in range(phiNFM.shape[2]):
				ref_phase = abs(phiNFM[:, ii, jj] - phiI[:, ii, jj])
				if float(np.sum(abs(ref_phase) < 2))/len(ref_phase) > 0.90:
					ref_phase = np.zeros_like(ref_phase)
				phs_diff[:, ii, jj] = ref_phase


		# plt.ion()
		# for i in range(0, int(config.T/config.dt), 100):
		# plt.subplot(1,6,1)
		# plt.imshow(s2d_nsheets[10])
		# plt.subplot(1,6,2)
		# plt.imshow(np.mean(s2d_nsheets, axis=0))
		# plt.subplot(1,6,3)
		# plt.imshow(np.mean(phiI, axis=0))
		# plt.subplot(1,6,4)
		# plt.imshow(np.mean(phiI, axis=0) == np.max(np.mean(phiI, axis=0)))
		# plt.subplot(1,6,5)
		# plt.imshow(np.mean(phiNFM, axis=0) -  np.mean(phiI, axis=0))
		# plt.subplot(1,6,6)
		# a = np.mean(phiNFM, axis=0) -  np.mean(phiI, axis=0)
		# plt.imshow(a == np.max(a))
		# plt.show()


		# plt.ion()
		# for t in range(0, int(config.T/config.dt), 100):
		# 	plt.subplot(1,6,1)
		# 	plt.imshow(image)
			
		# 	plt.subplot(1,6,2)
		# 	plt.imshow(rI[t, :, :])
			
		# 	plt.subplot(1,6,3)
		# 	plt.imshow(rNFM[t, :, :])
			
		# 	plt.subplot(1,6,4)
		# 	plt.imshow(rNFM[t, :, :] - rI[t, :, :])

		# 	plt.subplot(1,6,5)
		# 	plt.imshow(np.abs(phs_diff[t, :, :]))

		# 	plt.subplot(1,6,6)
		# 	plt.imshow(abs(phiNFM[t, :, :] - phiI[t, :, :]))
		# 	plt.xlabel(str(t))

		# 	plt.pause(0.005)
		# 	plt.show()


		# tensor = [np.sum(self.d2dnfm.Waff.copy().dot(s2d_nsheets[i, :, :]), axis=(2,3)) for i in range(len(s2d_nsheets))]
		# tensor = np.array(tensor)
		# plt.ion()
		# for ai in range(0, 10):
		# 	for aj in range(0, 10):
		# 		plt.clf()
		# 		plt.subplot(1,5,1)
		# 		plt.plot(np.cos(phiNFM[:, ai,aj] - phiI[:, ai,aj]), 'r')
		# 		plt.plot(np.cos(tensor[:, ai, aj]), 'b')


		# 		plt.subplot(1,5,2)
		# 		plt.plot(phiNFM[:, ai,aj], 'r')
		# 		plt.plot(phiI[:, ai,aj], 'b')

		# 		plt.subplot(1,5,3)
		# 		plt.plot(phiNFM[:, ai,aj] - phiI[:, ai,aj])
				
				
		# 		plt.subplot(1,5,4)
		# 		plt.plot(tensor[:, ai, aj])

		# 		plt.subplot(1,5,5)
		# 		plt.plot(d2d_nsheets[:, ai, aj], 'r')
		# 		plt.plot(s2d_nsheets[:, ai, aj], 'c')

		# 		plt.pause(0.05)
		# 		plt.show()

		self.deltaw = self.deltaweights_polar(rI, rNFM, phiI, phiNFM)
		# self.deltaw = self.deltaweights_xy(s2d_nsheets, d2d_nsheets)
		return phiI, phiNFM

	def fit_train_data(self, images, epochs = 40):
		a = []
		assert len(images.shape) == 3
		for i in tqdm(range(epochs)):
			for jj in tqdm(range(0, len(images), 25)):
				_, _ = self.one_fit_D2D(images[jj])
				self.d2dnfm.updateWeights(self.deltaw)

				# self.display_wts()
				a.append(np.mean(self.deltaw))
		plt.plot(a)
		plt.show()
		pass

	def response(self, images, simulations=0):
		response_maps_per_simulations = np.zeros((4, 10, 10))
		assert len(images.shape) == 3
		for sim in tqdm(range(simulations)):
			# response_maps = np.zeros((4, 10, 10))

			for i, jj in enumerate(range(1, len(images)+1, 25)):
				Iphs, NFMphs = self.one_fit_D2D(images[jj])
				winning_stat = self.calculate_winning_statistics(Iphs, NFMphs)
				# plt.imshow(winning_stat)
				# plt.show()
				response_maps_per_simulations[i, :, :] += winning_stat

			# response_maps_per_simulations[sim, :, :] = np.max(response_maps, axis=0)
		plt.imshow(np.argmax(response_maps_per_simulations, axis=0))
		plt.show()
		pass

	def save(self):
		np.save('./nfm_weights_4.npy', self.d2dnfm.Waff)

	def check_resp(self, image, aff_path = './nfm_weights.npy'):
		self.d2dnfm.loadWeights(path=aff_path)
		phase, _ = self.one_fit_D2D(image)
		phase_map = np.mean(phase - phase[:, 0,0].reshape(-1, 1, 1), 0)
		return phase_map

	def display_wts(self):
		plt.ion()
		weights = self.d2dnfm.Waff.reshape(10,10,10,10)
		a = np.empty((100,100))
		for i in range(10):
			for j in range(10):
				a[i*10:(i+1)*10, j*10:(j+1)*10] = weights[i,j,:,:]
		plt.imshow(a)
		plt.show()
		plt.pause(0.1)



if __name__ == '__main__':
	nfm = NFM(test=False)
	images = OrientationBars.reshape(-1, 10, 10)
	# np.random.shuffle(images)
	# count = 1
	# for i in range(0, 90, 1):
	# 	pm = nfm.check_resp(images[i])
	# 	pm[pm < 0.95*np.max(pm)] = 0
	# 	plt.subplot(10, 9, count)
	# 	plt.imshow(pm)
	# 	count += 1
	# plt.show()

	nfm.fit_train_data(images, epochs = 1200)
	nfm.save()

	nfm.response(images, simulations=2)

	# weights = np.load('./nfm_weights.npy').reshape(10,10,10,10)
	weights = np.load('./nfm_weights_4.npy').reshape(10,10,10,10)
	a = np.empty((100,100))
	for i in range(10):
		for j in range(10):
			a[i*10:(i+1)*10, j*10:(j+1)*10] = weights[i,j,:,:]
	plt.imshow(a)
	plt.show()
