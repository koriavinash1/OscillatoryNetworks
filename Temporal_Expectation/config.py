import numpy as np

class Config():
	def __init__(self):
		self.ita = 1e-3
		self.dt  = 0.01
		self.T   = 100
		self.N   = 10

		# Lateral training 
		self.TrainingTime = int(self.T/self.dt)
		self.saveLat  = (self.TrainingTime == int(self.T/self.dt))

		# time for projection ...
		self.rdeltaT = 1500
		self.gdeltaT = 1500
		self.omega  = 2.*np.pi/ 25.0
		self.mu     = 1
		self.eps    = 1

		self.Thresh = 0.0015
		self.generate = self.Train = True	
		self.random   = False

		self.epoch = 50
	pass