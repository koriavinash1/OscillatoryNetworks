import numpy as np

class Config():
	def __init__(self):


		# time for projection ...
		self.microT  = 60

		self.rdeltaT = self.gdeltaT = 1200
		self.omega  = 2.*np.pi/ 25.0
		self.mu     = 1
		self.eps    = 1

		self.Thresh = 0.0015
		self.generate = False
		self.Train = False
		self.random   = False
		

		self.ita = 1e-3
		self.T   = 200 if not self.generate else 1000
		self.dt  = 0.01
		self.N   = 10

		# Lateral training 
		self.TrainingTime = 0.5*int(self.T/self.dt)
		self.saveLat  = (self.TrainingTime == int(self.T/self.dt))
		
		self.epoch = 250
	pass