import numpy as np

class Config():
	def __init__(self):


		# time for projection ...
		self.microT  = 200

		self.rdeltaT = 1500
		self.gdeltaT = 1500
		self.omega  = 2.*np.pi/ 25.0
		self.mu     = 1
		self.eps    = 1

		self.Thresh = 0.0015
		self.generate = True
		self.Train = True
		self.random   = False
		

		self.ita = 1e-3
		self.T   = 100 if not self.generate else 10000
		self.dt  = 0.01
		self.N   = 10

		# Lateral training 
		self.TrainingTime = 0 #int(self.T/self.dt)
		self.saveLat  = (self.TrainingTime == int(self.T/self.dt))
		
		self.epoch = 250
	pass