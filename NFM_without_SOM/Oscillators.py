import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from configure import Config
from SOM import SOM
import pdb
Gstat = GaussianStatistics()
config= Config()


OrientationBars = np.load('./Orientation_bars.npy')
images = OrientationBars.reshape(-1, 10, 10)
twts = np.zeros((10, 10, 10, 10))
for i in range(10):
    for j in range(10):
        ai = np.random.randint(0, 4)
        twts[i, j, :, :] = images[ai*22]

b = np.empty((100, 100))
for i in range(10):
    for j in range(10):
        b[i*10:(i+1)*10, j*10:(j+1)*10] = twts[i,j,:,:]

plt.imshow(b)
plt.show()

# NFM 2D implementation
#========================================================================================
# NFM S2D..................

class CoupledNFM(object):
    def __init__(self, size, exe_rad = 2,inhb_rad = 4, exe_ampli = 2., inhb_ampli = 1., aff = 1):
        self.eRad  = exe_rad
        self.iRad  = inhb_rad
        # gaussian statistics...
        self.eA  = exe_ampli
        self.iA  = inhb_ampli
        # neural field parameters
        self.Z     = 0.05*abs(np.random.randn(size[0], size[1]))
        self.W     = 0.05*abs(np.random.randn(size[0], size[1]))

        self.Wlat  = np.zeros((size[0], size[1], size[0], size[1]))
        self.Wlattemp  = np.zeros((size[0], size[1], size[0]+2*self.iRad, size[1]+2*self.iRad))
        self.aff   = aff
        for i in range(self.iRad, size[0]+self.iRad):
            for j in range(self.iRad, size[1]+self.iRad):
                # lateral weights
                gauss1, gauss2   = Gstat.DOG(self.iRad, self.eRad, self.iA, self.eA)
                # Gstat.Visualize(gauss1+gauss2, _type = '2d')
                self.kernel= gauss1 - gauss2
                # print self.kernel.shape
                self.Wlattemp[i-self.iRad, j-self.iRad, i-self.iRad:i+self.iRad, j-self.iRad:j+self.iRad] = self.kernel

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                self.Wlat[i,j,:,:] = self.Wlattemp[i,j,self.iRad:self.iRad + size[0], self.iRad:self.iRad + size[1]]
        self.Wlat = self.Normalize(self.Wlat)
        # plt.imshow(self.Wlat[5,5,:,:])
        # plt.show()

    def Normalize(self, mat):
        mat = mat/ np.max(abs(mat))
        # mat = mat / np.sum(abs(mat))
        # mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
        # mat = (mat - np.mean(mat))/ np.var(mat)**0.5
        return mat*0.01

    def lateralDynamics(self, verbose = True, ci=config.ci):
        temp_eff = self.aff
        temp_inh = np.zeros_like(self.Z)

        for m in range(self.Z.shape[0]):
            for n in range(self.Z.shape[1]):
                temp_inh[m,n] = np.sum((self.Z - self.Z[m,n])*self.Wlat[m, n, :, :])

        # temp_inh = self.Normalize(temp_inh)
        # temp_inh[temp_inh<0] = 0
        I = config.ai*temp_eff + config.bi*temp_inh + ci
        # I[I < 0] = 0
        # temp_inh = temp_inh + self.Z*np.max(self.Wlat[1,1,:,:])

        v1 = self.Z
        w1 = self.W

        fv   = v1*(config.a - v1)*(v1 - 1)
        vdot = (fv - w1 + I)/config.freq_ctrl
        wdot = config.b*v1 - config.gamma*w1

        v1 = v1 + vdot*config.dt
        w1 = w1 + wdot*config.dt

        self.Z = v1
        self.W = w1

        # self.Z = self.Normalize(self.Z)
        # self.W = self.Normalize(self.W)
        if verbose:
            # print 'max I: {}'.format(temp_eff) + '  min: {}'.format(temp_eff)
            print 'max I.........: {}'.format(np.max(I)) + '  min ................: {}'.format(np.min(I))

    def fanoFactor(self, sig):
        return np.var(sig)/np.mean(sig)

    def sanity_check(self, I, Zold, Wold, dt=0.01):
        """
            To find oscillatory regimes for single neuron
        """
        v1 = Zold
        w1 = Wold

        fv = v1*(0.5 - v1)*(v1 - 1)
        vdot = float(fv - w1 + I)/config.freq_ctrl
        wdot = 0.1*v1 - 0.1*w1

        v1 = v1 + vdot*dt
        w1 = w1 + wdot*dt

        # v1[v1 < 0] = 0
        return v1, w1


#============================================================================================
# NFM D2D

class FreqAdaptiveCoupledNFM_D2D(object):
    " D2D is Dynamic input to Dynamic system\
    With all to all connections usage: >> "

    def __init__(self, 
                size=(10,10), 
                exe_rad = 2,
                inhb_rad = 10, 
                exe_ampli = 20., 
                inhb_ampli = 10., 
                test=None):

        self.eRad  = exe_rad
        self.iRad  = inhb_rad

        # gaussian statistics...
        self.eA  = exe_ampli
        self.iA  = inhb_ampli
        
        # neural field parameters
        self.o     = 10*abs(np.random.randn(size[0], size[1]))
        self.Z     = 0.05*abs(np.random.randn(size[0], size[1]))
        self.W     = 0.05*abs(np.random.randn(size[0], size[1]))
        

        if test == None:
            # self.Waff  = np.random.uniform(0, 1, (size[0], size[1], size[0], size[1]))
            # self.Waff  = np.load('./SOM_weights.npy').reshape(10,10,10,10)
            # np.random.shuffle(self.Waff)
            self.Waff = twts
            
            for ii in range(10):
                for jj in range(10):
                    self.Waff[ii, jj, :, :]  = self.Normalize(self.Waff[ii, jj, :, :])
            
        else:
            self.Waff  = np.load(test).reshape(10,10,10,10)


        # lateral connection defination..............
        self.Wlat  = np.zeros((size[0], size[1], size[0], size[1]))
        self.Wlattemp  = np.zeros((size[0], size[1], size[0]+2*self.iRad, size[1]+2*self.iRad))
        
        for i in range(self.iRad, size[0]+self.iRad):
            for j in range(self.iRad, size[1]+self.iRad):
                # lateral weights
                gauss1, gauss2   = Gstat.DOG(self.iRad, self.eRad, self.iA, self.eA)
                # Gstat.Visualize(gauss1+gauss2, _type = '2d')
                self.kernel= gauss1 - gauss2
                self.Wlattemp[i-self.iRad, j-self.iRad, i-self.iRad:i+self.iRad, j-self.iRad:j+self.iRad] = self.kernel

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                self.Wlat[i,j,:,:] = self.Wlattemp[i,j,self.iRad:self.iRad + size[0], self.iRad:self.iRad + size[1]]

        print ("MXH Before Normalization,  max lateral: ", np.max(self.Wlat), "  min lateral: ", np.min(self.Wlat))
        
        for ii in range(10):
            for jj in range(10):
                # plt.imshow(self.Wlat[ii,jj,:,:])
                # plt.show()
                self.Wlat[ii, jj, :, :]  = self.Normalize(self.Wlat[ii, jj, :, :])
        
        print ("MXH After normalization,  max lateral: ", np.max(self.Wlat), "  min lateral: ", np.min(self.Wlat))
        # plt.imshow(self.Wlat[4,5,:,:])
        # plt.show()

    def Normalize(self, mat, type_='L1'):
        " "
        if type_ == 'L1':
            mat = mat/ np.sum(abs(mat))
        elif type_ == 'MinMax':
            mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
            # mat = np.tanh(mat)
        elif type_ == 'Zscore':
            mat = (mat - np.mean(mat))/ np.var(mat)**0.5
        else:
            raise ValueError("Invalid Type found")
        return mat


    def lateralDynamics(self, aff, verbose = True, ci=config.ci):
        """

        """
        temp_aff = np.sum(self.Waff.copy()*(aff.reshape(1,1,10, 10)), axis=(2,3))
        temp_aff = temp_aff if not np.isnan(temp_aff).any() else 0.0


        temp_lat = np.zeros_like(self.Z)

        for m in range(self.Z.shape[0]):
            for n in range(self.Z.shape[1]):
                temp_lat[m,n] = np.sum((self.Z - self.Z[m,n])*self.Wlat[m, n, :, :])
        
        temp_lat = temp_lat + self.Z # high self weights........
        temp_lat = temp_lat if not np.isnan(temp_lat).any() else 0.0


        temp_aff = 0.1*temp_aff
        temp_lat = 0.025*temp_lat
        I = 0.03 + temp_lat + temp_aff


        v1 = self.Z
        w1 = self.W
        omega = self.o


        a     =  0.139
        gamma = 2.54
        eps   = 0.008


        fv   = v1*(a - v1)*(v1 - 1)
        vdot = (fv - w1 + I)/0.01
        wdot = eps*(v1 - gamma*w1)

        # fv   = v1*(v1+12)*(1.-v1)
        # vdot = fv - w1 + 0.5*I
        # wdot = omega*(v1 - 0.01*w1)

        # omegadot = -0.05*I*(1/np.sqrt(v1**2 + w1**2))

        v1 = v1 + vdot*config.dt
        w1 = w1 + wdot*config.dt
        # omega = omega + omegadot*config.dt

        self.Z = v1
        self.W = w1
        self.o = omega

        if verbose:
            print 'max lat.........: {}'.format(np.max(temp_lat)) + '  min lat.......: {}'.format(np.min(temp_lat))
            print 'max aff.........: {}'.format(np.max(temp_aff)) + '  min lat.......: {}'.format(np.min(temp_aff))
            print 'max I ....: {}'.format(np.max(I)) + '  min I....: {}'.format(np.min(I))



    def updateWeights(self, deltaw):
        self.Waff += deltaw
        for ii in range(10):
            for jj in range(10):
                self.Waff[ii, jj, :, :]  = self.Normalize(self.Waff[ii, jj, :, :])
        pass


    def loadWeights(self, path = './nfm_weights.npy'):
        self.Waff  = np.load(path)
        # print ("Weights loaded shape :{}".format(self.Waff.shape))
        pass


    def fanoFactor(self, sig):
        return np.var(sig)/np.mean(sig)


    def sanity_check(self, I, Zold, Wold, dt=0.01):
        """
            To find oscillatory regimes for single neuron
        """
        v1 = Zold
        w1 = Wold

        fv = v1*(0.5 - v1)*(v1 - 1)
        vdot = float(fv - w1 + I)/config.freq_ctrl
        wdot = 0.1*v1 - 0.1*w1

        v1 = v1 + vdot*dt
        w1 = w1 + wdot*dt

        # v1[v1 < 0] = 0
        return v1, w1