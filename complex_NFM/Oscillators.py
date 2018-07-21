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
# twts = np.zeros((10, 10, 10, 10))
# for i in range(10):
#     for j in range(10):
#         ai = np.random.randint(0, 4)
#         twts[i, j, :, :] = images[ai*22]

# b = np.empty((100, 100))
# for i in range(10):
#     for j in range(10):
#         b[i*10:(i+1)*10, j*10:(j+1)*10] = twts[i,j,:,:]

# plt.imshow(b)
# plt.show()


# ===================================================================================
# NFM 2D implementation
class CoupledNFM(object):
    " D2D is Dynamic input to Dynamic system\
    With all to all connections usage: >> "

    def __init__(self, 
                size       = (10,10), 
                exe_rad    = 2,
                inhb_rad   = 10, 
                exe_ampli  = 2., 
                inhb_ampli = 1., 
                test       = None):

        self.eRad = exe_rad
        self.iRad = inhb_rad

        # gaussian statistics...
        self.eA   = exe_ampli
        self.iA   = inhb_ampli
        
        # complex neural field parameters ...
        self.Z    = 0.05*np.random.randn(size[0], size[1]) + 0.05*np.random.randn(size[0], size[1])*1j
        self.W    = 0.05*np.random.randn(size[0], size[1]) + 0.05*np.random.randn(size[0], size[1])*1j

        # lateral connection defination ...
        self.Wlat     = np.zeros((size[0], size[1], size[0], size[1]), dtype='complex64')
        self.Wlattemp = np.zeros((size[0], size[1], size[0]+2*self.iRad, size[1]+2*self.iRad), dtype='complex64')
        

        # weight initialization with real and imaginary maxican hat
        for i in range(self.iRad, size[0]+self.iRad):
            for j in range(self.iRad, size[1]+self.iRad):
                # lateral weights
                gauss1, gauss2   = Gstat.DOG(self.iRad, self.eRad, self.iA, self.eA)
                # Gstat.Visualize(gauss1+gauss2, _type = '2d')
                self.kernel= gauss1 - gauss2 + (gauss1 - gauss2)*1j
                self.Wlattemp[i-self.iRad, j-self.iRad, i-self.iRad:i+self.iRad, j-self.iRad:j+self.iRad] = self.kernel

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                self.Wlat[i,j,:,:] = self.Wlattemp[i,j,self.iRad:self.iRad + size[0], self.iRad:self.iRad + size[1]]

        print ("MXH Before Normalization,  max lateral: ", np.max(self.Wlat), "  min lateral: ", np.min(self.Wlat))
        
        # normalization L1
        for ii in range(10):
            for jj in range(10):
                # plt.imshow(self.Wlat[ii,jj,:,:])
                # plt.show()
                self.Wlat[ii, jj, :, :]  = self.Normalize(self.Wlat[ii, jj, :, :])
        
        print ("MXH After normalization,  max lateral: ", np.max(self.Wlat), "  min lateral: ", np.min(self.Wlat))
        pdb.set_trace()


    def Normalize(self, mat, type_='L1'):
        "performs different types of normalization"

        if type_ == 'L1':
            mat = mat/ abs(np.sum(mat))
        elif type_ == 'MinMax':
            mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
            # mat = np.tanh(mat)
        elif type_ == 'Zscore':
            mat = (mat - np.mean(mat))/ np.var(mat)**0.5
        else:
            raise ValueError("Invalid Type found")
        return mat


    def lateralDynamics(self, aff = 1+1j, verbose = True, ci=config.ci):
        "Dynamics of NFM sheet..."

        temp_lat = np.zeros_like(self.Z)
        for m in range(self.Z.shape[0]):
            for n in range(self.Z.shape[1]):
                temp_lat[m,n] = np.sum((self.Z - self.Z[m,n])*self.Wlat[m, n, :, :])
        
        # high self weights ...
        temp_lat = temp_lat + self.Z
        temp_lat = temp_lat if not np.isnan(temp_lat).any() else 0.0

        # temp_lat = self.Normalize(temp_lat)
        temp_aff = 0.05*aff
        temp_lat = 0.05*temp_lat
        I = temp_lat + temp_aff

        v1 = self.Z
        w1 = self.W

        a     =  0.139
        gamma = 2.54
        eps   = 0.008


        fv   = v1*(a - v1)*(v1 - 1)
        vdot = (fv - w1 + I)/0.08
        wdot = eps*(v1 - gamma*w1)

        # fv   = v1*(v1+12)*(1.-v1)
        # vdot = fv - w1 + 0.5*I
        # wdot = omega*(v1 - 0.01*w1)


        v1 = v1 + vdot*config.dt
        w1 = w1 + wdot*config.dt

        self.Z = v1
        self.W = w1

        if verbose:
            print 'max lat.........: {}'.format(np.max(temp_lat)) + '  min lat.......: {}'.format(np.min(temp_lat))
            print 'max aff.........: {}'.format(np.max(temp_aff)) + '  min lat.......: {}'.format(np.min(temp_aff))
            print 'max I ....: {}'.format(np.max(I)) + '  min I....: {}'.format(np.min(I))


    def fanoFactor(self, sig):
        return np.var(sig)/np.mean(sig)