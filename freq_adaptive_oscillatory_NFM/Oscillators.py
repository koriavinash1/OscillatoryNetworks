import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from configure import Config
from SOM import SOM

Gstat = GaussianStatistics()
config= Config()

# NFM 2D implementation
#===============================================================================================================

class FreqAdaptiveCoupledNFM(object):
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
        # TODO:  all to all connection
        temp_eff = self.aff

        # input manipulation for freq_control...
        # temp_eff = self.Normalize(temp_eff)
        temp_eff = 0.99*temp_eff + 0.010
        temp_eff = 1.01 - temp_eff


        temp_inh = np.zeros_like(self.Z)
        for m in range(self.Z.shape[0]):
            for n in range(self.Z.shape[1]):
                temp_inh[m,n] = np.sum((self.Z - self.Z[m,n])*self.Wlat[m, n, :, :])
        # temp_inh = self.Normalize(temp_inh)
        # temp_inh[temp_inh<0] = 0
        # I = config.ai*temp_eff + config.bi*temp_inh + ci
        # I[I < 0] = 0
        temp_inh = temp_inh + self.Z*np.max(self.Wlat[1,1,:,:])
        v1 = self.Z
        w1 = self.W

        fv   = v1*(config.a - v1)*(v1 - 1)
        vdot = (fv - w1 + config.bi*temp_inh + ci + (1.01-temp_eff)*config.fr)/temp_eff
        wdot = config.b*v1 - config.gamma*w1

        v1 = v1 + vdot*config.dt
        w1 = w1 + wdot*config.dt

        self.Z = v1
        self.W = w1

        # self.Z = self.Normalize(self.Z)
        # self.W = self.Normalize(self.W)
        if verbose:
            # print 'max I: {}'.format(temp_eff) + '  min: {}'.format(temp_eff)
            print 'max I.........: {}'.format(np.max(config.bi*temp_inh)) + '  min ................: {}'.format(np.min(config.bi*temp_inh))

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

#===============================================================================================================

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


#===============================================================================================================

class FreqAdaptiveCoupledNFM_D2D(object):
    " D2D is Dynamic input to Dynamic system\
    With all to all connections usage: >> "

    def __init__(self, size, exe_rad = 2,inhb_rad = 4, exe_ampli = 2., inhb_ampli = 1):
        self.eRad  = exe_rad
        self.iRad  = inhb_rad
        # gaussian statistics...
        self.eA  = exe_ampli
        self.iA  = inhb_ampli
        # neural field parameters
        self.Z     = 0.05*abs(np.random.randn(size[0], size[1]))
        self.W     = 0.05*abs(np.random.randn(size[0], size[1]))
        self.Waff  = 0.05*abs(np.random.randn(size[0], size[1], size[0], size[1]))
        self.Waff  = np.load('./SOM_weights.npy').reshape(10,10,10,10)
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
        self.Wlat = self.Normalize(self.Wlat)


    def Normalize(self, mat, type_='L1'):
        " "
        if type_ == 'L1':
            mat = mat/ np.max(abs(mat))
        elif type_ == 'MinMax':
            mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
        elif type_ == 'Zscore':
            mat = (mat - np.mean(mat))/ np.var(mat)**0.5
        else:
            raise ValueError("Invalid Type found")
        return mat*0.01


    # def lateralDynamics(self, aff,  verbose = True, ci=config.ci):
    #     # TODO:  all to all connection
    #     temp_eff = aff

    #     # input manipulation for freq_control...
    #     # temp_eff = self.Normalize(temp_eff)
    #     temp_eff = 0.99*temp_eff + 0.010
    #     temp_eff = 1.01 - temp_eff


    #     temp_inh = np.zeros_like(self.Z)
    #     for m in range(self.Z.shape[0]):
    #         for n in range(self.Z.shape[1]):
    #             temp_inh[m,n] = np.sum((self.Z - self.Z[m,n])*self.Wlat[m, n, :, :])
    #     # temp_inh = self.Normalize(temp_inh)
    #     # temp_inh[temp_inh<0] = 0
    #     # I = config.ai*temp_eff + config.bi*temp_inh + ci
    #     # I[I < 0] = 0
    #     temp_inh = temp_inh + self.Z*np.max(self.Wlat[1,1,:,:])
    #     v1 = self.Z
    #     w1 = self.W

    #     fv   = v1*(config.a - v1)*(v1 - 1)
    #     vdot = (fv - w1 + config.bi*temp_inh + ci + (1.01-temp_eff)*config.fr)/temp_eff
    #     wdot = config.b*v1 - config.gamma*w1

    #     v1 = v1 + vdot*config.dt
    #     w1 = w1 + wdot*config.dt

    #     self.Z = v1
    #     self.W = w1

    #     # self.Z = self.Normalize(self.Z)
    #     # self.W = self.Normalize(self.W)
    #     if verbose:
    #         # print 'max I: {}'.format(temp_eff) + '  min: {}'.format(temp_eff)
    #         print 'max I.........: {}'.format(np.max(config.bi*temp_inh)) + '  min ................: {}'.format(np.min(config.bi*temp_inh))

    def lateralDynamics(self, aff, verbose = True, ci=config.ci):
        """

        """
        temp_aff = np.sum(self.Waff.dot(aff), axis=(2,3))
        temp_aff = self.Normalize(temp_aff, type_='Zscore')
        temp_lat = np.zeros_like(self.Z)

        for m in range(self.Z.shape[0]):
            for n in range(self.Z.shape[1]):
                temp_lat[m,n] = np.sum((self.Z - self.Z[m,n])*self.Wlat[m, n, :, :])
        
        temp_lat = temp_lat + self.Z # to ensure self is always high

        I = 0.6*temp_lat + ci

        v1 = self.Z
        w1 = self.W

        fv   = v1*(0.5 - v1)*(v1 - 1)
        vdot = (fv - w1 + I)/0.08
        wdot = 0.1*v1 - 0.1*w1

        v1 = v1 + vdot*config.dt
        w1 = w1 + wdot*config.dt

        self.Z = v1
        self.W = w1

        if verbose:
            print 'max I.........: {}'.format(np.max(config.bi*temp_inh)) + '  min ..........: {}'.format(np.min(config.bi*temp_inh))

    def updateWeights(self, deltaw):
        self.Waff -= deltaw
        self.Waff  = self.Normalize(self.Waff)
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
