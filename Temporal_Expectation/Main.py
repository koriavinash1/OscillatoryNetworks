import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from tqdm import tqdm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from Gratings import *
from Oscillator import CoupledNFM
from config import Config
from SOM import *
import pdb

Gstat = GaussianStatistics()
config= Config()
grts  = Gratings()
som   = SOM()
somwts= np.load('./SOM_weights.npy') 

rand = lambda N: np.random.randn(N, N)

class Main(object):
    def __init__(self, deltaT):
        self.deltaT = deltaT
        self.Zs     = np.zeros((config.N, config.N, int(config.T/config.dt)), dtype='complex64')
        self.Flag   = False
        self.Flagcount  = 0
        self.oscillator = CoupledNFM()
        pass


    def display(self, Z, i, fig,  _type = '3d'):
        plt.clf()
        plt.ion()
        if _type == '3d':
            X = np.arange(0, config.N)
            Y = np.arange(0, config.N)
            X, Y = np.meshgrid(X, Y)
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
        plt.pause(0.5)
        pass


    def runNFM(self, _display = False, blink=True):
        """
        """
        # plt.ion()
        for t in range(0, int(config.T/config.dt), 1):
            
            if t % config.deltaT == 0:
                temp_aff = som.response(rand(config.N), somwts)

            if blink:
                if np.random.randint(1) and t > 3000:
                    self.Flag = True

                if self.Flag and self.Flagcount < config.deltaT:
                    temp_aff = som.response(grts.fixedGrating(theta = 45), somwts) # [45, -45]
                    self.Flagcount += 1
                else:
                    self.Flag = False

            # plt.imshow(temp_aff)
            # plt.title(t)
            # plt.pause(0.005)

            self.oscillator.lateralDynamics(temp_aff)
            if self.Flag and self.Flagcount < config.deltaT: self.oscillator.updateLatWeights()
            self.Zs[:,:, t] = self.oscillator.Z

            if _display and t % 100 == 0: self.display(np.real(self.oscillator.Z), t, plt.figure('plot'))


        return self.Zs

    def filteringSignal(self):
        """
            Delta wave filtering...
            (0 - 4 Hz)
        """
        from scipy.signal import butter, lfilter

        filtered_ZsR = np.zeros((config.N, config.N, int(config.T/config.dt)))
        filtered_ZsI = np.zeros((config.N, config.N, int(config.T/config.dt)))

        fs = 100
        N = 10000
        

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            N, Wn = signal.buttord([1, 8], [2, 10], 1, 10, True)
            b, a = butter(N, Wn, btype='band', analog=False)
            return b, a


        def butter_bandpass_filter(data, fs = 100, lowcut = 0, highcut = 5, order=2):
            x = data
            freqs = np.fft.fftfreq(len(data), 1/fs)
            idx = np.argsort(freqs)
            ps = np.abs(np.fft.fft(x))**2
            plt.figure()
            plt.plot(freqs[idx], ps[idx])
            plt.title('Power spectrum (np.fft.fft)')

            f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
            plt.figure()
            plt.semilogy(f, np.sqrt(Pxx_spec))
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Linear spectrum [V RMS]')
            plt.title('Power spectrum (scipy.signal.welch)')
            plt.show()
            return x

        for xx in range(config.N):
            for yy in range(config.N):
                filtered_ZsR[xx, yy, :] = butter_bandpass_filter(np.real(self.Zs[xx, yy, :]))
                filtered_ZsI[xx, yy, :] = butter_bandpass_filter(np.imag(self.Zs[xx, yy, :]))

        return filtered_ZsR + 1j * filtered_ZsI


    def viewSignals(self, data):
        """
            data format complex number
        """
        # x = np.random.randint(0, config.N, 5)
        # y = np.random.randint(0, config.N, 5)

        # plt.figure('Real')
        # for i in range(5):
        #   plt.subplot(5, 1, i + 1)
        #   plt.plot(np.real(data[x[i], y[i], :]))
        #   plt.title("x:" + str(x[i]) + "  y:" + str(y[i]))

        # plt.figure('Imaginary')
        # for i in range(5):
        #   plt.subplot(5, 1, i + 1)
        #   plt.plot(np.imag(data[x[i], y[i], :]))
        #   plt.title("x:" + str(x[i]) + "  y:" + str(y[i]))

        signal = np.mean(data, axis=(0,1)) 
        plt.subplot(2,1,1)
        plt.plot(np.real(signal))
        plt.subplot(2,1,2)
        plt.plot(np.imag(signal))
        plt.show()
        pass


    def classifier(self, data):
        """
            delta Signal classifier 
        """
        # based on visualization determine threshold..
        signal = np.mean(data, axis=(0,1))
        # print (np.mean(np.real(signal)), np.var(np.real(signal)))
        # print (np.mean(np.imag(signal)), np.var(np.imag(signal)))

        if np.var(np.real(signal))*100 - 0.1 >= config.Thresh:
            return True

        else: return False


    def performExp(self, blink=True):
        """
        NFM => Filter => classifier
        """
        Z  = self.runNFM(blink=blink)
        # plt.plot(Z[5,5,:])
        # plt.show()
        # fZ = self.filteringSignal()
        # self.viewSignals(fZ)
        print (self.classifier(Z))
        return np.max(np.abs(np.real(np.mean(Z, axis=(0,1)))))

if __name__ == '__main__':
    threshFalse = []
    exp = Main(config.deltaT)
    for _ in tqdm(range(15)):
        threshFalse.append(exp.performExp(blink=False))

    threshTrue = []
    exp = Main(config.deltaT)
    for _ in tqdm(range(15)):
        threshTrue.append(exp.performExp())

    plt.plot(np.array(threshTrue), 'r')
    plt.plot(np.array(threshFalse), 'b')
    plt.show()
    print (np.mean(np.array(threshFalse)), np.mean(np.array(threshTrue)))