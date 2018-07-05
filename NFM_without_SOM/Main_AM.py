import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from configure import Config
from tqdm import tqdm
from SOM import SOM
from Oscillators import FreqAdaptiveCoupledNFM as NFM
import os
from tqdm import tqdm
from sklearn import linear_model
from scipy.signal import hilbert
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model
import pandas as pd

Gstat = GaussianStatistics()
config= Config()

# for 3D plot...
X = np.arange(0, config.N)
Y = np.arange(0, config.N)
X, Y = np.meshgrid(X, Y)

# ----------------------------------------------

def display(Z, _type = '2d'):
    if _type == '2d':
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, nfm.Z/np.max(nfm.Z) , cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        ax.set_title(str(i))
        # Customize the z axis.
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, aspect=10)
    else:
        plt.imshow(Z)
    plt.show()
    pass

def Normalize(mat):
    # mat = mat/ np.max(mat)
    # mat = mat / np.sum(abs(mat))
    mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
    # mat = (mat - np.mean(mat))/ np.var(mat)**0.5
    return mat

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def Generate_data():
    Orientation_bars = np.load(config.Orientation_path)
    som = SOM(data=Orientation_bars, output_size = (10,10))
    SOM_wts  = np.load(config.SOM_weights_path)

    for ci in tqdm(config.ci_range):
        config.ci = ci
        Xdata, Ydata = [], []
        for ang in range(config.nOrientations):
            angle = int(ang*90/config.nOrientations)
            test_bar = Orientation_bars[angle,:]
            SOM_output = som.response(test_bar, SOM_wts)
            # print ('SOM max: {}'.format(np.max(SOM_output)) + '  SOM min: {}'.format(np.min(SOM_output)))
            for j in range(config.nSimulations):
                nfm = NFM(size=(config.N, config.N),
                            exe_rad = config.eRad,
                            inhb_rad = config.iRad,
                            exe_ampli = config.eA,
                            inhb_ampli = config.iA,
                            aff = SOM_output)
                nsheets = []
                for i in range(0, int(config.T/config.dt)):
                    nfm.lateralDynamics(verbose = False, ci = ci)
                    if i % int(config.sdt/config.dt) == 0:
                        nsheets.append(nfm.Z.flatten(order='F'))

                nsheets = np.array(nsheets)

                for i in range(100):
                    plt.subplot(4, 3, 3*ang + 1)
                    plt.plot(nsheets[:,i])

                # plt.legend(["(5,9)th neuron", "(5,8)th neuron", "(5,7)th neuron", "(0,0)th neuron"])
                plt.xlabel('time')
                plt.ylabel('Neuron value (V)')
                plt.title(str(2*angle)+" degree")

                plt.subplot(4, 3, 3*ang + 2)
                plt.imshow(np.mean(nsheets, 0).reshape(10,10))
                plt.xlabel("Variance: {:.2f}".format(np.var(nsheets)))
                plt.title("time averaged NFM")
                plt.subplot(4, 3, 3*ang + 3)
                plt.imshow(SOM_output.reshape(10,10).T)
                plt.title("SOM output")

                # print nsheets.shape, np.mean(nsheets[config.transtime:int(config.T/config.sdt),:], 0).shape
                # if j == 5:
                #     plt.imshow(np.mean(nsheets, 0).reshape(10,10))
                #     plt.show()
                Xdata.append(np.mean(nsheets[config.transtime:int(config.T/config.sdt),:], 0))
                Ydata.append(ang)

        plt.show()
        Xdata = np.array(Xdata)
        Ydata = np.array(Ydata)

        index = np.random.randint(0, len(Xdata), len(Xdata))
        Xdata = Xdata[index]
        Ydata = Ydata[index]

        path = './AM/data/ci_'+str(config.ci)
        if not os.path.exists(path): os.mkdir(path)
        np.save(os.path.join(path, 'Xdata.npy'), Xdata)
        np.save(os.path.join(path, 'Ydata.npy'), Ydata)
    pass

def train_case():
    # Ypred = model_prediction*90/config.nOrientations
    acc, loss, ci_list = [], [], []
    for ci in tqdm(config.ci_range):
        config.ci = ci
        path = './AM/data/ci_'+str(config.ci)
        Xdata = np.load(os.path.join(path, 'Xdata.npy'))
        Ydata = np.load(os.path.join(path, 'Ydata.npy'))
        total = len(Xdata)
        Xtrain, Ytrain = Xdata[: int(0.7*total)], Ydata[: int(0.7*total)]
        Xtest, Ytest = Xdata[int(0.7*total):], Ydata[int(0.7*total):]
        Ytrain = keras.utils.to_categorical(Ytrain, num_classes=config.nOrientations)
        Ytest = keras.utils.to_categorical(Ytest, num_classes=config.nOrientations)

        model = Sequential([
                    Dense(256, input_shape=(config.N**2,)),
                    Activation('relu'),
                    Dropout(0.2),
                    Dense(4),
                    Activation('softmax')
                ])

        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # Train the model, iterating on the data in batches of 32 samples
        model.fit(Xtrain, Ytrain, epochs=50, batch_size=4, validation_data=(Xtest, Ytest))
        # reg = linear_model.LinearRegression()
        # reg.fit(Normalize(Xtrain), Ytrain)

        score = model.evaluate(Xtest, Ytest, batch_size=4)
        print(score)
        acc.append(score[1])
        loss.append(score[0])
        ci_list.append(ci)
        model.save(os.path.join(path, 'model.h5'))
        # np.save(os.path.join(path, 'weights.npy'), reg.coef_)

    df = pd.DataFrame()
    df['ci'] = ci_list
    df['val_loss'] = loss
    df['val_acc'] = acc
    df.to_csv('./AM/logs/ci_acc.csv')
    pass


def test_case(angle = 135):
    Orientation_bars = np.load(config.Orientation_path)
    som = SOM(data=Orientation_bars, output_size = (10,10))
    SOM_wts  = np.load(config.SOM_weights_path)
    acc_ci, pred_ci, recon_ci = [], [], []
    for ci in tqdm(config.ci_range):
        config.ci = ci
        # path = os.path.join('./PM/data/ci_'+str(config.ci), 'weights.npy')
        path = os.path.join('./AM/data/ci_'+str(config.ci), 'model.h5')
        predictions, accs, recon_acc = [], [], []
        # coeffs = np.load(path)

        model = load_model(path)
        test_bar = Orientation_bars[angle//2,:]
        SOM_output = som.response(test_bar, SOM_wts)
        # print ('SOM max: {}'.format(np.max(SOM_output)) + '  SOM min: {}'.format(np.min(SOM_output)))
        # plt.imshow(SOM_output)
        # plt.show()
        for j in range(config.nSimulations):
            nfm = NFM(size=(config.N, config.N),
                        exe_rad = config.eRad,
                        inhb_rad = config.iRad,
                        exe_ampli = config.eA,
                        inhb_ampli = config.iA,
                        aff = SOM_output)
            nsheets = []
            for i in range(0, int(config.T/config.dt)):
                nfm.lateralDynamics(verbose = False, ci=ci)
                if i % int(config.sdt/config.dt) == 0:
                    nsheets.append(nfm.Z.flatten(order = 'F'))

            nsheets = np.array(nsheets) # [config.transtime:int(config.T/config.sdt),:]

            pred = model.predict(np.mean(nsheets[config.transtime:int(config.T/config.sdt),:], 0).reshape(1, config.N**2), batch_size = 1)
            pred = np.argmax(pred)

            pred = 2*pred*90/config.nOrientations
            predictions.append(pred)
            accs.append((angle - pred) <= 90/config.nOrientations)


        predictions = np.array(predictions)
        accs = np.array(accs)
        recon_acc = np.array(recon_acc)

        pred_ci.append(np.mean(predictions))
        acc_ci.append(np.mean(accs))

        unique, counts = np.unique(predictions, return_counts=True)
        print unique, counts, unique == angle
        try:
            recon_ci.append(float(counts[unique == angle]) / np.sum(counts))
        except:
            recon_ci.append(0.0)

    # plt.plot(config.ci_range, pred_ci)
    # plt.show()

    # plt.plot(config.ci_range, acc_ci)
    # plt.show()

    # recon_ci = Normalize(recon_ci[1:])
    # moving_average(recon_ci, n=2)
    # print recon_ci
    # plt.plot(config.ci_range, recon_ci)
    # plt.show()
    np.save('./AM/logs/test_case_angle'+str(angle)+'.npy', recon_ci)
    pass


Generate_data()
# train_case()
# a = []
# for angle in [0, 45, 90, 135]:
#     a.append(test_case(angle))
# a = np.array(a)
# np.save('./AM/logs/all.npy', a)





# In[51]:

# for i in range(config.N):
#     for j in range(config.N):
#         if np.max(abs(neuronsheets[:, i, j][config.transtime:])) > 0.1:
#             print i ,j
#             plt.plot(neuronsheets[:, i, j])
#             plt.show()



# Ivec = np.arange(1, 20)/20.0
# Iterations = 5000
# Zold, Wold = np.random.randn(1), np.random.randn(1)
# for I in Ivec:
#     neuron_activity = []
#     for _ in range(Iterations):
#         Z, W = nfm.sanity_check(I, Zold, Wold, dt=0.01)
#         Zold = Z
#         Wold = W
#         neuron_activity.append(Z)
#     print ("I value: {}".format(I))
#     plt.plot(neuron_activity)
#     plt.show()
