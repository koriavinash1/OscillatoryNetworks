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
from NFM import NFM
import os
import pandas as pd
from tqdm import tqdm
from sklearn import linear_model
from scipy.signal import hilbert
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model

Gstat = GaussianStatistics()
config= Config()

# for 3D plot...
X = np.arange(0, config.N)
Y = np.arange(0, config.N)
X, Y = np.meshgrid(X, Y)

# ----------------------------------------------

def display(Z, _type = '3d'):
    if _type == '3d':
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

def moving_average(a, n=5) :
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
                        nsheets.append(nfm.Z)

                    # if i % 100 == 0:
                    #     display(nfm.Z)

                nsheets = np.array(nsheets)
                ref_wave = nsheets[:,0,0]
                ref_phase = np.unwrap(np.angle(hilbert(ref_wave)))
                if float(np.sum(abs(np.diff(ref_phase)) < 0.1))/len(ref_phase) > 0.90:
                    ref_phase = np.zeros_like(ref_phase)

                for ii in range(config.N):
                    for jj in range(config.N):
                        wave = nsheets[:,ii,jj]
                        phase_wave = np.unwrap(np.angle(hilbert(wave)))
                        if float(np.sum(abs(np.diff(phase_wave)) < 0.1))/len(phase_wave) > 0.90:
                            phase_wave = np.zeros_like(phase_wave)


                        # if ii == 5 and jj == 5:
                        #     plt.subplot(5, 1, 1)
                        #     plt.plot(ref_phase)
                        #     plt.subplot(5, 1, 2)
                        #     plt.plot(ref_wave)
                        #     plt.subplot(5, 1, 3)
                        #     plt.plot(wave)
                        #     plt.subplot(5, 1, 4)
                        #     plt.plot(phase_wave)
                        #     plt.subplot(5, 1, 5)
                        #     plt.plot(np.sin(phase_wave - ref_phase))
                        #     plt.show()

                        Xdata.append(np.sin(phase_wave - ref_phase)[config.transtime:])
                        Ydata.append(ang)

        Xdata = np.array(Xdata)
        Ydata = np.array(Ydata)
        index = np.random.randint(0, len(Xdata), len(Xdata))
        Xdata = Xdata[index]
        Ydata = Ydata[index]

        path = './PM/data/ci_'+str(config.ci)
        if not os.path.exists(path): os.mkdir(path)
        np.save(os.path.join(path, 'Xdata.npy'), Xdata)
        np.save(os.path.join(path, 'Ydata.npy'), Ydata)
    pass

def train_case():
    # Ypred = model_prediction*90/config.nOrientations
    acc, loss, ci_list = [], [], []
    for ci in tqdm(config.ci_range):
        config.ci = ci
        path = './PM/data/ci_'+str(config.ci)
        Xdata = np.load(os.path.join(path, 'Xdata.npy'))
        Ydata = np.load(os.path.join(path, 'Ydata.npy'))
        total = len(Xdata)
        Xtrain, Ytrain = Xdata[: int(0.7*total)], Ydata[: int(0.7*total)]
        Xtest, Ytest = Xdata[int(0.7*total):], Ydata[int(0.7*total):]
        Ytrain = keras.utils.to_categorical(Ytrain, num_classes=config.nOrientations)
        Ytest = keras.utils.to_categorical(Ytest, num_classes=config.nOrientations)

        model = Sequential([
                    Dense(4, input_shape=(int(config.T/config.sdt) - config.transtime,)),
                    # Activation('relu'),
                    Dropout(0.2),
                    # Dense(4),
                    Activation('softmax')
                ])

        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # Train the model, iterating on the data in batches of 32 samples
        model.fit(Xtrain, Ytrain, epochs=50, batch_size=32, validation_data=(Xtest, Ytest))
        # reg = linear_model.LinearRegression()
        # reg.fit(Normalize(Xtrain), Ytrain)

        score = model.evaluate(Xtest, Ytest, batch_size=128)
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
    df.to_csv('./PM/logs/ci_acc.csv')
    pass


def test_case(angle = 135):
    Orientation_bars = np.load(config.Orientation_path)
    som = SOM(data=Orientation_bars, output_size = (10,10))
    SOM_wts  = np.load(config.SOM_weights_path)
    acc_ci, pred_ci, recon_ci = [], [], []
    for ci in tqdm(config.ci_range):
        config.ci = ci
        # path = os.path.join('./PM/data/ci_'+str(config.ci), 'weights.npy')
        path = os.path.join('./PM/data/ci_'+str(config.ci), 'model.h5')
        predictions, accs, recon_acc = [], [], []
        # coeffs = np.load(path)

        model = load_model(path)
        test_bar = Orientation_bars[angle//2,:]
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
                    display(nfm.Z)
                    nsheets.append(nfm.Z)

            nsheets = np.array(nsheets)

            # if j == 5:
            #     plt.plot(nsheets[:, 0, 4])
            #     plt.show()

            ref_wave = nsheets[:,0,0]
            ref_phase = np.unwrap(np.angle(hilbert(ref_wave)))
            if float(np.sum(abs(np.diff(ref_phase)) < 0.1))/len(ref_phase) > 0.95:
                ref_phase = np.zeros_like(ref_phase)

            temp_pred, temp_acc = [], []
            for ii in range(config.N):
                for jj in range(config.N):
                    # pred = coeffs.dot(sheets[:,i,j][config.transtime:])
                    wave = nsheets[:,ii,jj]
                    phase_wave = np.unwrap(np.angle(hilbert(wave)))
                    if float(np.sum(abs(np.diff(phase_wave)) < 0.1))/len(phase_wave) > 0.95:
                        phase_wave = np.zeros_like(phase_wave)

                    # if i == 5 and j == 5:
                    #     plt.subplot(3, 1, 1)
                    #     plt.plot(wave)
                    #     plt.subplot(3, 1, 2)
                    #     plt.plot(phase_wave)
                    #     plt.subplot(3, 1, 3)
                    #     plt.plot(np.sin(moving_average(np.unwrap(phase_wave - ref_phase))))
                    #     plt.show()

                    pred = model.predict(np.sin(phase_wave - ref_phase)[config.transtime:].reshape(1, int(config.T/config.sdt)-config.transtime), batch_size = 1)
                    pred = np.argmax(pred)
                    pred = 2*pred*90/config.nOrientations
                    temp_pred.append(pred)

            unique, counts = np.unique(np.array(temp_pred), return_counts=True)
            print unique, counts, unique == angle
            pred = unique[np.argmax(counts)]
            predictions.append(pred)
            accs.append(abs(angle - pred) <= 90/config.nOrientations)
            try:
                recon_acc.append(float(counts[unique == angle]) / np.sum(counts))
            except:
                recon_acc.append(0.0)

        predictions = np.array(predictions)
        accs = np.array(accs)
        recon_acc = np.array(recon_acc)

        pred_ci.append(np.mean(predictions))
        acc_ci.append(np.mean(accs))
        recon_ci.append(np.mean(recon_acc))

    # plt.plot(config.ci_range, pred_ci)
    # plt.show()

    plt.plot(config.ci_range, acc_ci)
    plt.show()

    # recon_ci = Normalize(recon_ci[1:])
    # moving_average(recon_ci, n=2)

    plt.plot(config.ci_range, recon_ci)
    plt.show()
    # np.save('./PM/logs/test_case_angle_'+str(angle)+'.npy', recon_ci)
    pass


#
# Generate_data()
# train_case()
test_case()







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
