'''
Author: Daniel Uvaydov
The following script is used to accompany datasets from the 2021 INFOCOM paper:

D. Uvaydov, S. D’Oro, F. Restuccia and T. Melodia,
"DeepSense: Fast Wideband Spectrum Sensing Through Real-Time In-the-Loop Deep Learning,"
IEEE INFOCOM 2021 - IEEE Conference on Computer Communications, 2021.

train_test_CNN.py:
This sample script takes the training and testing .h5 dataset generated using previous scripts
(in this case for the SDR case) then trains and tests a simple CNN

Usage:
To train: python train_test_CNN.py train
To test: python train_test_CNN.py test
'''

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LeakyReLU, Flatten, Input
from tensorflow.keras.models import Model
import argparse

# Parse Arguments
parser = argparse.ArgumentParser(description='Training or testing CNN')
parser.add_argument('-m', '--mode', type=str, default="train", help="choose training/testing mode")
parser.add_argument('-snr', '--snr', nargs='+', default=["0", "10"], help="SNR list for training")
parser.add_argument('-t', '--train_name', type=str, default="train_model", help="Name for saved model")
parser.add_argument('-s', '--test_model', type=str, default="test_model", help="Test model name")

args = parser.parse_args()

#Open dataset .h5 file either for training or testing
idx = 0
for snr in args.snr:
    dset_fp = './sim_lte_code/' + 'lte_' + snr + '_128_' + args.mode + '.h5'
    dset = h5py.File(dset_fp, 'r')
    X = dset['X'][()]
    Y = dset['y'][()]
    X = np.swapaxes(X, 0, 2)
    Y = np.swapaxes(Y, 0, 1)
    if idx == 0:
        Xall = X
        Yall = Y
        idx = 1
    else:
        Xall = np.concatenate((Xall, X), axis=0)
        Yall = np.concatenate((Yall, Y), axis=0)

perm = np.random.permutation(Xall.shape[0])
labels = Yall[perm, :]
data = Xall[perm, :, :] 



if args.mode == 'train':

    #Model parameters
    n_classes = 4       #number of classes for SDR case
    dim = data.shape[1]    #Number of I/Q samples being taken as input
    n_channels = 2      #One channel for I and the other for Q

    #Build model
    inputs = Input(shape=(dim, n_channels))
    x = Conv1D(16, 3, input_shape=(dim, n_channels), name='conv1')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv1D(16, 3, name='conv2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='pool1')(x)
    x = Conv1D(32, 5, name='conv3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv1D(32, 5, name='conv4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='pool2')(x)
    x = Flatten()(x)
    x = Dense(64, name='dense1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    outputs = Dense(n_classes, activation='sigmoid', name='out')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    #Compile Model
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    #Train
    model.fit(x=data, y=labels, validation_split=0.1, batch_size=256, epochs=150, verbose=1, shuffle=True)
    model.save(args.train_name)

elif args.mode == 'test':

    #Load a pretrained model
    model = tf.keras.models.load_model(args.test_model)

    #Compile model
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    #Test model
    score = model.evaluate(x=data, y=labels, verbose=1)
    print('Loss: ' + str(score[0]))
    print('Acc: ' + str(score[1]))