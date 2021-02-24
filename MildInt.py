import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.compat.v1 import keras
from tensorflow.keras.layers import GRU, Dense, Input, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import AUC
from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os


class MildInt(object):
    def __init__(self):
        pass


    def test_train_split(self, X, y):
        # split into train and test
        split_fraction = 0.7
        train_split = int(split_fraction * int(len(X)))

        # split into input and outputs
        train_X, train_y = X[:train_split, :], y[:train_split]
        test_X, test_y = X[train_split:, :], y[train_split:]

        return train_X, train_y, test_X, test_y
    
    
    def run_single_model(self, X, y):
        model = Sequential()

        train_X, train_y, test_X, test_y = self.test_train_split(X, y)

        model.add(GRU(4, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
        # GRU for mri
        # GRU for csf
        # model.add(Dense(units=2, activation='sigmoid'))  # demographic
        model.add(Dense(units=1, activation='sigmoid')) # logistic regression

        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        model.compile(
            loss='mse', 
            optimizer='adam',
            metrics=[AUC()]
        )
        # model.summary()

        # fit network
        gru_history = model.fit(
            train_X, train_y, 
            epochs=100, # try different values
            batch_size=64, 
            validation_data=(test_X, test_y), 
            shuffle=False
        )

        predictions = model.predict(test_X).ravel()
        fpr, tpr, thresholds = roc_curve(test_y, predictions)
        auc_val = auc(fpr, tpr)
        return auc_val

    
    def run_integrated_model(self, demo_X, demo_y, cog_X, cog_y, csf_X, csf_y, mri_X, mri_y):
        # input = Input(shape=(3, ))
        cog_z = GRU(4, activation='tanh')(cog_X)
        csf_z = GRU(4, activation='tanh')(csf_X)
        mri_z = GRU(4, activation='tanh')(mri_X)
        demo_z = Dense(2, activation='relu')(demo_X)

        z = concatenate(cog_z, csf_z, mri_z, demo_z)
        output = Dense(1, activation='sigmoid')(z)

        model = Model(
            input=[cog_X, csf_X, mri_X, demo_X], 
            output=[cog_z, csf_z, mri_z, demo_z]
        )

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy'
        )

        


    

