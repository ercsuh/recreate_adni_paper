import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.compat.v1 import keras
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os
from DataManager import *


class MildInt(object):
    def __init__(self, modals):
        self.dm = DataManager()
        self.X = {}
        self.y = self.dm.get_labels()
        self.initialize_data(modals)

    
    def initialize_data(self, modals):
        """
        Get
        """
        for modal in modals:
            X = self.dm.get_modal_data(modal)
            self.X[modal] = X


    def split_data(self):
        """
        Train/test split.
        """
        train_X_data = []
        test_X_data = []
        # for modal in self.X.keys():
        #     X_train, X_test = train_test_split(self.X[modal], test_size=0.3, random_state=0)
        #     train_X_data.append(X_train)
        #     test_X_data.append(X_test)

        split_fraction = 0.7

        for modal in self.X.keys():
            X = self.X[modal]
            train_split = int(split_fraction * int(len(X)))
            train_X, test_X = X[:train_split, :], X[train_split:, :]
            train_X_data.append(train_X)
            test_X_data.append(test_X)
            # print(f"train_X: {train_X.shape}")
            # print(f"test_X: {test_X.shape}")


        train_split = int(split_fraction * int(len(self.y)))
        y_train, y_test = train_test_split(self.y, test_size=0.3, random_state=0)
        # print(f"train_y: {y_train.shape}")
        # print(f"test_y: {y_test.shape}")
        return train_X_data, test_X_data, y_train, y_test

    
    def run_integrated_model(self):
        """
        Builds and runs a Keras functional API model that takes in multi-modal data. 
        """
        print("[INFO] processing data...")
        train_X, test_X, train_y, test_y = self.split_data()

        print("[INFO] creating model...")
        # input tensors
        cog_input = Input(shape=(self.X['cog'].shape[1], self.X['cog'].shape[2]))
        csf_input = Input(shape=(self.X['csf'].shape[1], self.X['csf'].shape[2]))
        mri_input = Input(shape=(self.X['mri'].shape[1], self.X['mri'].shape[2]))
        demo_input = Input(shape=(self.X['demo'].shape[1]))

        # latent tensors
        cog_z = GRU(2, return_sequences=False, activation='linear')(cog_input)
        csf_z = GRU(5, return_sequences=False, activation='linear')(csf_input)
        mri_z = GRU(3, return_sequences=False, activation='linear')(mri_input)
        demo_z = Dense(2, activation='relu')(demo_input)

        # concatentate latent tensors
        z = keras.layers.concatenate([cog_z, csf_z, mri_z, demo_z])

        # logistic regression
        output = Dense(1, activation='sigmoid')(z)

        # create and compile model
        model = Model(
            inputs=[cog_input, csf_input, mri_input, demo_input], 
            outputs=output
        )

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy'
        )

        print("[INFO] model summary...")
        model.summary()

        print("[INFO] training model...")
        model.fit(train_X, train_y, epochs=5, batch_size=1, verbose=1)

        print("[INFO] predicting MCI to AD conversion...")
        pred_y = model.predict(test_X)
        return pred_y, test_y

    
    def evaluate_model(self, y_predictions, y_test):
        eval_metrics = {}
        y_pred = y_predictions.ravel().round()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        eval_metrics['FPR'] = fpr
        eval_metrics['TPR'] = tpr
        eval_metrics['AUC'] = auc(fpr, tpr)
        eval_metrics['ACC'] = accuracy_score(y_test, y_pred)
        eval_metrics['SEN'] = recall_score(y_test, y_pred)
        eval_metrics['SPE'] = precision_score(y_test, y_pred)
        return eval_metrics


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

        


    

