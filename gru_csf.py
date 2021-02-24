import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.compat.v1 import keras
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC
import os
from matplotlib import pyplot


#cognitive performance = ['ADNI_MEM','ADNI_EF']
#CSF = ['LOGABETA','LOGTAU','LOGPTAU','LOGPTAU/ABETA','LOGTAU/ABETA']


def gru(X, y):
    model = Sequential()

    # split into train and test
    split_fraction = 0.7
    train_split = int(split_fraction * int(len(X)))

    # split into input and outputs
    train_X, train_y = X[:train_split, :], y[:train_split]
    test_X, test_y = X[train_split:, :], y[train_split:]

    # reshape input to be 3D [samples, time steps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    model.add(GRU(4, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(units=1, activation='sigmoid'))  # sigmoid/softmax
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(
        loss='mae', 
        optimizer='adam',
        metrics=[AUC()]
    )
    model.summary()

    # fit network
    gru_history = model.fit(
        train_X, train_y, 
        epochs=10, # try different values
        batch_size=64, 
        validation_data=(test_X, test_y), 
        shuffle=False
    )
    
    # pyplot.plot(gru_history.history['loss'], label='GRU train', color='red')
    # pyplot.plot(gru_history.history['val_loss'], label='GRU test', color='blue')
    # pyplot.legend()
    # pyplot.show()

    predictions = model.predict(test_X)
    # print()




if __name__ == "__main__":
    cwd = os.getcwd()
    df = pd.read_csv(os.path.join(cwd, "data", "adni_all_garam.csv"))

    # drop RIDs that are not EMCI or LMCI status
    index_names = df[~(df['DX.at.baseline'].isin(['EMCI', 'LMCI']))].index
    df.drop(index_names, inplace=True)

    # extract cdata
    csf_df = df[['RID', 'LOGABETA','LOGTAU','LOGPTAU','LOGPTAU/ABETA','LOGTAU/ABETA']]
 
    X, y = [], []
    max_RID_len = 10  # keep number observations same across all RIDs
    labels = pd.read_csv("data/total_data.tsv", sep="\t")

    for RID in csf_df['RID'].unique():
        curr_RID = csf_df[csf_df['RID'] == RID]

        single = []
        for i in range(max_RID_len):
            if i < len(curr_RID):
                curr = np.array(curr_RID.iloc[i,:][['LOGABETA','LOGTAU','LOGPTAU','LOGPTAU/ABETA','LOGTAU/ABETA']].tolist())
                curr[np.isnan(curr)] = 0
            else: 
                curr = [.0 for j in range(4)]
            single.append(curr)

        X.append(single)
        y.append(labels[labels['RID'] == RID].iloc[0]['Y'])
    
    X = np.array(X)
    y = np.array(y)

    gru(X, y)
    
