"""
This module extracts modal-specific data in numpy format from ADNI. 
Modal options:
   - cognitive performance ('cog')
   - csf biomarkers ('csf')
   - mri imaging ('mri')
"""
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataManager(object):
    def __init__(self):
        self.df = pd.read_csv(
            os.path.join(os.getcwd(), 'data', 'adni_all_garam.csv'),
            sep=','
        )
        # only interested in RIDs with EMCI or LMCI status
        index_names = self.df[~(self.df['DX.at.baseline'].isin(['EMCI', 'LMCI']))].index
        self.df.drop(index_names, inplace=True)
        # unique RIDs
        self.RIDs = self.df['RID'].unique()

    
    def get_labels(self):
        file = os.path.join(
            os.getcwd(), 
            "data", "ADNI_All_Longitudinal_Diagnosis_Data.csv"
        )

        long_df = pd.read_csv(file, sep=',')
        cp_df = pd.DataFrame(columns=[
            'RID', 
            'CP.VISCODE', 
            'DX.at.baseline', 
            'Y']
        )

        for rid in self.RIDs:
            RID_df = long_df.loc[long_df['RID'] == rid]
            cp = RID_df.loc[RID_df['DXCHANGE'] == 'Conversion: MCI to Dementia']
            
            if cp.empty:
                # MCI non-converter
                new_row = {
                    'RID': rid,
                    'CP.VISCODE': RID_df['VISCODE'].iloc[-1],
                    'DX.at.baseline': RID_df['DX at baseline'].iloc[0],
                    'Y': int(0)
                }
                cp_df = cp_df.append(new_row, ignore_index=True)
            else:
                # MCI converter
                new_row = {
                    'RID': rid,
                    'CP.VISCODE': cp['VISCODE'].iloc[0],
                    'DX.at.baseline': cp['DX at baseline'].iloc[0],
                    'Y': int(1)
                }
                cp_df = cp_df.append(new_row, ignore_index=True)
        
        # cp_df.to_csv("labels.csv", sep="\t", index=False)
        return np.array(cp_df['Y'].values.tolist())


    def get_modal_data(self, modal_type):
        """
        Specifies columns from ADNI data that are associated with the modal type. 
        @param modal (string): 'cog', 'csf', or 'mri'
        @return (numpy): Results from `process_modal_data`. See method for more info
        """
        if modal_type == 'cog':
            cols = ['RID', 'ADNI_MEM','ADNI_EF']
        elif modal_type == 'csf':
            cols = ['RID', 'LOGABETA','LOGTAU','LOGPTAU','LOGPTAU/ABETA','LOGTAU/ABETA']
        elif modal_type == 'mri':
            cols = ['RID', 'BL_ICV', 'BL_HippVol', 'BL_Thick_EntCtx']
        elif modal_type == 'demo':
            cols = ['RID', 'PTGENDER', 'PTEDUCAT', 'AGE']

        if cols is not None:
            return self.process_modal_data(modal_type, cols)



    def process_modal_data(self, modal_type, modal_cols):
        """
        Processes modal-specific data so that each RID has the same number of time points.
        @param modal_cols (list): modal-specific column names in ADNI data
        @return X (numpy): features in numpy array format
        @return y (numpy): labels in numpy array format
        """
        modal_df = self.df[modal_cols]
        max_num_timepoints = 5 if modal_type == 'cog' or modal_type == 'csf' else 1
        X = []

        for RID in self.RIDs:  #modal_df['RID'].unique():
            RID_df = modal_df[modal_df['RID'] == RID].fillna(0)
            
            if modal_type == 'cog' or modal_type == 'csf':
                sequences = np.array(RID_df[modal_cols[1:]].transpose().values)
                RID_data = pad_sequences(
                    sequences, 
                    maxlen=max_num_timepoints,
                    dtype='float',
                    padding='post',
                    truncating='post', 
                    value=0.0
                )
                RID_data = RID_data.transpose()
            else:
                RID_data = np.array(RID_df.iloc[0,:][modal_cols[1:]].tolist())
            
            X.append(RID_data)
        
        X = np.array(X)
        return X





    
