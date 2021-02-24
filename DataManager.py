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


class DataManager(object):
    def __init__(self):
        self.labels = pd.read_csv(
            os.path.join(os.getcwd(), 'data', 'total_data.tsv'),
            sep='\t'
        )
        self.df = pd.read_csv(
            os.path.join(os.getcwd(), 'data', 'adni_all_garam.csv'),
            sep=','
        )

        # only interested in RIDs with EMCI or LMCI status
        index_names = self.df[~(self.df['DX.at.baseline'].isin(['EMCI', 'LMCI']))].index
        self.df.drop(index_names, inplace=True)


    def get_modal_data(self, modal):
        """
        Specifies columns from ADNI data that are associated with the modal type. 
        @param modal (string): 'cog', 'csf', or 'mri'
        @return (numpy): Results from `process_modal_data`. See method for more info
        """
        if modal == 'cog':
            cols = ['RID', 'ADNI_MEM','ADNI_EF']
        elif modal == 'csf':
            cols = ['RID', 'LOGABETA','LOGTAU','LOGPTAU','LOGPTAU/ABETA','LOGTAU/ABETA']
        elif modal == 'mri':
            cols = ['RID', 'BL_ICV', 'BL_HippVol', 'BL_Thick_EntCtx']

        if cols is not None:
            return self.process_modal_data(cols)


    def process_modal_data(self, modal_cols):
        """
        Processes modal-specific data so that each RID has the same number of time points.
        @param modal_cols (list): modal-specific column names in ADNI data
        @return X (numpy): features in numpy array format
        @return y (numpy): labels in numpy array format
        """
        modal_df = self.df[modal_cols]
        max_num_timepoints = 10  # keep number of time points same across all RIDs
        X, y = [], []

        for RID in modal_df['RID'].unique():
            RID_df = modal_df[modal_df['RID'] == RID]

            RID_data = []
            for i in range(max_num_timepoints):
                if i < len(RID_df):
                    curr = np.array(RID_df.iloc[i,:][modal_cols[1:]].tolist())
                    curr[np.isnan(curr)] = 0
                else: 
                    curr = [.0 for j in range(len(modal_cols)-1)]
                RID_data.append(curr)
            X.append(RID_data)
            y.append(self.labels[self.labels['RID'] == RID].iloc[0]['Y'])
        
        X = np.array(X)
        y = np.array(y)
        return X, y





    
