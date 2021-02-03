#keep only MCI patients
#patients that converted to AD = 1, rest of patients = 0
#save timepoint when patients converted to AD
import pandas as pd
import os


if __name__ == "__main__":
    cwd = os.getcwd()
    adni_file = os.path.join(cwd, "data", "ADNI_All_Longitudinal_Diagnosis_Data.csv")

    df = pd.read_csv(adni_file, sep=',')
    cp_df = pd.DataFrame(columns=[
        'CP.VISCODE', 
        'DX.at.baseline', 
        'RID', 
        'Y']
    )

    mci_statuses= ['EMCI', 'LMCI']
    for rid in df['RID'].unique():
        sample_df = df.loc[df['RID'] == rid]

        if sample_df['DX at baseline'].iloc[0] in mci_statuses:
            cp = sample_df.loc[sample_df['DXCHANGE'] == 'Conversion: MCI to Dementia']
            
            if cp.empty:
                # MCI non-converter
                new_row = {
                    'CP.VISCODE': sample_df['VISCODE'].iloc[-1],
                    'DX.at.baseline': sample_df['DX at baseline'].iloc[0],
                    'RID': rid,
                    'Y': 0.0
                }
                cp_df = cp_df.append(new_row, ignore_index=True)
            else:
                # MCI converter
                new_row = {
                    'CP.VISCODE': cp['VISCODE'].iloc[0],
                    'DX.at.baseline': cp['DX at baseline'].iloc[0],
                    'RID': rid,
                    'Y': 1.0
                }
                cp_df = cp_df.append(new_row, ignore_index=True)
        else:
            continue

    cp_df.to_csv("conversion_point_suh.csv", sep=",", index=False)

    # merge processed file with clinical data
    clinical_file = os.path.join(cwd, "data", "ADNI_All_Clinical_Data_BL.csv")
    clinical_df = pd.read_csv(clinical_file, sep=",")
    
    total_df = cp_df.merge(
        clinical_df, 
        left_on='RID', 
        right_on='RID'
    )

    total_df.to_csv("total_data.tsv", sep="\t", index=False)
