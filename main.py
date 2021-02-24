from DataManager import *
from MildInt import *
import os


df = pd.read_csv(os.path.join(os.getcwd(), "data", "total_data.tsv"), sep="\t")

df.PTGENDER = pd.Categorical(df.PTGENDER)
df.PTGENDER = df.PTGENDER.cat.codes
demo_X = np.array(df[['PTGENDER', 'PTEDUCAT', 'AGE']].values.tolist())
demo_y = np.array(df['Y'].values.tolist())

dm = DataManager()
cog_X, cog_y = dm.get_modal_data('cog')
csf_X, csf_y = dm.get_modal_data('csf')
mri_X, mri_y = dm.get_modal_data('mri')

model = MildInt()
# csf_auc = model.run_single_model(csf_X, csf_y)
# mri_auc = model.run_single_model(mri_X, mri_y)
# cog_auc = model.run_single_model(cog_X, cog_y)

# print("----single modal AUCs----")
# print(f"COG: {cog_auc}")
# print(f"CSF: {csf_auc}")
# print(f"MRI: {mri_auc}")

model.run_integrated_model(demo_X, demo_y, cog_X, cog_y, csf_X, csf_y, mri_X, mri_y)