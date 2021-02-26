"""
Main script.
"""
from MildInt import *
import matplotlib.pyplot as plt
import os


model = MildInt(['cog', 'csf', 'mri', 'demo'])

for modal in model.X.keys():
    print(f"{modal}: {model.X[modal].shape}")

print(f"y: {model.y.shape}")


# csf_auc = model.run_single_model(csf_X, csf_y)
# mri_auc = model.run_single_model(mri_X, mri_y)
# cog_auc = model.run_single_model(cog_X, cog_y)

# print("----single modal AUCs----")
# print(f"COG: {cog_auc}")
# print(f"CSF: {csf_auc}")
# print(f"MRI: {mri_auc}")

pred_y, test_y = model.run_integrated_model()
metrics = model.evaluate_model(pred_y, test_y)
print("[INFO] printing results...")
print(f"AUC: {metrics['AUC']}")
print(f"ACC: {metrics['ACC']}")
print(f"SEN: {metrics['SEN']}")
print(f"SPE: {metrics['SPE']}")


plt.figure()
lw = 2
plt.plot(metrics['FPR'], metrics['TPR'], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % metrics['AUC'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Integrated model - Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig("figures/integrated_model_roc.jpg")