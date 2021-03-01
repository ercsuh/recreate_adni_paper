"""
Main script.
- [x] prefix data with zeros (pad_sequences)
- [x] mri is a dense layer
- [x] max_num_timepoints = 2/3 for cog, and 5 for csf
- [ ] normalize data, once with all, and another with only training data
"""
from MildInt import *
import matplotlib.pyplot as plt
import os


model = MildInt(['cog', 'csf', 'mri', 'demo'])

# for modal in model.X.keys():
#     print(f"{modal}: {model.X[modal].shape}")

# print(f"y: {model.y.shape}")


# csf_auc = model.run_single_model(csf_X, csf_y)
# mri_auc = model.run_single_model(mri_X, mri_y)
# cog_auc = model.run_single_model(cog_X, cog_y)

# print("----single modal AUCs----")
# print(f"COG: {cog_auc}")
# print(f"CSF: {csf_auc}")
# print(f"MRI: {mri_auc}")

pred_y, test_y = model.run_integrated_model("all")
metrics = model.evaluate_model(pred_y, test_y)
print("[INFO] printing results (normalizing all values)...")
for metric in metrics.keys():
    if metric != 'FPR' and metric != 'TPR':
        print(f"{metric}: {metrics[metric]}")


# model = MildInt(['cog', 'csf', 'mri', 'demo'])
# pred_y1, test_y1 = model.run_integrated_model("training_only")
# metrics1 = model.evaluate_model(pred_y1, test_y1)
# print("[INFO] printing results (normalizing training values only)...")
# for metric in metrics1.keys():
#     if metric != 'FPR' and metric != 'TPR':
#         print(f"{metric}: {metrics1[metric]}")


# plt.figure()
# lw = 2
# plt.plot(metrics['FPR'], metrics['TPR'], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % metrics['AUC'])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Integrated model - Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.savefig("figures/integrated_model_roc.jpg")