# recreate_adni_paper

## ADNI data format
| Column      | Description |
| ----------- | ----------- |
| RID | Sample ID |
| VISCODE | Time from baseline |
| DX at baseline | Patient status at baseline. CN = normal, EMCI = early mild cognitive impairment, LMCI = late mild cognitive impairment, AD = Alzheimer's disease |
| DXCHANGE | Patient changing status. Conversion, stable, reversion. |



keep only MCI patients
patients that converted to AD = 1, rest of patients = 0
save timepoint when patients converted to AD