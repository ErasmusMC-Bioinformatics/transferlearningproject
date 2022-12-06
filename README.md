# Transfer learning project

This repository is the official implementation of Julian van Toledo's MSc Project. 

## Requirements

See imports in notebook

## Results

Our model achieves the following performance on :

### [Survival prediction on ICGC PDAC data]

Pre-train on: STAD,CHOL,SARC,TGCT,COAD (TCGA: https://portal.gdc.cancer.gov/)

Predict survival on: Pancreatic Ductal Adenocarcinoma (https://dcc.icgc.org/)

| Model name         | C-index  |
| ------------------ |---------------- |
| Non-transfer learning Model  |     0.655202       |
| Transfer learning Model  |    0.6782734        |

