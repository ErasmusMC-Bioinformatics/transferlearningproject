# Transfer learning project

This repository is the official implementation of Julian van Toledo's MSc Project. 

## Requirements
Python 3.7.12 <br>
R version 4.2.0 <br>
Keras - 2.9.0 <br>

See imports in nnetsurvivaltransfer.ipynb for further dependencies

## Scripts
nnetsurvivaltransfer.ipynb - Jupyter Notebook containing the transfer learning model and various experiments that were performed <br>
preprocessingscript.R - Preprocessing of the TCGA data <br>
Preprocessing_icgc.R - Preprocessing of the ICGC data <br>
TensorBoardNotebook.ipynb - Notebook containing a simple script to start TensorBoard which allows users to view training logs of their model <br>
nnet_survival.py - Script of nnet_survival containing loss function, proportional hazards layer, and function to make survival matrices
cox_nnet_v2.py - Script containing the Cox-nnet model which is used for comparison to our model
