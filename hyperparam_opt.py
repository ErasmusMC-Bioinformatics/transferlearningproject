import argparse

import os
import keras.optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow.keras.backend as K
import keras_tuner
import nnet_survival
import keras_tuner as kt
import tensorflow as tf
import importlib
# import lasagne
import nnet_survival
# import cox_nnet_v2 as cox_nnet

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from matplotlib.pyplot import figure
from tensorflow import keras
# from kerastuner_tensorboard_logger import (
#     TensorBoardLogger,
#     setup_tb  # Optional
# )
from matplotlib.pyplot import figure
from sklearn.model_selection import GridSearchCV
# from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import load_model
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored,brier_score
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sksurv.util import Surv
from keras_tuner import HyperModel
from tensorflow.keras.metrics import Metric

class SurvivalHyperModel(HyperModel):
    def __init__(self, n_intervals, weights=[]):
        self.n_intervals = n_intervals
        self.weights = weights
    def build(self, hp):
        model = Sequential()
        hp_l2_value = hp.Choice('l2_value', values=[0.01,0.001,0.0001,0.00001])
        model.add(Dense(np.sqrt(xf.shape[1]), input_dim=xf.shape[1], bias_initializer='zeros',activation='relu', kernel_regularizer=regularizers.l2(hp_l2_value)))
        model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
        model.add(nnet_survival.PropHazards(self.n_intervals))
        model.compile(loss=nnet_survival.surv_likelihood(self.n_intervals), optimizer=optimizers.Adam(learning_rate=0.00001),run_eagerly=False)
        if self.weights:
            model.layers[0].set_weights(self.weights)
            return model
        else:
            return model
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args,
                         shuffle=False,
                         **kwargs,
                        )
    
class CVTuner(keras_tuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y,epochs=1, *args, **kwargs):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=24)
        val_losses = []
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=[8,16,32])
        fold = 1
        hp = trial.hyperparameters
        for train_indices, test_indices in rskf.split(x,y[1]):
            early_stopping_hp = EarlyStopping(monitor='val_loss', patience=(100),min_delta=0.0005, restore_best_weights=True)
            dir_for_logs = ("Logs/ktuner/logs_pdac_21/" + str(trial.trial_id) + "/fold_" + str(fold))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=dir_for_logs, histogram_freq=0, embeddings_freq=0, write_graph=False, update_freq='batch')
            callbacks_list = [early_stopping_hp,tensorboard_callback]
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = np.array(y[0])[train_indices], np.array(y[0])[test_indices]
            y_train_labels, y_test_labels = np.array(y[1])[train_indices], np.array(y[1])[test_indices]
            model = self.hypermodel.build(hp)
            model.fit(x_train, y_train,validation_data=(x_test,y_test), batch_size=kwargs['batch_size'], epochs=epochs, callbacks=callbacks_list, verbose=0)       
            y_pred = model.predict(x_test, verbose=0)
            surv_prob=np.cumprod(y_pred, axis=1)[:,-1]
            val_losses.append(model.evaluate(x_test, y_test))
            fold += 1
        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})


def main(mrna_full,mrna_subset,mrna_paad,mrna_icgc,clinical_full,clinical_subset,clinical_paad,clinical_icgc,logfile_path,project_name,directory_name):

    print("Loading in files...")
    x_full = pd.read_csv(mrna_full, sep='\t')
    y_full = pd.read_csv(clinical_full, sep=',')
    x_subset = pd.read_csv(mrna_subset, sep='\t')
    y_subset = pd.read_csv(clinical_subset, sep=',')
    x_paad = pd.read_csv(mrna_paad, sep='\t')
    y_paad = pd.read_csv(clinical_paad, sep=',')
    x_icgc = pd.read_csv(mrna_icgc, sep='\t')
    y_icgc = pd.read_csv(clinical_icgc, sep=',')
    print("Loaded files!")

    print("Making sure all datasets are ordered the same, and have the same number of columns/rows...")
    columns = x_paad.columns.str.split(' / ')
    correct_columns = []
    for column in columns:
        correct_columns.append(column[0])

    x_paad.columns = correct_columns

    columns = x_full.columns.str.split(' / ')
    correct_columns = []
    for column in columns:
        correct_columns.append(column[0])

    x_full.columns = correct_columns

    columns = x_subset.columns.str.split(' / ')
    correct_columns = []
    for column in columns:
        correct_columns.append(column[0])

    x_subset.columns = correct_columns

    # Drop the genes in the TCGA PDAC data that are not in the full 32 cancer dataset
    print("Dropping genes not common between datasets...")
    labels_to_drop = x_paad.columns.difference(x_full.columns)
    x_paad = x_paad.drop(labels=labels_to_drop,axis=1)

    x_paad = x_paad.loc[:,~x_paad.columns.duplicated()].copy()
    x_full = x_full.loc[:,~x_full.columns.duplicated()].copy()
    x_subset = x_subset.loc[:,~x_subset.columns.duplicated()].copy()

    x_paad = x_paad.drop(labels=(x_paad.columns.difference(x_full.columns)),axis=1)
    x_icgc = x_icgc.drop(labels=(x_icgc.columns.difference(x_full.columns)),axis=1)
    x_full = x_full.drop(labels=(x_full.columns.difference(x_paad.columns)),axis=1)
    x_subset = x_subset.drop(labels=(x_subset.columns.difference(x_paad.columns)),axis=1)

    x_paad = x_paad.drop(labels=(x_paad.columns.difference(x_icgc.columns)),axis=1)
    x_full = x_full.drop(labels=(x_full.columns.difference(x_icgc.columns)),axis=1)
    x_subset = x_subset.drop(labels=(x_subset.columns.difference(x_icgc.columns)),axis=1)

    x_paad = x_paad.drop(labels=(x_paad.columns.difference(x_icgc.columns)),axis=1)
    x_full = x_full.drop(labels=(x_full.columns.difference(x_icgc.columns)),axis=1)
    x_subset = x_subset.drop(labels=(x_subset.columns.difference(x_icgc.columns)),axis=1)

    #print(x_paad.shape)
    #print(x_full.shape)
    #print(x_subset.shape)
    #print(x_icgc.shape)

    x_icgc = x_icgc[xt.columns]
    x_full = x_full[xt.columns]
    x_subset = x_subset[xt.columns]

    print("Datasets are all equal now")

    print("Preparing full tcga dataset for hyperparameter optimization...")

    xf = x_full.to_numpy()
    yf = y_full

    ytime = yf['time'].to_numpy()
    ystatus = yf['vital_status'].to_numpy()
    ystatusbool = yf['vital_status'].astype(bool)

    # Create time quantiles to later be used in stratification
    print("Creating time quantiles...")
    first_quantile = np.quantile(ytime,0.25)
    second_quantile = np.quantile(ytime,0.5)
    third_quantile = np.quantile(ytime,0.75)

    yquantile = []
    for time in ytime:
        if time >= 0 and time < first_quantile:
            yquantile.append(1)
        elif time >= first_quantile and time < second_quantile:
            yquantile.append(2)
        elif time >= second_quantile and time < third_quantile:
            yquantile.append(3)
        elif time >= third_quantile:
            yquantile.append(4)

    number_of_events = (yf.vital_status == 1).sum()
    number_of_nonevents = (yf.vital_status == 0).sum()

    print("Events:", number_of_events)
    print("Non Events:", number_of_nonevents)

    breaks=np.asarray([0,55,100,135,180,240,260,322,370,430,455,520,580,620,730,840,970,1140,1350,1590,1900,2540,9200])
    n_intervals = len(breaks)-1

    # Create the survival array using the nnet_survival fuction 'make_surv_array'
    # This function takes the following input:
    #    ytime - numpy array of each patient's time until death/censor value
    #    ystatus - numpy array of each patient's survival status (1 for death event, 0 for alive/censored)
    #    breaks - the breaks for the time intervals that were found

    yf = nnet_survival.make_surv_array(ytime, ystatus, breaks)

    # For every survival matrix, status value, and quantile value merge the status and quantile values together and merge with the survival matrix.
    yf_final = ([],[])

    for matrix,status,quantile in zip(yf,ystatus,yquantile):
        yf_final[0].append(matrix)
        status_quantile = (str(int(status)) + "_" + str(quantile))
        yf_final[1].append(status_quantile)
    
    print("Dataset prepared!")

    print("Starting hyperparameter optimization...")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Scale the input features using MinMaxScaler

    scaler = MinMaxScaler()
    xf_t = scaler.fit_transform(xf)

    # logfile - Filename in which the results will be stored of the hyperparameter tuning.
    #           Call this file with Tensorboard to view loss curves and hyperparameter tuning results.
    #
    # name_of_project - Name of the Keras Tuner project.
    #                   Make sure this is different if you want to try a new search, else it will 
    #                   just give you the results of the project previous project that had the same name !!!!
    #
    # name_of_dir - Directory where projects will be stored.

    logfile = logfile_path
    name_of_project = project_name
    name_of_dir = "Hyperparameter_opt_projects/"

    # The Cross validation tuner class, takes as input:
    #    hypermodel - A hypermodel class, which is your model but contains hyperparameter values for every parameter you want to test
    #    directory - Directory where the hyper parameter tuning project will be stored
    #    logger - Takes a TensorBoardLogger value as input which will store the loss of the model during training steps, 
    #             which can be used in TensorBoard to create plots
    #    oracle - The keras tuner optimization method that you want to use, refer to: https://keras.io/api/keras_tuner/oracles/ for possible oracles.
    #             takes as input:
    #             objective - The objective of the hyperparameter tuning, and what determines whether a set of parameters is 'good'
    #             max_trials - The number of trials you wish to perform, each trial will contain a set of parameters to test.
    #
    # After creating the class, use .search to initiate the hyperparameter tuning, takes as input:
    #    xf_t - the scaled input features to train on
    #    yf_final - the label on which predictions will be assessed, contains the survival matrix and status_quantile for a patient
    #    epochs - the maximum number of epochs you wish to train (Early stopping is included in the class, so value is determined automatically)
    #    verbose - 1 to see progress of hyperparameter tuning in console, 0 to not see progress.
                                                    
    tuner = CVTuner(
        hypermodel=SurvivalHyperModel(n_intervals),
        project_name=name_of_project,
        directory=name_of_dir,
        logger = TensorBoardLogger(metrics=["val_loss"], logdir=logfile),
        oracle=kt.oracles.BayesianOptimization(
            objective='val_loss',
            max_trials=10))
    tuner.search(xf_t,yf_final,epochs=40000,verbose=1)

    print("Hyperparameter optimization finished! Check the log file for results")


if __name__ == "__main__":
    
    #Set inputs, depends on if called from python script or command line, dont forget to point towards the /processed/ folder in the respective folders
    #Give the paths here to the respective files
    mrna_full = 
    mrna_subset = 
    mrna_paad =
    mrna_icgc = 
    
    clinical_full = 
    clinical_subset = 
    clinical_paad =
    clinical_icgc = 

    # Paths for files and directories created during hyperparameter optimization
    # logfile_path - The path where the log file will be created, the log file contains all the results of the hyperparameter optimization and can be opened with TensorBoard to view the results visually
    # project_name - The name of the current hyperparameter optimization project, change this everytime you want to do a new run, else it will just use the same files ending instantly and giving you the same results
    logfile_path = 
    project_name =     

    main(mrna_full,mrna_subset,mrna_paad,mrna_icgc,clinical_full,clinical_subset,clinical_paad,clinical_icgc,logfile_path,project_name)