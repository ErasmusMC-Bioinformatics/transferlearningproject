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

def main():

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

    print("Preparing ICGC data for benchmark...")
    #
    # Data preperation ICGC 
    #

    xv = xv.to_numpy()

    yvtime = yv['time'].to_numpy()
    yvstatus = yv['donor_vital_status'].to_numpy()

    # Create a numpy array which contains the status and survival in days for each patient, this is used by the Cox-ph model as the labels

    ystatustime = []
    for status,time in zip(yvstatus,yvtime):
        ystatustime.append((status,time))

    ysurvival_data = np.array(ystatustime, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    # Check how many death events are in the data so you know how many death events you need per time interval

    number_of_events = (yv['donor_vital_status'] == 1).sum()
    number_of_nonevents = (yv['donor_vital_status'] == 0).sum()

    print(number_of_events)
    print(number_of_nonevents)

    # Here we set the breaks for our time intervals which we obtained with the plot above

    vbreaks = np.array([0,165,230,260,345,390,425,460,600,1100,1880])

    n_intervals = len(vbreaks) - 1

    # Create the survival array for the ICGC PDAC patients using the nnet_survival fuction 'make_surv_array'
    # This function takes the following input:
    #    ytime - numpy array of each patient's time until death/censor value
    #    ystatus - numpy array of each patient's survival status (1 for death event, 0 for alive/censored)
    #    vbreaks - the breaks for the time intervals that were found

    y_v = nnet_survival.make_surv_array(yvtime, yvstatus, vbreaks)

    # Create time quantiles to later be used in stratification

    first_quantile_v = np.quantile(yvtime,0.25)
    second_quantile_v = np.quantile(yvtime,0.5)
    third_quantile_v = np.quantile(yvtime,0.75)

    # For every patient's time value in yvtime check in what quantile it falls and make a new list that contains the corresponding time quantile for each patient

    yvquantile = []
    for time in yvtime:
        if time >= 0 and time < first_quantile_v:
            yvquantile.append(1)
        elif time >= first_quantile_v and time < second_quantile_v:
            yvquantile.append(2)
        elif time >= second_quantile_v and time < third_quantile_v:
            yvquantile.append(3)
        elif time >= third_quantile_v:
            yvquantile.append(4)

    # For every survival matrix, status value, and quantile value merge the status and quantile values together and merge with the survival matrix.

    yv_final = ([],[])

    for matrix,status,quantile in zip(y_v,yvstatus,yvquantile):
        yv_final[0].append(matrix)
        status_quantile = (str(int(status)) + "_" + str(quantile))
        yv_final[1].append(status_quantile)

    results_file = "Results/benchmark.txt"

    # Set optimal parameters found with hyperparameter tuning on TCGA PDAC data

    optimal_l2 = 0.00001
    optimal_batch = 8

    # Create a model to initialize the weights of the source model into (should have the same architecture)

    init_model = Sequential()
    init_model.add(Dense(np.sqrt(xv.shape[1]), input_dim=xv.shape[1], bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(0.00001)))
    init_model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
    init_model.add(nnet_survival.PropHazards(10))
    init_model.compile(loss=nnet_survival.surv_likelihood(10), optimizer=optimizers.Adam(learning_rate=0.000001))

    # Load the weights of the source model into the initialization model

    init_model.load_weights("Weights/source_finetune_model_subset")
    

    # Create Repeated Stratified Kfold cross validation object

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=24)

    k = 1

    train_performance_coxph = []
    test_performance_coxph = []

    train_performance_coxnnet = []
    test_performance_coxnnet = []

    train_performance_transfersnnet = []
    test_performance_transfersnnet = []

    print("Starting benchmark...")

    # Start the repeated 5 fold cross validation (will repeat 50 times, based on n_splits = 5 and n_repeats = 10)
    for train_indices, test_indices in rskf.split(xv,yv_final[1]):
        print("Loop: " + str(k) + "/50")
        
        # Create splits of the input data
        # Split the input features of the ICGC PDAC data
        x_train, x_test = xv[train_indices], xv[test_indices]
        # Split the input labels of the ICGC PDAC data (survival matrices)
        y_train, y_test = np.array(yv_final[0])[train_indices], np.array(yv_final[0])[test_indices]
        # Split the time values for each patient
        yvtime_train, yvtime_test = yvtime[train_indices], yvtime[test_indices]
        # Split the status values for each patient
        yvstatus_train, yvstatus_test = yvstatus[train_indices], yvstatus[test_indices]
        # Split the custom survival labels used by the cox-ph model
        ysurvival_data_train, ysurvival_data_test = ysurvival_data[train_indices], ysurvival_data[test_indices]

        
        # Scale the input train and test features using MinMaxScaler
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        ### Cox-PH ###
        
        # Create the Cox-PH elastic net model
        cox_elastic = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01)
        
        # Fit the model on the scaled train data and custom survival data
        cox_elastic.fit(x_train_scaled,ysurvival_data_train)
        
        # Calculate the C-index on the train data
        c_index_train = cox_elastic.score(x_train_scaled,ysurvival_data_train)
        # Calculate the C-index on the test data
        c_index_test = cox_elastic.score(x_test_scaled,ysurvival_data_test)
        
        # Add the C-index train score to train_performance_coxph
        train_performance_coxph.append(c_index_train)
        # Add the C-index test score to test_performance_coxph
        test_performance_coxph.append(c_index_test)
    
        ### Cox-nnet ###

        model_params = dict(node_map = None, input_split = None)
        search_params = dict(method="adam", learning_rate=0.01, momentum=0.9,
                                    max_iter=2000, stop_threshold=0.995, patience=1000, patience_incr=2, rand_seed=123,
                                    eval_step=23, lr_decay=0.9, lr_growth=1.0)
        cv_params = dict(cv_seed=1, n_folds=5, cv_metric="loglikelihood", L2_range=np.arange(-4.5, 1, 0.5))
        
        # Train model the model using the optimal L2 parameter found with the hyperparmaeter tuning
        L2_reg = -4.5
        model_params = dict(node_map = None, input_split = None, L2_reg=np.exp(L2_reg))
        cox_nnet_model, cox_nnet_cost_iter = cox_nnet.trainCoxMlp(x_train_scaled, yvtime_train,yvstatus_train, model_params, search_params, verbose=False)
        
        # Make a prediction on the training data
        cox_nnet_theta_train = cox_nnet_model.predictNewData(x_train_scaled)
        # Make a prediction on the testing data
        cox_nnet_theta_test = cox_nnet_model.predictNewData(x_test_scaled)

        # Calculate train C-index from the time data, predictions made by cox-nnet and the patient status data, store the results in train_performance_coxnnet
        train_performance_coxnnet.append(concordance_index(yvtime_train,-cox_nnet_theta_train,yvstatus_train))
        # Calculate test C-index from the time data, predictions made by cox-nnet and the patient status data, store the results in test_performance_coxnnet
        test_performance_coxnnet.append(concordance_index(yvtime_test,-cox_nnet_theta_test,yvstatus_test))

        ### OUR MODEL ###
        
        # Create a model to run on the ICGC PDAC (target) data (this uses the source model's weights)
        model = Sequential()
        model.add(Dense(np.sqrt(xv.shape[1]), input_dim=xv.shape[1], bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(optimal_l2)))
        model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
        model.add(nnet_survival.PropHazards(10))
        model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=optimizers.Adam(learning_rate=0.000001))

        
        # Transfer weights of source model's second hidden layer to fine tuning model's second hidden layer
        model.layers[1].set_weights(init_model.layers[1].get_weights())

        # Create a tensorboard callback which saves the loss during the training of the model
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=("ktuner/logs_comparison_new/fold_" + str(k)), histogram_freq=0, embeddings_freq=0, write_graph=False, update_freq='batch')
        callbacks_list = [tensorboard_callback]

        # Fit the model to the input data x_train_scaled and y_train
        history = model.fit(x_train_scaled,y_train,batch_size=optimal_batch,epochs=2600, callbacks=callbacks_list,verbose=0)

        # Make predictions on the train data for the training performance
        y_pred = nnet_survival.nnet_pred_surv(model.predict(x_train_scaled,verbose=0), vbreaks, 365)
        # Calculate C-index for train data using time, predictions and status values
        c_index_train = concordance_index(yvtime_train, y_pred, yvstatus_train)
        
        # Make predictions on the test data for test performance
        y_pred = nnet_survival.nnet_pred_surv(model.predict(x_test_scaled,verbose=0), vbreaks, 365)
        # Calculate C-index for test data using time, predictions and status values
        c_index_test = concordance_index(yvtime_test, y_pred, yvstatus_test)

        # Save the train C-index of our model to train_performance_transfersnnet
        train_performance_transfersnnet.append(c_index_train)
        # Save the test C-index of our model to test_performance_transfersnnet
        test_performance_transfersnnet.append(c_index_test)
        
        # Save the average C-index  and total C-index list each approach every fold
        with open(results_file, 'a') as o:
            print("C-indexes, Cox-PH:\n",file=o)
            print("Train:\n",file=o)
            print("Average C-index: " + str(sum(train_performance_coxph)/k),file=o)
            print(str(train_performance_coxph),file=o)
            print("Test:\n",file=o)
            print("Average C-index: " + str(sum(test_performance_coxph)/k),file=o)
            print(str(test_performance_coxph),file=o)
            print("\n",file=o)
            print("C-indexes, Cox-nnet:\n",file=o)
            print("Train:\n",file=o)
            print("Average C-index: " + str(sum(train_performance_coxnnet)/k),file=o)
            print(str(train_performance_coxnnet),file=o)
            print("Test:\n",file=o)
            print("Average C-index: " + str(sum(test_performance_coxnnet)/k),file=o)
            print(str(test_performance_coxnnet),file=o)
            print("\n",file=o)
            print("C-indexes, our model:\n",file=o)
            print("Train:\n",file=o)
            print("Average C-index: " + str(sum(train_performance_transfersnnet)/k),file=o)
            print(str(train_performance_transfersnnet),file=o)
            print("Test:\n",file=o)
            print("Average C-index: " + str(sum(test_performance_transfersnnet)/k),file=o)
            print(str(test_performance_transfersnnet),file=o)
            print("\n",file=o)
        
        o.close()
        
        k += 1
    print("Benchmark finished! Check Results/ folder")


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

    main(mrna_full,mrna_subset,mrna_paad,mrna_icgc,clinical_full,clinical_subset,clinical_paad,clinical_icgc)