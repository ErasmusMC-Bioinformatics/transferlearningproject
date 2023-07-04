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
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        skf.get_n_splits(x,y[1])
        val_losses = []
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=[8,16,32])
        fold = 1
        hp = trial.hyperparameters
        for train_indices, test_indices in skf.split(x,y[1]):
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



def main(approach):
    #
    # Import multi-cancer Data
    #
    print("Importing multi-cancer mrna data", end='')
    mrna_list = []
    mrna_folder = "Data/mrnaDataUnscaled"
    clinical_folder = "Data/ClinicalData"
    for filename in sorted(os.listdir(mrna_folder)):
        if not filename.endswith(".csv"):
            continue
        f = os.path.join(mrna_folder, filename)
        data = pd.read_csv(f, sep=",")
        mrna_list.append(data)
        print(".", end='')
    print(" Done!")
    print(f"Imported from {len(mrna_list)} files")
        
    print("Importing multi-cancer clinical data", end='')
    clinical_list = []
    for filename in sorted(os.listdir(clinical_folder)):
        if not filename.endswith(".txt"):
            continue
        f = os.path.join(clinical_folder, filename)
        data = pd.read_csv(f, sep="\t")
        clinical_list.append(data)
        print(".", end='')
    print(" Done!")
    print(f"Imported from {len(clinical_list)} files")

    """
    Prepare multi-cancer data for tensorflow

    Every data set is handled individually, and merged in the end.

        First The clinical samples are matched to the genomic samples.
        Then samples are filtered out of the clinical data that are not in the mRNA data
        After that all Not availables and Discrepancys are replaced with NaN.
        Vital status Dead and Alive are set to 1 and 0 respectively.
        Days are made numerical.
        Remove patients that have time 0 or NaN for time/status.
        Set Index of mRNA and clinical to patient barcode, and transpose the mRNA data so that the patients are in each row, and the genes in each column.
    """

    x_list = []
    clinical_processed_list = []

    # Disable a warning we don't care about
    warn = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None  # default='warn'

    print("Preprocessing mrna/clinical data", end='')
    for i, mrna_clinical in enumerate(zip(mrna_list,clinical_list)):
        mrna, clinical = mrna_clinical
        mrna_id = mrna.columns[1:]

        # Match clinical samples to genomic samples
        clinical.columns = clinical.iloc[0]
        clinical = clinical.iloc[2:]

        # Make intersection of patient id's that are in mrna and clinicaldata
        clinical_id = clinical['bcr_patient_barcode']
        intersection = list(set(mrna_id) & set(clinical_id))
        intersection.sort()
        intersection = pd.Series(intersection)


        # Filter out samples in clinicaldata that are not in mrna
        a = clinical['bcr_patient_barcode'].isin(intersection)
        clinical = clinical[a]



        # Create clinicaldata dataframe with the important features
        clinicalnew = clinical[['bcr_patient_barcode',
                                    "vital_status",
                                    "days_to_last_followup",
                                    "days_to_death"]]
        
        #print(clinicalnew['vital_status'].value_counts())
        

        # Set missing data to NaN
        clinicalnew['vital_status'] = clinicalnew['vital_status'].replace("[Discrepancy]", 0)
        clinicalnew['vital_status'] = clinicalnew['vital_status'].replace("[Not Available]",np.nan)
        clinicalnew["days_to_death"] = clinicalnew['days_to_death'].replace("[Not Applicable]", np.nan)
        clinicalnew["days_to_death"] = clinicalnew['days_to_death'].replace("[Not Available]", np.nan)
        clinicalnew["days_to_death"] = clinicalnew['days_to_death'].replace("[Discrepancy]", np.nan)
        clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Not Available]", np.nan)
        clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Discrepancy]", np.nan)
        clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Discrepancy]", np.nan)
        clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Completed]", np.nan)

        # In vital_status set dead = 1 alive = 0
        clinicalnew['vital_status'] = clinicalnew['vital_status'].replace("Dead", 1)
        clinicalnew["vital_status"] = clinicalnew['vital_status'].replace("Alive", 0)

        # Set days to numeric values
        clinicalnew["days_to_last_followup"] = pd.to_numeric(clinicalnew["days_to_last_followup"])
        clinicalnew["days_to_death"] = pd.to_numeric(clinicalnew["days_to_death"])

        # Combine days to death and days to last follow up to create a total time.
        clinicalnew['time'] = clinicalnew['days_to_death'].combine_first(clinicalnew['days_to_last_followup'])

        # Remove patients that have time 0 (so no follow up, just one recording)
        clinicalnew = clinicalnew[clinicalnew.time != 0]

        #Remove patients with nan for time or status
        clinicalnew = clinicalnew.dropna(subset=['time'])
        clinicalnew = clinicalnew.dropna(subset=['vital_status'])
        
        #Remove patients where time is negative
        clinicalnew = clinicalnew[clinicalnew.time >= 0]

        mrna.rename(columns={'Unnamed: 0': 'bcr_patient_barcode'}, inplace=True)
        mrna = mrna.set_index('bcr_patient_barcode')

        mrna = mrna.transpose()
        mrna = mrna[mrna.index.isin(clinicalnew['bcr_patient_barcode'])]
        mrna = mrna.reindex(np.random.RandomState(seed=1).permutation(mrna.index))
        
        clinicalnew = clinicalnew.set_index('bcr_patient_barcode')
        clinicalnew = clinicalnew.loc[~clinicalnew.index.duplicated(), :]
        clinicalnew = clinicalnew.reindex(index=mrna.index)
        
        x_list.append(mrna)
        clinical_processed_list.append(clinicalnew)

        print(".", end='')

    print(" Done!")

    pd.options.mode.chained_assignment = warn


    # Concatenate the list of processed RNA-seq pandas dataframes together to one large dataset
    # Drop the NA values
    # Concatenate the list of processed Clinical pandas dataframes together to one large dataset
    print("Concatenating datasets...")
    xf_full = pd.concat(x_list)
    xf_full = xf_full.dropna(axis=1)
    clinicalf_full = pd.concat(clinical_processed_list)

    #
    # Import subset of cancers
    #
    print("Importing subset mrna data", end='')

    mrna_list = []
    mrna_folder = "Data/subsetMrna"
    clinical_folder = "Data/subsetClinical"
    for filename in sorted(os.listdir(mrna_folder)):
        f = os.path.join(mrna_folder, filename)
        data = pd.read_csv(f, sep=",")
        mrna_list.append(data)
        print(".", end='')

    print(" Done!")
    print(f"Imported from {len(mrna_list)} files")
        
    print("Importing subset clinical data", end='')
    clinical_list = []
    for filename in sorted(os.listdir(clinical_folder)):
        f = os.path.join(clinical_folder, filename)
        data = pd.read_csv(f, sep="\t")
        clinical_list.append(data)
        print(".", end='')

    print(" Done!")
    print(f"Imported from {len(clinical_list)} files")

    #
    # Prepare subset of cancer data for tensorflow 
    #
    x_list = []
    clinical_processed_list = []

    # Disable a warning we don't care about
    warn = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None  # default='warn'

    print("Preprocessing mrna/clinical subset", end='')

    for mrna,clinical in zip(mrna_list,clinical_list):
        mrna_id = mrna.columns[1:]

        # Match clinical samples to genomic samples
        clinical.columns = clinical.iloc[0]
        clinical = clinical.iloc[2:]

        # Make intersection of patient id's that are in mrna and clinicaldata
        clinical_id = clinical['bcr_patient_barcode']
        intersection = list(set(mrna_id) & set(clinical_id))
        intersection.sort()
        intersection = pd.Series(intersection)


        # Filter out samples in clinicaldata that are not in mrna
        a = clinical['bcr_patient_barcode'].isin(intersection)
        clinical = clinical[a]



        # Create clinicaldata dataframe with the important features
        clinicalnew = clinical[['bcr_patient_barcode',
                                    "vital_status",
                                    "days_to_last_followup",
                                    "days_to_death"]]
        
        #print(clinicalnew['vital_status'].value_counts())
        

        # Set missing data to NaN
        clinicalnew['vital_status'] = clinicalnew['vital_status'].replace("[Discrepancy]", 0)
        clinicalnew['vital_status'] = clinicalnew['vital_status'].replace("[Not Available]",np.nan)
        clinicalnew["days_to_death"] = clinicalnew['days_to_death'].replace("[Not Applicable]", np.nan)
        clinicalnew["days_to_death"] = clinicalnew['days_to_death'].replace("[Not Available]", np.nan)
        clinicalnew["days_to_death"] = clinicalnew['days_to_death'].replace("[Discrepancy]", np.nan)
        clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Not Available]", np.nan)
        clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Discrepancy]", np.nan)
        clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Discrepancy]", np.nan)
        clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Completed]", np.nan)

        # In vital_status set dead = 1 alive = 0
        clinicalnew['vital_status'] = clinicalnew['vital_status'].replace("Dead", 1)
        clinicalnew["vital_status"] = clinicalnew['vital_status'].replace("Alive", 0)

        # Set days to numeric values
        clinicalnew["days_to_last_followup"] = pd.to_numeric(clinicalnew["days_to_last_followup"])
        clinicalnew["days_to_death"] = pd.to_numeric(clinicalnew["days_to_death"])

        # Combine days to death and days to last follow up to create a total time.
        clinicalnew['time'] = clinicalnew['days_to_death'].combine_first(clinicalnew['days_to_last_followup'])

        # Remove patients that have time 0 (so no follow up, just one recording)
        clinicalnew = clinicalnew[clinicalnew.time != 0]

        #Remove patients with nan for time or status
        clinicalnew = clinicalnew.dropna(subset=['time'])
        clinicalnew = clinicalnew.dropna(subset=['vital_status'])
        
        #Remove patients where time is negative
        clinicalnew = clinicalnew[clinicalnew.time >= 0]

        mrna.rename(columns={'Unnamed: 0': 'bcr_patient_barcode'}, inplace=True)
        mrna = mrna.set_index('bcr_patient_barcode')

        mrna = mrna.transpose()
        mrna = mrna[mrna.index.isin(clinicalnew['bcr_patient_barcode'])]
        mrna = mrna.reindex(np.random.RandomState(seed=1).permutation(mrna.index))
        
        clinicalnew = clinicalnew.set_index('bcr_patient_barcode')
        clinicalnew = clinicalnew.loc[~clinicalnew.index.duplicated(), :]
        clinicalnew = clinicalnew.reindex(index=mrna.index)
        
        x_list.append(mrna)
        clinical_processed_list.append(clinicalnew)

        print(".", end='')

    print(" Done!")
        
    pd.options.mode.chained_assignment = warn

    # Concatenate the list of processed RNA-seq pandas dataframes together to one large dataset
    # Drop the NA values
    # Concatenate the list of processed Clinical pandas dataframes together to one large dataset

    print("Concatenating datasets...")
    xf_sub = pd.concat(x_list)
    xf_sub = xf_sub.dropna(axis=1)
    clinicalf_sub = pd.concat(clinical_processed_list)

    #
    # TCGA PDAC data preparation
    #
    print("Reading TCGA PDAC data...")
    clinicaldata = pd.read_csv(f"Data/TargetDataUnscaled/nationwidechildrens.org_clinical_patient_paad.txt", sep='\t')
    mrna = pd.read_csv(f"Data/TargetDataUnscaled/PAAD.csv", sep=',')

    # Disable a warning we don't care about
    warn = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None  # default='warn'

    mrna_id = mrna.columns[1:].tolist()

    # Match clinical samples to genomic samples
    clinicaldata.columns = clinicaldata.iloc[0]
    clinicaldata = clinicaldata.iloc[2:]


    # Make intersection of patient id's that are in mrna and clinicaldata
    clinical_id = clinicaldata['bcr_patient_barcode']
    intersection = list(set(mrna_id) & set(clinical_id))
    intersection.sort()
    intersection = pd.Series(intersection)


    # Filter out samples in clinicaldata that are not in mrna
    a = clinicaldata['bcr_patient_barcode'].isin(intersection)
    clinicaldata = clinicaldata[a]


    # Create clinicaldata dataframe with the important features
    clinicalnew = clinicaldata[['bcr_patient_barcode',
                                "vital_status",
                                "days_to_last_followup",
                                "days_to_death"]]



    # Set missing data to NaN
    clinicalnew['vital_status'] = clinicalnew['vital_status'].replace("[Discrepancy]", 0)
    clinicalnew["days_to_death"] = clinicalnew['days_to_death'].replace("[Not Applicable]", np.nan)
    clinicalnew["days_to_death"] = clinicalnew['days_to_death'].replace("[Not Available]", np.nan)
    clinicalnew["days_to_death"] = clinicalnew['days_to_death'].replace("[Discrepancy]", np.nan)
    clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Not Available]", np.nan)
    clinicalnew['days_to_last_followup'] = clinicalnew['days_to_last_followup'].replace("[Discrepancy]", np.nan)

    # In vital_status set dead = 1 alive = 0
    clinicalnew['vital_status'] = clinicalnew['vital_status'].replace("Dead", 1)
    clinicalnew["vital_status"] = clinicalnew['vital_status'].replace("Alive", 0)

    # Set days to numeric values
    clinicalnew["days_to_last_followup"] = pd.to_numeric(clinicalnew["days_to_last_followup"])
    clinicalnew["days_to_death"] = pd.to_numeric(clinicalnew["days_to_death"])

    # Combine days to death and days to last follow up to create a total time.
    clinicalnew['time'] = clinicalnew['days_to_death'].combine_first(clinicalnew['days_to_last_followup'])

    # Remove patients that have time 0 (so no follow up, just one recording)
    clinicalnew = clinicalnew[clinicalnew.time != 0]

    #Remove patients with nan for time
    clinicalnew = clinicalnew.dropna(subset=['time'])

    mrna.rename(columns={'Unnamed: 0': 'bcr_patient_barcode'}, inplace=True)
    mrna = mrna.set_index('bcr_patient_barcode')

    mrna = mrna.transpose()
    mrna = mrna[mrna.index.isin(clinicalnew['bcr_patient_barcode'])]
    mrna = mrna.reindex(np.random.RandomState(seed=1).permutation(mrna.index))
    clinicalnew = clinicalnew.set_index('bcr_patient_barcode')
    clinicalnew = clinicalnew.reindex(index=mrna.index)

    pd.options.mode.chained_assignment = warn

    xt = mrna
    yt = clinicalnew

    #
    # TCGA ICGC Data
    #
    print("Reading TCGA ICGC data...")
    clinicaldata = pd.read_csv(f"Data/ICGCDataUnscaled/PDAC_ICGC_clinical.csv", sep=',')
    mrna = pd.read_csv(f"Data/ICGCDataUnscaled/PDAC_ICGC.csv", sep=',')

    # Disable a warning we don't care about
    warn = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None  # default='warn'

    # Create clinicaldata dataframe with the important features
    clinicalnew = clinicaldata[['icgc_donor_id',
                                "donor_vital_status",
                                "donor_survival_time"]]



    # Set missing data to NaN
    clinicalnew['donor_vital_status'] = clinicalnew['donor_vital_status'].replace("", np.nan)

    # In vital_status set dead = 1 alive = 0
    clinicalnew['donor_vital_status'] = clinicalnew['donor_vital_status'].replace("deceased", 1)
    clinicalnew["donor_vital_status"] = clinicalnew['donor_vital_status'].replace("alive", 0)

    # Set days to numeric values
    clinicalnew["donor_survival_time"] = pd.to_numeric(clinicalnew["donor_survival_time"])

    # Combine days to death and days to last follow up to create a total time.
    clinicalnew['time'] = clinicalnew['donor_survival_time']

    # Remove patients that have time 0 (so no follow up, just one recording)
    clinicalnew = clinicalnew[clinicalnew.time != 0]

    #Remove patients with nan for time
    clinicalnew = clinicalnew.dropna(subset=['time'])

    mrna.rename(columns={'Unnamed: 0': 'icgc_donor_id'}, inplace=True)
    mrna = mrna.set_index('icgc_donor_id')

    clinicalnew = clinicalnew.set_index('icgc_donor_id')

    pd.options.mode.chained_assignment = warn

    xv = mrna
    yv = clinicalnew

    xv = xv.drop(['DO49201'])

    #
    # Match the genes in all 3 datasets to each other
    #
    #print(xt.shape)
    #print(xf_full.shape)
    #print(xf_sub.shape)
    #print(xv.shape)
    # print(xtip.shape)
    print("Do some column things?")
    columns = xt.columns.str.split(' / ')
    correct_columns = []
    for column in columns:
        correct_columns.append(column[0])

    xt.columns = correct_columns

    columns = xf_full.columns.str.split(' / ')
    correct_columns = []
    for column in columns:
        correct_columns.append(column[0])

    xf_full.columns = correct_columns

    columns = xf_sub.columns.str.split(' / ')
    correct_columns = []
    for column in columns:
        correct_columns.append(column[0])

    xf_sub.columns = correct_columns

    # Drop the genes in the TCGA PDAC data that are not in the full 32 cancer dataset
    print("Dropping genes not common between datasets...")
    labels_to_drop = xt.columns.difference(xf_full.columns)
    xt = xt.drop(labels=labels_to_drop,axis=1)

    xt = xt.loc[:,~xt.columns.duplicated()].copy()
    xf_full = xf_full.loc[:,~xf_full.columns.duplicated()].copy()
    xf_sub = xf_sub.loc[:,~xf_sub.columns.duplicated()].copy()

    xt = xt.drop(labels=(xt.columns.difference(xf_full.columns)),axis=1)
    xv = xv.drop(labels=(xv.columns.difference(xf_full.columns)),axis=1)
    xf_full = xf_full.drop(labels=(xf_full.columns.difference(xt.columns)),axis=1)
    xf_sub = xf_sub.drop(labels=(xf_sub.columns.difference(xt.columns)),axis=1)

    xt = xt.drop(labels=(xt.columns.difference(xv.columns)),axis=1)
    xf_full = xf_full.drop(labels=(xf_full.columns.difference(xv.columns)),axis=1)
    xf_sub = xf_sub.drop(labels=(xf_sub.columns.difference(xv.columns)),axis=1)

    xt = xt.drop(labels=(xt.columns.difference(xv.columns)),axis=1)
    xf_full = xf_full.drop(labels=(xf_full.columns.difference(xv.columns)),axis=1)
    xf_sub = xf_sub.drop(labels=(xf_sub.columns.difference(xv.columns)),axis=1)

    #print(xt.shape)
    #print(xf_full.shape)
    #print(xf_sub.shape)
    #print(xv.shape)

    xv = xv[xt.columns]
    xf_full = xf_full[xt.columns]
    xf_sub = xf_sub[xt.columns]

    # Set pandas dataframes to numpy arrays, also create numpy arrays from the time and vital status data
    # The large 32 cancer dataset is used here

    xf = xf_full.to_numpy()
    yf = clinicalf_full

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

    # print(yf['time'].max())

    # Create a plot where you set the time interval bins in the bins variable, the plot should be as straight as possible, containing an equal number of death events per time interval
    #test = yf.loc[yf['vital_status'] == 1]
    #plt.hist(test['time'], bins=[0,55,100,135,180,240,260,322,370,430,455,520,580,620,730,840,970,1140,1350,1590,1900,2540,9200],ec='black')


    # Here we set the breaks for our time intervals which we obtained with the plot above
    # Use the first breaks for the large 32 cancer dataset, use the second breaks for the subset of cancers

    # breaks=np.asarray([0,55,100,135,180,240,260,322,370,430,455,520,580,620,730,840,970,1140,1350,1590,1900,2540,9200])
    breaks=np.asarray([0,40,98,153,180,240,280,330,370,428,495,580,660,750,860,990,1170,1400,1660,2100,7000])
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


    importlib.reload(nnet_survival)

    # Set the optimal parameters found for the source model
    print("Creating model")
    optimal_l2 = 0.01
    optimal_batch = 256

    # Scale the input features using MinMaxScaler

    scaler = MinMaxScaler()
    xf_t = scaler.fit_transform(xf)

    # Set the labels to the survival matrices

    yf = np.array(yf_final[0])

    # Create the model with the optimal parameters

    optimal_model = Sequential()
    optimal_model.add(Dense(np.sqrt(xf.shape[1]), input_dim=xf.shape[1], bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(optimal_l2)))
    optimal_model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
    optimal_model.add(nnet_survival.PropHazards(n_intervals))
    opt = optimizers.Adam(learning_rate=0.000001)
    optimal_model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=opt)


    # Create a tensorboard callback which saves the loss during the training of the model

    dir_for_log = ("Logs/ktuner/source_model_full_paad_correct")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=dir_for_log, histogram_freq=0, embeddings_freq=0, write_graph=False, update_freq='batch')
    callbacks_list = [tensorboard_callback]

    # Fit the model to the input data xf_t and yf
    print("Training model...")
    history = optimal_model.fit(xf_t,yf,batch_size=optimal_batch,epochs=1200, callbacks=callbacks_list,verbose=0)

    # Save the weights of the source model
    print("Saving weights of ...")
    optimal_model.save_weights("Weights/setting_4_1")   

    # Get performance of source model on full dataset which is very optimistic

    print("Performance model on full dataset (very optimistic):")
    y_pred = optimal_model.predict(xf_t, verbose=0)
    oneyr_surv = np.cumprod(y_pred[:, 0:np.nonzero(breaks > 365)[0][0]], axis=1)[:, -1]
    c_index_test = concordance_index(ytime, oneyr_surv, ystatus)
    print("C_index: " + str(c_index_test))

    #
    # Data preperation TCGA PDAC
    #

    # Set pandas dataframes of TCGA RNA-seq data to numpy array
    # Set pandas dataframes of TCGA clinical data (time and vital_status) to numpy arrays

    xt = xt.to_numpy()
    #
    ytime = yt['time'].to_numpy()
    ystatus = yt['vital_status'].to_numpy()
    ystatusbool = yt['vital_status'].astype(bool).to_numpy()

    # Here we set the breaks for our time intervals which we obtained with the plot above

    breaks=np.array([0,110,130,190,250,310,475,550,650,730,2200])
    n_intervals = len(breaks) - 1

    # Create the survival array for the TCGA PDAC patients using the nnet_survival fuction 'make_surv_array'
    # This function takes the following input:
    #    ytime - numpy array of each patient's time until death/censor value
    #    ystatus - numpy array of each patient's survival status (1 for death event, 0 for alive/censored)
    #    breaks - the breaks for the time intervals that were found

    y_t = nnet_survival.make_surv_array(ytime, ystatus, breaks)

    # Create time quantiles to later be used in stratification

    first_quantile = np.quantile(ytime,0.25)
    second_quantile = np.quantile(ytime,0.5)
    third_quantile = np.quantile(ytime,0.75)


    # For every patient's time value in ytime check in what quantile it falls and make a new list that contains the corresponding time quantile for each patient

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

    # For every survival matrix, status value, and quantile value merge the status and quantile values together and merge with the survival matrix.

    yt_final = ([],[])

    for matrix,status,quantile in zip(y_t,ystatus,yquantile):
        yt_final[0].append(matrix)
        status_quantile = (str(int(status)) + "_" + str(quantile))
        yt_final[1].append(status_quantile)

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
    # vbreaks = np.array([0,18,100,120,132,150,173,177,190,400,2250])

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

    ### Run TCGA PDAC Model with transferred weights from Multi-Cancer Model (Fine-tuning step)
    ### This is equal to the second block for experiment 3, Figure 4.

    importlib.reload(nnet_survival)

    # Scale the input features using MinMaxScaler
    scaler = MinMaxScaler()
    xt_t = scaler.fit_transform(xt)

    # Set the optimal parameters found for the model
    optimal_l2 = 0.00001
    optimal_batch = 128

    # Create a model to initialize the weights of the source model into (should have the same architecture)
    init_model = Sequential()
    init_model.add(Dense(np.sqrt(xt.shape[1]), input_dim=xt.shape[1], bias_initializer='zeros', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    init_model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
    init_model.add(nnet_survival.PropHazards(20))
    init_model.compile(loss=nnet_survival.surv_likelihood(20), optimizer=optimizers.Adam(learning_rate=0.000001))

    # Load the weights of the source model into the initialization model
    init_model.load_weights("Weights/setting_4_1")


    # Create a model to run the fine tuning step on (this uses the source model's weights)
    model = Sequential()
    model.add(Dense(np.sqrt(xt.shape[1]), input_dim=xt.shape[1], bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(optimal_l2)))
    model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
    model.add(nnet_survival.PropHazards(n_intervals))
    model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=optimizers.Adam(learning_rate=0.000001))

    # Transfer weights of source model's second hidden layer to fine tuning model's second hidden layer
    model.layers[1].set_weights(init_model.layers[1].get_weights())


    # Create a tensorboard callback which saves the loss during the training of the model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=("ktuner/logs_source_finetune_model_subset"), histogram_freq=0, embeddings_freq=0, write_graph=False, update_freq='batch')
    callbacks_list = [tensorboard_callback]

    # Fit the model to the input data xt_t and y_t
    history = model.fit(xt_t,y_t,batch_size=optimal_batch,epochs=2600, callbacks=callbacks_list,verbose=0) 

    # Save the weights of the model
    model.save_weights("Weights/setting_4_2")

    ### Train ICGC PDAC and validate with 5-fold CV
    #### Experiment 3
    # Set transfer_learning to True, 
    # set name of results_file, 
    # load weights of model trained on all TCGA cancers, 
    # and TCGA PDAC data at init_model.load_weights("path_to_weights")

    importlib.reload(nnet_survival)

    # Set transfer_learning to True, if you want transfer learning, else set it too False
    transfer_learning = True

    # File which will store the C-indexes obtained
    results_file = "Results/setting_4_c"

    cindex_list = []
    cindex_train_list = []

    total_cindexes = []
    total_cindexes_train = []

    # Set optimal parameters found with hyperparameter tuning
    optimal_l2 = 0.00001
    optimal_batch = 128


    if transfer_learning == True:
        # Create a model to initialize the weights of the source model into (should have the same architecture)
        
        init_model = Sequential()
        init_model.add(Dense(np.sqrt(xt.shape[1]), input_dim=xt.shape[1], bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(0.00001)))
        init_model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
        init_model.add(nnet_survival.PropHazards(10))
        init_model.compile(loss=nnet_survival.surv_likelihood(10), optimizer=optimizers.Adam(learning_rate=0.000001))

        # Load the weights of the source model into the initialization model

        init_model.load_weights("Weights/setting_4_2")

        # Create Repeated Stratified Kfold cross validation object

        # was 5 but gave error: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
        # assume it's because of missing TIP data?
        # rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=24)
        rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=24)

        k = 1

        test_performance = []
        train_performance = []


        # Start the repeated 5 fold cross validation (will repeat 50 times, based on n_splits = 5 and n_repeats = 10)
        for train_indices, test_indices in rskf.split(xv,yv_final[1]):
            print("fold: " + str(k) + "/50")
            
            # Create splits of the input data
            # Split the input features of the ICGC PDAC data
            x_train, x_test = xv[train_indices], xv[test_indices]
            # Split the input labels of the ICGC PDAC data (survival matrices)
            y_train, y_test = np.array(yv_final[0])[train_indices], np.array(yv_final[0])[test_indices]
            # Split the status_quantile variables for each patient
            y_train_labels, y_test_labels = np.array(yv_final[1])[train_indices], np.array(yv_final[1])[test_indices]
            # Split the status values for each patient
            ystatus_train, ystatus_test = yvstatus[train_indices], yvstatus[test_indices]
            # Split the time values for each patient
            ytime_train, ytime_test = yvtime[train_indices], yvtime[test_indices]

            # Scale the input train and test features using MinMaxScaler
            scaler = MinMaxScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            # Create a model to run on the ICGC PDAC (target) data (this uses the source model's weights)
            model = Sequential()
            model.add(Dense(np.sqrt(xv.shape[1]), input_dim=xv.shape[1], bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(optimal_l2)))
            model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
            model.add(nnet_survival.PropHazards(n_intervals))
            model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=optimizers.Adam(learning_rate=0.0000001))
            
            if transfer_learning == True:
                # Transfer weights of source model's second hidden layer to fine tuning model's second hidden layer
                if approach == 1:
                    model.layers[0].set_weights(init_model.layers[0].get_weights())
                elif approach == 2:
                    model.layers[0].set_weights(init_model.layers[0].get_weights())
                    model.layers[0].trainable = False
                elif approach == 3:
                    model.layers[1].set_weights(init_model.layers[1].get_weights())
                elif approach == 4:
                    model.layers[1].set_weights(init_model.layers[1].get_weights())
                    model.layers[0].trainable = False
                    model.layers[1].trainable = False
                elif approach == 5:
                    model.layers[0].set_weights(init_model.layers[0].get_weights())
                    model.layers[1].set_weights(init_model.layers[1].get_weights())
                elif approach == 6:
                    model.layers[0].set_weights(init_model.layers[0].get_weights())
                    model.layers[1].set_weights(init_model.layers[1].get_weights())
                    model.layers[0].trainable = False
                    model.layers[1].trainable = False
                    
                
            # Create a tensorboard callback which saves the loss during the training of the model
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=("ktuner/logs_pdac_only/fold_" + str(k)), histogram_freq=0, embeddings_freq=0, write_graph=False, update_freq='batch')
            callbacks_list = [tensorboard_callback]

            # Fit the model to the input data x_train_scaled and y_train
            history = model.fit(x_train_scaled,y_train,batch_size=optimal_batch,epochs=2600, callbacks=callbacks_list,verbose=0)

            # Make predictions on the train data for the training performance
            y_pred = nnet_survival.nnet_pred_surv(model.predict(x_train_scaled,verbose=0), vbreaks, 365)
            # Calculate C-index for train data using time, predictions and status values
            c_index_train = concordance_index(ytime_train, y_pred, ystatus_train)

            # Make predictions on the test data for test performance
            y_pred = nnet_survival.nnet_pred_surv(model.predict(x_test_scaled,verbose=0), vbreaks, 365)
            # Calculate C-index for test data using time, predictions and status values
            c_index_test = concordance_index(ytime_test, y_pred, ystatus_test)

            # Store the C-index values in lists to write to .txt files
            train_performance.append(c_index_train)
            test_performance.append(c_index_test)

            total_cindexes.append(c_index_test)
            total_cindexes_train.append(c_index_train)

            # Save results of current fold to file
            with open(results_file, 'a') as o:
                print("Fold: " + str(k) + "\n",file=o)
                print("C-index train set: " + str(c_index_train), file=o)
                print("C-index test set: " + str(c_index_test), file=o)

            k += 1

        # Save average C-index values and list of total C-indexes to .txt file
        with open(results_file, 'a') as o:
            print("\n",file=o)
            print("Average C-index train total: " + str(sum(total_cindexes_train) / 50),file=o)
            print("Average C-index test total: " + str(sum(total_cindexes) / 50),file=o) 
            print(str(total_cindexes),file=o)
            print(str(total_cindexes_train),file=o)

    ### Test survival probabilities for patient 5
    test_patient = xv[5]
    xv = np.delete(xv, 5, 0)
    test_patient_y = yv_final[0][5]
    y_train_all = list(yv_final[0])
    y_train_all = np.delete(y_train_all, 5, 0)

    # Set the file that stores the survival probabilities

    results_file = "Results/setting_4_surv"

    # Set optimal parameters found with hyperparameter tuning on TCGA PDAC data
    optimal_l2 = 0.00001
    optimal_batch = 128

    # Create a model to initialize the weights of the source model into (should have the same architecture)

    init_model = Sequential()
    init_model.add(Dense(np.sqrt(xv.shape[1]), input_dim=xv.shape[1], bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(0.00001)))
    init_model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
    init_model.add(nnet_survival.PropHazards(10))
    init_model.compile(loss=nnet_survival.surv_likelihood(10), optimizer=optimizers.Adam(learning_rate=0.000001))

    # Load the weights of the source model into the initialization model

    init_model.load_weights("Weights/setting_4_2")
    

    # Create Repeated Stratified Kfold cross validation object

    # was 5 but gave error: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
    # assume it's because of missing TIP data?
    # rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=24)
    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=24)

    k = 1

    surv_probablity = []

    # Set the prediction time you want to get the survival probability of

    pred_time = 456.25

    # Start the repeated 5 fold cross validation (will repeat 50 times, based on n_splits = 5 and n_repeats = 10)
    for _ in range(50):
        print("Loop: " + str(k) + "/50")
        
        # Scale the input train and test features using MinMaxScaler
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(xv)
        x_test_scaled = scaler.transform(test_patient.reshape(1, -1))
        # xv_ext_train.iloc[:,0:14242] = scaler.transform(xv_ext_train.iloc[:,0:14242])
        # xv_ext_test.iloc[:,0:14242] = scaler.transform(xv_ext_test.iloc[:,0:14242])

        # Create a model to run on the ICGC PDAC (target) data (this uses the source model's weights)
        model = Sequential()
        model.add(Dense(np.sqrt(xv.shape[1]), input_dim=xv.shape[1], bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(optimal_l2)))
        model.add(Dense(1, use_bias=0, kernel_initializer='zeros'))
        model.add(nnet_survival.PropHazards(10))
        model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=optimizers.Adam(learning_rate=0.000001))

        
        # Transfer weights of source model's second hidden layer to fine tuning model's second hidden layer
        model.layers[1].set_weights(init_model.layers[1].get_weights())

        # Create a tensorboard callback which saves the loss during the training of the model
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=("ktuner/logs_pdac_icgc_testpatient/fold_" + str(k)), histogram_freq=0, embeddings_freq=0, write_graph=False, update_freq='batch')
        callbacks_list = [tensorboard_callback]

        # Fit the model to the input data x_train_scaled and y_test_all
        history = model.fit(x_train_scaled,y_train_all,batch_size=optimal_batch,epochs=2600, callbacks=callbacks_list,verbose=0)

        # Predict the survival probability of patient 5 at set pred_time
        y_pred = nnet_survival.nnet_pred_surv(model.predict(x_test_scaled,verbose=0), vbreaks, pred_time)
        
        # Save survival probability to list
        surv_probablity.append(y_pred)
        
        k += 1

    # Store survival probabilities to file
    with open(results_file, 'a') as o:
        print("Survival probabilities for " + str(pred_time) + " days:\n", file=o)
        print(str(surv_probablity),file=o)
        print(str(sum(surv_probablity)/50),file=o)
        o.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="EMC Transfer Learning",
        description="https://github.com/ErasmusMC-Bioinformatics/transferlearningproject",
    )
    parser.add_argument("--approach", type=int, choices=[1,2,3,4,5,6], default=3)

    args = parser.parse_args()

    approach = args.approach

    main(approach)