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

def processTCGA(mrna_folder, clinical_folder, type):
    #
    # Import multi-cancer Data
    #
    print("Importing multi-cancer mrna data", end='')
    mrna_list = []

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
    x = pd.concat(x_list)
    x = x.dropna(axis=1)
    clinical = pd.concat(clinical_processed_list)

    print("Storing datasets...")
    filename_store_mrna = str(str(mrna_folder) + "/processed/x.csv")
    filename_store_clinical = str(str(clinical_folder) + "/processed/clinical.csv")

    x.to_csv(filename_store_mrna, sep=',')
    clinical.to_csv(filename_store_clinical, sep=',')

def processICGC(mrna_folder,clinical_folder):
    #
    # ICGC Data
    #
    print("Reading TCGA ICGC data...")
    clinicaldata = pd.read_csv(str(mrna_folder) + f"PDAC_ICGC_clinical.csv", sep=',')
    mrna = pd.read_csv(str(clinical_folder) + f"PDAC_ICGC.csv", sep=',')

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

    print("Storing datasets...")
    filename_store_mrna = str(str(mrna_folder) + "/processed/x.csv")
    filename_store_clinical = str(str(clinical_folder) + "/processed/clinical.csv")

    xv.to_csv(filename_store_mrna, sep=',')
    yv.to_csv(filename_store_clinical, sep=',')


def main(mrna_folder, clinical_folder,type):
    print("Start processing...")
    if type == 1:
        processTCGA(mrna_folder,clinical_folder)
    if type == 2:
        processICGC(mrna_folder,clinical_folder)
    print("Finished processing! Check your input folders for processed files")
    

if __name__ == "__main__":
    
    #Set inputs, depends on if called from python script or command line
    mrna_folder = 
    clinical_folder = 
    type = 

    main(mrna_folder,clinical_folder,type)