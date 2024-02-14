# -*- coding: utf-8 -*-
"""
Author: Elisa Friedmann (efriedmann@umass.edu)

This script calls all dataframe preprocessing and prep functions for the random forest, 
runs the random forest, and predicts all unmatched LS2 and fusion images. Returns a training df, result df, test set dfs, and prediction dfs.
"""

#Master Functions
import os
import numpy as np
import random
import pandas as pd
import math
from numpy import mean, arange, std
import xarray as xr
import sklearn
import csv
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from datetime import datetime
import plotly
import plotly.express as px
import joblib
from datetime import date
from sklearn import ensemble, datasets, tree
from sklearn.tree import plot_tree
from sklearn.compose import make_column_selector, ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.pipeline import Pipeline, make_pipeline
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, accuracy_score, log_loss
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from collections import OrderedDict
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE, RFECV

# importing  all the functions defined in modelFunctions.py
from 0_RFmodelFunctions import *

##################
# Vars, Paths, and Seeds
##################
train_size = 0.9

thresholdTSS = 3000
thresholdTSSstr = str(thresholdTSS)
pixelCount = 3

experiment = 'allModels2'
test = 'conus3000_all'
stage = 'tests' #results

print('Experiment:', experiment)
print('Test:', test)




# log transformations
cols_to_log = ['tss', 'GR2', 'red', 'green', 'blue', 'nir', 'swir1', 'swir2',
            'NR', 'BR', 'GR', 'SR', 'BG', 'RG', 'NG', 'SG', 'BN', 'GN', 'RN', 'SN', 'BS', 'GS', 'RS', 'NS', 'R_GB', 'R_GN', 'R_GS', 'R_BN', 'R_BS', 'R_NS', 'G_BR', 'G_BN', 'G_BS', 'G_RN', 'G_RB', 'G_NS', 'B_RN', 'B_RS', 'B_GN', 'B_GS', 'B_NS', 'B_RG','N_RG', 'N_RB', 'N_RS', 'N_GB', 'N_GS', 'N_BS', 'GN2', 'RN2']


slurm = int(os.environ['SLURM_ARRAY_TASK_ID'])
 
random.seed(slurm)
random_state = random.randint(0,5000)
today = datetime.today().strftime('%Y-%m-%d')

##################
# Read and Clean
##################

########### Matchup Training/Testing Data ##############
# Concatenate all DataFrames into a single DataFrame
path = "/yourPath/allMatchups_conus/"
files = [os.path.join(path, file) for file in os.listdir(path)]
matchup_df = pd.concat((pd.read_csv(f, engine='python') for f in files if f.endswith('csv')), ignore_index=True)



# preprocess and clean
preprocessed_df = preprocess_dataframe(dataframe = matchup_df, thresholdTSS = thresholdTSS, pixelCount = pixelCount, cols_to_log = cols_to_log)
print('# Sites:', (preprocessed_df.SiteID.nunique()))

# prep for rf
dfs = rf_prep_dataframe(df = preprocessed_df) 
#print(len(matchup_upArea_dict))

# print(list(dfs.values())[0].columns.values.tolist())


########### LS2 Unmatched Prediction Image Data ##############

# Call in LS2 Unmatched Data
ls2_raw = pd.read_csv('/yourPath/LS2/LS2NoMatch_2000-2023.csv')

ls2 = preprocess_preds(df = ls2_raw, pixelCount = pixelCount, cols_to_log = cols_to_log)
print('# LS2 Predictions:', ls2.shape)

ls2_prep = rf_prep_preds(df = ls2, dataSource = 'LS2')


########### Fusion Unmatched Prediction Image Data ##############

# Call in Fusion Unmatched Data
path = "/yourPath/FusionNoMatchups/"
files = [os.path.join(path, file) for file in os.listdir(path)]
fusion_raw = pd.concat((pd.read_csv(f, engine='python') for f in files if f.endswith('csv')), ignore_index=True)

fusion = preprocess_preds(df = fusion_raw, pixelCount = pixelCount, cols_to_log = cols_to_log)
print('# Fusion Predictions:', fusion.shape)

fusion_prep = rf_prep_preds(df = fusion, dataSource = 'Fusion')


###################
# RF Model
##################

# Run the function
results  = run_random_forest(dfs = dfs, train_size = train_size, random_state = random_state, ls2_prep = ls2_prep, fusion_prep = fusion_prep)

# Print the results
for result in results:
    print('RESULTS:', result)
    

pd.DataFrame(results).to_csv(r'/yourPath/' + experiment + '/tests/' + test + '/modelScores/'+ str(random_state) + '_' + experiment + '_' + test + '_modelScore.csv', index = False)

