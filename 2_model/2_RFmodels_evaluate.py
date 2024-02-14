# -*- coding: utf-8 -*-
"""
Author: Elisa Friedmann (efriedmann@umass.edu)

This script takes the outputs from 1_RFmodels.py and outputs figures to evaluate model skill across all experiments. Outputs figures in jpg files.
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
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, accuracy_score, log_loss
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from collections import OrderedDict
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE, RFECV
from PIL import Image

# importing  all the functions defined in modelFunctions.py
from 0_RFmodelFunctions import *

###############
# Declare vars
###############
experiment = 'allModels2'
test = 'conus3000_all'
stage = 'tests' #results

print('Experiment:\n', experiment)
print('Test:\n', test)

#Define metrics
##############
metricList = ['Test Rel Error', 'Test MAE', 'Test RMSE', 'Test RMSLE', 'Test MAPE', 'Test MDAE', 'Test MDAPE']

# Define colors for each 'Model'
###################################
colors = {'LS2Fusion': 'orange', 'S2': 'red', 'Fusion': 'green', 'Landsat':'blue'}
#Define paths
#############
general_path = '/nas/cee-water/cjgleason/ellie/SNiP/RFmodel/CONUS/RF_batch/' + experiment +'/' + stage + '/'+ test

#Get all models' scores and Concatenate all DataFrames into a single DataFrame
#######################
path = general_path + '/modelScores/'
files = [os.path.join(path, file) for file in os.listdir(path)]
modelScores = pd.concat((pd.read_csv(f) for f in files if f.endswith('.csv')), ignore_index=True)

modelScores = pd.DataFrame(modelScores.loc[modelScores['Test'] == test])
print('MODEL SCORE COLUMNS:\n', modelScores.columns.values.tolist())

#Separate models into dfs
###########################
LS2Fusion = pd.DataFrame(modelScores.loc[modelScores['Experiment'] == 'LS2Fusion'].copy())
Landsat = pd.DataFrame(modelScores.loc[modelScores['Experiment'] == 'Landsat'].copy())
Fusion = pd.DataFrame(modelScores.loc[modelScores['Experiment'] == 'Fusion'].copy())
S2 = pd.DataFrame(modelScores.loc[modelScores['Experiment'] == 'S2'].copy())

#Dataframe dictionary separated by each model
##################################################
dfs = {'LS2Fusion': LS2Fusion, 'Landsat': Landsat, 'Fusion': Fusion, 'S2': S2} #, 'LandFusion': LandFusion , 'LS2': LS2,

#Save description of all stats by experiment and test for comparison later
############################################
describe_df = pd.concat([df.describe() for df in dfs.values()], keys=dfs.keys()).rename_axis(index=['Model', 'Stat'])
describe_df['Test'] = test
describe_df.to_csv(general_path + '/' + experiment + '_' + test + '_modelMetricResults.csv')


#obsPred Plot using all the TestSets
######################################
path = general_path + '/testSets/'
files = [os.path.join(path, file) for file in os.listdir(path)]
testSet_raw = pd.concat((pd.read_csv(f, low_memory=False) for f in files if f.endswith('csv')), ignore_index=True)

#Separate test sets into dfs
###############################
LS2Fusion_testSet = pd.DataFrame(testSet_raw.loc[testSet_raw['Experiment'] == 'LS2Fusion'].copy()).dropna(axis=1, how='all')
Landsat_testSet = pd.DataFrame(testSet_raw.loc[testSet_raw['Experiment'] == 'Landsat'].copy()).dropna(axis=1, how='all')
Fusion_testSet = pd.DataFrame(testSet_raw.loc[testSet_raw['Experiment'] == 'Fusion'].copy()).dropna(axis=1, how='all')
S2_testSet = pd.DataFrame(testSet_raw.loc[testSet_raw['Experiment'] == 'S2'].copy()).dropna(axis=1, how='all')


#Dataframe dictionary separated by each model
###############################################
dfs_testSet = {'LS2Fusion': LS2Fusion_testSet, 'Landsat': Landsat_testSet, 'Fusion': Fusion_testSet,'S2': S2_testSet} #, 'LandFusion': LandFusion_testSet}  'LS2': LS2_testSet, 


#############################
# Call the function for plots
###############################
plot_obsPred(dfs = dfs_testSet, predColumn = 'tssPred', obsColumn = 'tss', colors = colors, general_path = general_path, specific_path = '/plots/obsPred.png')
plot_bias(dfs_testSet, colors = colors, general_path = general_path, specific_path = '/plots/bias.png')


feat_importance(dataframes = dfs, colors = colors, general_path = general_path, specific_path = '/plots/featImp_plot.png')
plot_cdfs_to_image(dfs = dfs, metricList = metricList, colors = colors, general_path = general_path, specific_path = '/plots/cdf_plot.png')
barPlots(df = describe_df, metricList = metricList, colors = colors, general_path = general_path, specific_path = '/plots/barPlot.png')

