# -*- coding: utf-8 -*-
"""
Author: Elisa Friedmann (efriedmann@umass.edu)

Core functions of the model and evaluation:
    - Matched and Unmatched cleaning and thresholding
    - Model preparation
    - Model prediction
    - Model Evaluation
    - Figure generation

These function require that the preprocessing steps to generate images and their matchups
divided as LS2 and Fusion dataframes are complete.
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


# Categorical satellite delineation
def get_sat(satID):
    """
    Create string identifiers for source satellite.

    Parameters:
    df (pd.DataFrame): Input DataFrame with satellite numerical ID column.

    Returns:
    pd.DataFrame: DataFrame with new string sat identifier column.
    """
    if satID == 5.0:
        return 'L5'
    elif satID == 7.0:
        return 'L7'
    elif satID == 8.0:
        return 'L8'
    elif satID == 9.0:
        return 'L9'
    elif satID == 2.0:
        return 'S2'
    elif satID == 1.0:
        return 'F'
    else:
        return 'Other'


def col_log_transform(df, cols):
    """
    Apply log10 transformation to specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    cols (list): List of column names to apply log transformation.

    Returns:
    pd.DataFrame: DataFrame with log-transformed columns.
    """
    for col in cols:
        # Check if the column exists in the DataFrame
        if col in df.columns:
            df[f'{col}_log'] = np.log10(df[col].replace(0, np.nan))
            df = df.replace(np.nan, 0).copy()

    return df




def preprocess_dataframe(dataframe, thresholdTSS, pixelCount, cols_to_log):
    """
    Preprocess and clean fusion and LS2 matchup (training/testing) dataframes to prep for RF model.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    thresholdTSS: High TSS boundary to filter dataframe
    pixelCount: The minimum number of pixels needed per image
    cols_to_log: The columns to be log transformed for RF input features

    Returns:
    pd.DataFrame: DataFrame with filtered and log-transformed columns.
    """
    # Drop NA, synchronize column names if needed, and remove negative band values
    df = dataframe.dropna(subset=['red', 'date']).drop_duplicates(subset=['SiteID', 'date', 'red', 'tss'])
    df = df[(df.red > 0) & (df.nir > 0) & (df.blue > 0) & (df.green > 0) & 
            (df.swir1 > 0) & (df.swir2 > 0) & (df.nir > 0)]



    # Fill in missing values and create new columns
    df['TZID'] = 'UTC'
    df['type'] = 'Stream'
    df['units'] = df['units'].fillna('mg/l') 
    df['parameter'] = df['parameter'].fillna('Sediment') 
    df = df.loc[:, ~df.columns.duplicated()]

    # Apply log transformation
    df = col_log_transform(df = df, cols_to_log = cols_to_log)

    #Transform satellite to categorical variable
    df['sat_cat'] = df['sat'].apply(lambda x: get_sat(x))
    
    # Call the function to calculate 'hue' and add it to the DataFrame
    #df = calc_hue(df)    
    
    #Remove outliers and set thresholds for TSS and pixelCounts
    dfClean = df[(df.tss < thresholdTSS) & (df.R_GB < 2)& (df.B_RG < 2) & (df.B_RS < 5) & (df.BR_G > -5) & (df.units == 'mg/l') & (df.pixelCount >= pixelCount)]
    
    
    #Write it out
    dfClean.to_csv(r'/youPath/AllMatchups_2000-2023_RFpreprocessed.csv', index = False)
    
    return dfClean


def rf_prep_dataframe(df):
        """
    Prep fusion and LS2 dataframes to input in RF model.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    dfs (dictionary): A dictionary of four dataframes each separated into the sensor combination needed for each experiment.
    """
    df_train = df.loc[:, ~df.columns.duplicated()]
 


    # One-hot encode the specified columns
    #df_parameter = pd.get_dummies(df_train['parameter'])
    df_Matchup = pd.get_dummies(df_train['Matchup'])
    df_sat_cat = pd.get_dummies(df_train['sat_cat'])

    # Concatenate original dataframe with one-hot encoded columns
    df_train = pd.concat([df_train, df_Matchup, df_sat_cat], axis=1) #, df_parameter

    # Create a MinMaxScaler instance and transform the specified column
    scaler = MinMaxScaler()
    df_train['N_S_scaled'] = scaler.fit_transform(df_train[['N_S']])



    # Save context vars as a dictionary for later
    #upArea_dict = dict(zip(df_train['SiteID'].values, df_train['upArea'].values))
    
    
    # Drop specified columns including unlogged predictor
    df_train = df_train.reset_index(drop=True).dropna(axis=0)
    
    print('df_train:\n', df_train.columns.values.tolist())
    

    # Define four DataFrames
    LS2Fusion = df_train.drop(['LS2','L5', 'L7', 'L8', 'L9','S2', 'Fusion', 'F'], axis=1) #
    Fusion = df_train[(df_train['Matchup'] == 'Fusion')].drop(['Fusion', 'F', 'LS2','L5', 'L7', 'L8', 'L9', 'S2'], axis=1) # & (df['pixelCount'] > 10)
    LS2 = df_train[(df_train['Matchup'] == 'LS2')].drop(['Fusion', 'F', 'LS2','L5', 'L7', 'L8', 'L9', 'S2'], axis=1)
    S2 = df_train[(df_train['sat_cat'] == 'S2')].drop(['Fusion', 'F', 'LS2','L5', 'L7', 'L8', 'L9', 'S2'], axis=1)
    # LandFusion = df_train[(df_train['sat'] != 'S2')].drop(['Fusion', 'F', 'LS2','L5', 'L7', 'L8', 'L9', 'S2'], axis=1)
    Landsat = df_train[(df_train['Matchup'] == 'LS2') & (df_train['sat_cat'] != 'S2')].drop(['Fusion', 'F', 'LS2','L5', 'L7', 'L8', 'L9', 'S2'], axis=1)


    # Store the DataFrames in a dictionary with their variable names
    dfs = {'S2': S2, 'Landsat': Landsat,'LS2Fusion': LS2Fusion, 'Fusion': Fusion} #, 'LS2': LS2, 'LandFusion': LandFusion
    
    # Reset index, drop NaN rows, and return the resulting DataFrame and any context if scaled in some way
    return dfs #, _dict


def preprocess_preds(df, pixelCount, cols_to_log):
    """
    Preprocess and clean fusion and LS2 unmatched dataframes to prep for RF model prediction.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    pixelCount: The minimum number of pixels needed per image
    cols_to_log: The columns to be log transformed for RF input features

    Returns:
    pd.DataFrame: DataFrame with filtered and log-transformed columns.
    """
    
    # Filter df data based on specified conditions
    df = df[(df.red > 0) & (df.pixelCount > pixelCount) & (df.nir > 0) & (df.blue > 0) & 
            (df.green > 0) & (df.swir1 > 0) & (df.swir2 > 0) & (df.nir > 0) & 
            (df.R_GB < 2)& (df.B_RG < 2) & (df.B_RS < 5) & (df.BR_G > -5)]
    

    df = df.drop_duplicates(subset=['SiteID', 'date'])

    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[(df['date'] >= '2000-01-01')]
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month

    # Create a new column 'Region' based on the 'City' column
    df['sat_cat'] = df['sat'].apply(lambda x: get_sat(x))

    # Call the function to calculate 'hue' and add it to the df
    #df = calc_hue(df)
        
    # Apply log transformation
    df = col_log_transform(df, cols_to_log)
    
    #Drop any duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    return df

#prep for rf
def rf_prep_preds(df, dataSource):
    """
    Prep fusion and LS2 unmatched dataframes to predict in RF model.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    dataSource (string): A string denoting whether Fusion or LS2 data is being input to the function
    
    Returns:
    df (pd.Dataframe): A dataframe with the necessary prep for RF prediction
    """
    #Manually add all other Matchup and Fusion columns
    if dataSource == 'LS2':
        df.loc[:, 'Fusion'] = 0
        df.loc[:, 'LS2'] = 1
        df.loc[:, 'F'] = 0

        # One-hot encode the specified columns
        df_sat_cat = pd.get_dummies(df['sat_cat'])

        # Concatenate original dataframe with one-hot encoded columns
        df = pd.concat([df, df_sat_cat], axis=1).drop(['sat_cat'], axis = 1) #, df_parameter

    elif dataSource == 'Fusion':
        df.loc[:, 'Fusion'] = 1
        df.loc[:, 'LS2'] = 0
        df.loc[:, 'F'] = 1
        df.loc[:, 'L5'] = 0
        df.loc[:, 'L7'] = 0
        df.loc[:, 'L8'] = 0
        df.loc[:, 'L9'] = 0
        df.loc[:, 'S2'] = 0
        
    return df

def predict_unmatched(df_name, rf, X_test, preds_prepped_df):
     """
    Predict unmatched LS2 and Fusion data.

    Parameters:
    df_name (string): Input DataFrame name from the dictionary, denotes the experiment
    rf: The random forest trained model
    X_test (pd.DataFrame): The input features for the RF
    preds_prepped_df (pd.DataFrame): The unmatched LS2 or Fusion dataframe for prediction

    Returns:
    pd.DataFrame: DataFrame with predictions without logged columns to save space.
    """
   
    #Get training df and preds df and reset their index for rest of function
    preds_prepped_df = preds_prepped_df.sort_index(axis=1).reset_index(drop=True)
    #Find and filter the prediction df columns to match the training df
    cols_to_keep = preds_prepped_df.columns[preds_prepped_df.columns.isin(X_test.columns)]
    pred_df = preds_prepped_df[cols_to_keep]
    
    #Keep the columns not used for the rf to add back on later
    preds_df_all = preds_prepped_df
    

#         pred_df = prep_df
    pred_df = pred_df[X_test.columns]
    diff_cols = set(pred_df.columns).symmetric_difference(set(X_test.columns))
    print('columns differing between rf and dataframe', diff_cols)
    
    #Check for column na's which bonk the RF
    #By column
    naSum = pred_df.isna().sum()
    selected_cols = naSum[naSum > 0]
    print('COLS WITH NA VALUES:', selected_cols)

    #By row
    row_mask=pred_df.isnull().any(axis=1)
    col_mask=pred_df.isnull().any(axis=0) 
    print('LOC WITH ROW NA VALUES:', X_test.loc[row_mask,col_mask])
    
        
    preds_df_all['tssPred_log'] = rf.predict(pred_df)
    preds_df_all['tssPred'] = 10**(preds_df_all['tssPred_log'])
    preds_df_all['Experiment'] = df_name
    preds_df_all['test'] = test
    preds_df_all['RandomState'] = random_state
        
    
    # Remove columns and simplify write-out
    # List of columns to keep regardless of capital letters: my way to filter dataframe of all log transformed bands (saves space) without having to specify every column; could be more svelte
    columns_to_keep = ['SiteID', 'Matchup', 'sceneID', 'Experiment', 'RandomState', 'tssPred', 'tssPred_log', 'upArea', 'subArea', 'distMain']

    # Get the list of columns that contain capital letters
    columns_with_capital_letters = [col for col in preds_df_all.columns if any(c.isupper() for c in col)]
    # Remove the columns with capital letters except the ones in the 'columns_to_keep' list
    columns_to_remove = [col for col in columns_with_capital_letters if col not in columns_to_keep]
    predictionAll = preds_df_all.drop(columns=columns_to_remove).dropna(axis=1, how='all')
    print('PRED ALL COLUMNS:\n', predictionAll.columns.values.tolist())
    predictionAll = predictionAll[['nir_log','red_log','swir1_log','swir2_log','SiteID','blue','blue_sd','date','green','green_sd','lat','long','nir','nir_sd','red','red_sd','sat', 'swir1','swir1_sd', 'swir2','swir2_sd','Matchup', 'width', 'tssPred', 'tssPred_log', 'test', 'distMain', 'subArea','upArea','Experiment', 'RandomState']]
    
    return predictionAll


# Function to run Random Forest on each DataFrame
def run_random_forest(dfs, train_size, random_state, ls2prep, fusion_prep):
     """
    Master function to train Random Forest model as well as call functions to predict unmatched LS2 and Fusion data.

    Parameters:
    dfs (dictionary): Dictionary of dataframes that will each be run through the RF
    train_size: The random forest training/testing split (percentage)
    random_state (number): The random state generated by the seed, helps identify the result run and file names
    ls2prep (pd.DataFrame): The unmatched LS2 dataframe for prediction
    fusion_prep (pd.DataFrame): The unmatched Fusion or Fusion dataframe for prediction

    Returns:
    list: A list of the skill scores generated from the model.
    """
    results = []
    for df_name, df in dfs.items():
        print(df_name)
        #Reset index for consistency and drop any duplicate columns as safety
        df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)
      
        
        metadataColumns = ['tss', 'parameter', 'units', 'SiteID', 'width', 'particle_size', 'time', 'date', 'Matchup', 'sat_cat', 'pixelCount','sample_method', 'analytical_method', 'lat', 'long', 'sceneID', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'qa', 'dswe', 'blue_sd', 'green_sd', 'red_sd', 'nir_sd', 'swir1_sd', 'swir2_sd', 'qa_sd', 'dswe_sd', 'hillshade', 'hillshadow', 'hillshadow_sd', 'azimuth', 'zenith', 'elevation', 'sat', 'NR', 'BR', 'GR', 'SR', 'BG', 'RG', 'NG', 'SG', 'BN', 'GN', 'RN', 'SN', 'BS', 'GS', 'RS', 'NS', 'R_GN', 'R_GB', 'R_GS', 'R_BN', 'R_BS', 'R_NS', 'G_BR', 'G_BN', 'G_BS', 'G_RN', 'G_RB', 'G_NS', 'B_RG', 'B_RN', 'B_RS', 'B_GN', 'B_GS', 'B_NS', 'N_RG', 'N_RB', 'N_RS', 'N_GB', 'N_GS', 'N_BS', 'GR2', 'GN2', 'RN2', 'BR_G', 'fai', 'N_S', 'N_R', 'ndvi', 'ndwi', 'ndssi', 'gn_gn', 'TZID', 'date_unity', 'type','NR_log','SR_log', 'BG_log', 'NG_log', 'SG_log', 'BN_log', 'GN_log', 'RN_log', 'SN_log', 'BS_log', 'GS_log', 'RS_log', 'NS_log', 'R_BN_log', 'R_NS_log', 'G_BN_log', 'G_RN_log', 'B_RN_log', 'B_GN_log', 'B_GS_log', 'B_NS_log', 'N_RG_log', 'N_RB_log', 'N_RS_log', 'N_GB_log', 'N_GS_log', 'N_BS_log', 'GN2_log', 'N_S_scaled']
                         

        # Assuming 'target' is the label we want to predict
        X = df.drop('tss_log', axis=1)
        y = df['tss_log']
        random_state = random_state
        # Manually and randomly split the data into training and testing
        num_samples = len(df)
        num_train_samples = int(train_size * num_samples)
        train_indices = np.array(np.random.choice(num_samples, num_train_samples, replace=False))
        test_indices = np.array([idx for idx in range(num_samples) if idx not in train_indices])
        
        X_trainAll, X_testAll = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        #print('X test shape:\n', X_testAll.shape[0])

        #Save full metadata set for adding data
        test_df_all = X_testAll.copy()
        test_df_all['tss_log'] = y_test
        test_df_all = test_df_all.reset_index(drop=True)
        
        #partition off rf columns in X train/test
        X_train = X_trainAll.drop(columns = metadataColumns)
        X_test = X_testAll.drop(columns = metadataColumns)
       # print('X test shape:\n', X_test.shape[0])
        print('X test columns:\n', X_test.columns.values.tolist())

        # Train a Random Forest model
        if df_name == 'S2':
            rf = RandomForestRegressor(max_depth=100, max_features=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200, random_state = random_state)
        elif df_name == 'Landsat':
            rf = RandomForestRegressor(n_estimators = 1000, min_samples_leaf = 2, max_features = None, max_depth = None, random_state = random_state)
        elif df_name == 'Fusion':
            rf = RandomForestRegressor(n_estimators = 1200, max_features = None, max_depth = 60, random_state = random_state)
        elif df_name == 'LS2Fusion':
            rf = RandomForestRegressor(max_depth=80, max_features=None, min_samples_split=5,
                      n_estimators=1000, random_state = random_state)
        
        #Fit the model
        rf.fit(X_train, y_train)
        
        # save the model to a file
        joblib.dump(rf, '/yourPath/' + experiment +'/' + stage + '/'+ test +'/models/' + str(random_state) + '_' + df_name + '_' + test + '_rfModel' + '.joblib')

        
        # Predict on the test set and then save it out for figures later
        
        test_df_all['tssPred_log'] = rf.predict(X_test)
        test_df_all['tssPred'] = round(10**(test_df_all['tssPred_log']), 4)
        test_df_all['tss'] = round(10**(test_df_all['tss_log']), 4)
        test_df_all['Experiment'] = df_name
        test_df_all['test'] = test
        test_df_all['RandomState'] = random_state
                
        #Save the test set for checking your work/custom evaluation and figures
        test_df_all.to_csv('/yourPath/' + experiment +'/' + stage + '/'+ test +'/testSets/' + experiment + '_' + test + '_' + df_name + '_' + str(random_state) +  '_testSet.csv', index=False)
       
        
        #METRICS
        #RMSE
        RMSE_model = np.sqrt(mean_squared_error(10**(y_test), 10**(prediction)))

        # Calculate the absolute errors
        maeTest = mean_absolute_error(10**(y_test), 10**(prediction))
        mapeTest = mean_absolute_percentage_error(10**(y_test), 10**(prediction))

        errors_med = np.median((abs(10**(prediction) - 10**(y_test))))
        mdape = (np.median(abs(10**(prediction) - 10**(y_test)) / abs(10**(y_test)))) * (100)


        #Find the relative error and bias (Dethier et al. 2018) = 10^(median(log10(abs(pred/meas))) - 1)
        relError_model = (10**(np.median(abs(np.log10(10**(prediction) / 10**(y_test))))) - 1)


        #RMSLE
        rmsle_test = np.sqrt(mean_squared_log_error(10**(y_test), 10**(prediction)))
        
        # Calculate mean of actual values
        mean_actual = np.mean(10**(y_test))

        def calc_rrmse(true, pred):
            num = np.sum(np.square(true - pred))
            den = np.sum(np.square(pred))
            squared_error = num/den
            rrmse_loss = np.sqrt(squared_error)
            return rrmse_loss
    
        rrmse = calc_rrmse(10**(y_test), 10**(prediction))
        nrmse = ((np.sqrt(mean_squared_error(10**(y_test), 10**(prediction)))) / (np.mean(10**(y_test))))

        # Calculate bias for each prediction
        biases = 10**(y_test) - 10**(prediction)
        log_biases = (y_test) - (prediction)
        
        mean_bias = np.mean(biases)
        median_bias = np.median(biases)
        
        log_mean_bias = 10**(np.mean((log_biases)))
        
        pbias_med = np.median((10**(y_test) - 10**(prediction)) / abs(10**(y_test)))
        pbias_mean = np.mean((10**(y_test) - 10**(prediction)) / abs(10**(y_test)))
        
        
        feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
        feat_importances.sort_values(ascending=True, inplace=True)

        
        # Create a DataFrame to store feature names and importances
        feat_importances_dict = feat_importances.to_dict()
        sorted_importances = dict(sorted(feat_importances_dict.items(), key=lambda item: item[1], reverse=True))
        # Choose the top 10 items
        top_10_importances = dict(list(sorted_importances.items())[:10])
        # Get the keys and values as lists
        keys = list(top_10_importances.keys())
        values = list(top_10_importances.values())
        
        

        # Store the results
        results.append({'Experiment': df_name,
              'Test': test,
              'RandomState' : round(random_state),
              'Test Rel Error': round(relError_model, 2),
              'Percent Bias_med': round(pbias_med, 2),
              'Percent Bias_mean': round(pbias_mean, 2),
              'Test RMSLE': round(rmsle_test, 2),
              'Training Size': X_train.shape[0],
              'Testing Size': X_test.shape[0],
              'Test MAE': round(maeTest, 2),
              'Test MAPE': round(mapeTest, 2),
              'Test MDAE': round(errors_med, 2),
              'Test MDAPE': round(mdape, 2),
              'Test RMSE': round(RMSE_model, 2),
              'Test NRMSE': round(nrmse, 2),
              'Test Bias_mean': round(mean_bias, 2),
              'Test Log Bias_mean': round(log_mean_bias, 2),
              keys[0] : round(values[0], 2),
              keys[1] : round(values[1], 2),
              keys[2] : round(values[2], 2),
              keys[3] : round(values[3], 2),
              keys[4] : round(values[4], 2),
              keys[5] : round(values[5], 2),
              keys[6] : round(values[6], 2),
              keys[7] : round(values[7], 2),
              keys[8] : round(values[8], 2),
              keys[9] : round(values[9], 2)
          })
        
        
#         #LS2 PREDICTION
        merged_pred_ls2 = predict_unmatched(df_name = df_name, rf = rf, X_test = X_test, preds_prepped_df = ls2_prep)
        merged_pred_ls2.to_csv('/yourPath/' + experiment +'/' + stage + '/'+ test +'/modelPreds_ls2/' + str(random_state) + '_' + df_name  + '_ls2Preds.csv', index=False)
        
        #FUSION PREDICTION
        merged_pred_fusion = predict_unmatched(df_name = df_name, rf = rf, X_test = X_test, preds_prepped_df = fusion_prep)
        merged_pred_fusion.to_csv('/yourPath/' + experiment +'/' + stage + '/'+ test +'/modelPreds_fusion/' + str(random_state) + '_' + df_name  + '_fusionPreds.csv', index=False)
        

    
    return results






##############################
# Functions to Evaluate Model
##############################

def plot_cdfs_to_image(dfs, metricList, colors, general_path, specific_path):
     """
    Plot cumulative distribution function of model scores across the 100 iterations.

    Parameters:
    dfs (dictionary): Dictionary of dataframes that will each be run through the RF
    metricList (list): A list of the relevant scores to be plotted with the same name as the results df
    colors (dictionary): A dictionary of the experiment and associated color for plotting
    general_path (string): For saving output
    specific_path (string): For saving plot and not overwriting output
    
    Returns:
    None
    
    Saves:
    One jpg file of all resulting CDFs from each experiment.
    """
    images = []  # List to store individual images

    for metricName in metricList:
        plt.figure()  # Create a new figure for each column
        plt.title(f'Cumulative Distribution Function (CDF) for {metricName}')

        for df_name, df in dfs.items():
            data = df[metricName]
            sorted_data = np.sort(data)
            y = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
            plt.plot(sorted_data, y, label=f'{df_name}', color=colors[df_name])

        plt.xlabel(f'{metricName}', fontsize=30)
        plt.ylabel('Proportion', fontsize=30)
        plt.xticks(rotation = 45, fontsize=25)
        plt.yticks(fontsize=25)
        #plt.title(f'{metricName} CDF')
        plt.grid()
        plt.legend()

        # Save the plot as an image
        image_path = f"{metricName}_CDFplot.png"
        plt.tight_layout()
        plt.savefig(general_path + image_path)
        plt.close()

        # Open the saved image and append to the list
        images.append(Image.open(general_path + image_path))

    # Concatenate images horizontally
    padding = 20
    concatenated_image = Image.new('RGB', (sum(img.width for img in images) + (len(images) - 1) * padding, max(img.height for img in images)))
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width + padding  # Increment by the width of the image plus padding

    # Save the final concatenated image
    concatenated_image.save(general_path + specific_path)

    print('done CDFs')



def feat_importance(dataframes, colors, general_path, specific_path):
    """
    Plots the mean values with error bars for each column from the 12th column onward
    for a dictionary of dataframes, and appends them to a single image.

    Parameters:
    dfs (dictionary): Dictionary of dataframes that will each be run through the RF
    colors (dictionary): A dictionary of the experiment and associated color for plotting
    general_path (string): For saving output
    specific_path (string): For saving plot and not overwriting output
    
    Returns:
    None
    
    Saves:
    One jpg file of all resulting CDFs from each experiment.
    """

    images = []  # List to store individual images

    for df_name, df in dataframes.items():
        # Drop columns with all NaN values
        df = df.dropna(axis=1, how='all').drop(['RandomState'], axis=1)
              
        # Calculate mean and standard deviation for each column from 12th column onward
        mean_values = df.iloc[:, 16:].mean()
        std_values = df.iloc[:, 16:].std()

        # Sort columns by mean values in descending order
        sorted_columns = mean_values.sort_values(ascending=False).index
        sorted_mean_values = mean_values[sorted_columns]
        sorted_std_values = std_values[sorted_columns]

        # Create positions for the bars
        x_pos = range(len(sorted_columns))

        # Plot the bar chart with error bars
        plt.bar(x_pos, sorted_mean_values, yerr=sorted_std_values, capsize=5)

        # Add column names as x-axis labels
        plt.xticks(x_pos, sorted_columns, rotation=45, ha='right')

        # Add labels and title
        plt.xlabel('Features')
        plt.ylabel('Mean Values')
        plt.title(f'Feature Importance - {df_name}')

        # Save the plot as an image
        image_path = f"{df_name}_plot.png"
        plt.tight_layout()
        plt.savefig(general_path + image_path)
        plt.close()

        # Open the saved image and append to the list
        images.append(Image.open(general_path + image_path))

    # Concatenate images horizontally
    concatenated_image = Image.new('RGB', (sum(img.width for img in images), max(img.height for img in images)))
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the final concatenated image
    concatenated_image.save(general_path + specific_path)

    print('done featImp plots')


def barPlots(df, metricList, colors, general_path, specific_path):
     """
    Plot model score distributions as bars with error across the 100 model iterations.

    Parameters:
    dfs (dictionary): Dictionary of dataframes that will each be run through the RF
    metricList (list): A list of the relevant scores to be plotted with the same name as the results df
    colors (dictionary): A dictionary of the experiment and associated color for plotting
    general_path (string): For saving output
    specific_path (string): For saving plot and not overwriting output
    
    Returns:
    None
    
    Saves:
    One jpg file of all resulting CDFs from each experiment.
    """
    # Pivot the DataFrame for 'Model', 'Test', and 'Stat'
    pivot_df = df.pivot_table(values=metricList, index=['Model'], columns='Stat')
    #print(pivot_df)
    images = []

    # Loop through each metric
    for metric in metricList:
        # Plot a grouped bar chart with std as error bars
        bar = plt.bar(pivot_df.index.get_level_values(0),pivot_df[metric, 'mean'], yerr=pivot_df[metric, 'std'],
                                            capsize=5, error_kw={'capsize': 5}, color=[colors[model] for model in pivot_df.index.get_level_values(0)])

        # Add counts above the two bar graphs
        for i, rect in enumerate(bar):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, pivot_df[metric, 'std'][i] + height, f'{height:.2f}', ha='center', va='bottom')

            
        # ax = pivot_df[metric, 'mean'].plot(kind='bar', figsize=(12, 6), yerr=pivot_df[metric, 'std'],
        #                                     capsize=5, error_kw={'capsize': 5}, color=[colors[model] for model in pivot_df.index.get_level_values(0)])

        plt.xlabel('Experiment', fontsize=25)
        plt.ylabel(f'{metric} Values', fontsize=25)
        #plt.title(f'Mean and Std for {metric} by Model', fontsize=30)
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()

        # Save the plot as an image
        image_path = f"{metric}_plot.png"
        plt.tight_layout()
        plt.savefig(general_path + image_path)
        plt.close()

        # Open the saved image and append to the list
        images.append(Image.open(general_path + image_path))

    # Concatenate images horizontally
    concatenated_image = Image.new('RGB', (sum(img.width for img in images), max(img.height for img in images)))
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the final concatenated image
    concatenated_image.save(general_path + specific_path)

    print('barPlot Metrics done')

def plot_obsPred(dfs, predColumn, obsColumn, colors, general_path, specific_path):
     """
    Plot model observed vs predicted distributions across the 100 model iterations.

    Parameters:
    dfs (dictionary): Dictionary of dataframes that will each be run through the RF
    predColumn (column variable): A specified column name that is the model predictions
    obsColumn (column variable): A specified column name that is the model observations from matchups
    colors (dictionary): A dictionary of the experiment and associated color for plotting
    general_path (string): For saving output
    specific_path (string): For saving plot and not overwriting output
    
    Returns:
    None
    
    Saves:
    One jpg file of all resulting CDFs from each experiment.
    """
    images = []  # List to store individual images

    for df_name, df in dfs.items():
        if predColumn in df.columns:
            plt.scatter((df[obsColumn]), (df[predColumn]), c=colors[df_name], label = df_name, alpha = 0.1, s = 5)
            p1 = max(max((df[predColumn])), max((df[obsColumn])))
            p2 = min(min((df[predColumn])), min((df[obsColumn])))
            plt.plot([p1, p2], [p1, p2], color='black', linestyle='-')
            plt.xlabel('Observed TSS (mg/L)', fontsize=25)
            plt.ylabel('Predicted TSS (mg/L)', fontsize=25)
            plt.tick_params(axis='x', labelsize=20)
            plt.tick_params(axis='y', labelsize=20)
            plt.axis('equal')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(False)
            plt.legend(loc= 'upper left', fontsize = 15)
            #plt.title('TSS Predicted vs. Observed', fontsize = 40)

            # Save the plot as an image
            image_path = f"{df_name}_OBsPredplot.png"
            plt.tight_layout()
            plt.savefig(general_path + image_path)
            plt.close()

            # Open the saved image and append to the list
            images.append(Image.open(general_path + image_path))

        # Concatenate images horizontally
        concatenated_image = Image.new('RGB', (sum(img.width for img in images), max((img.height for img in images), default=2)))
        x_offset = 0
        for img in images:
            concatenated_image.paste(img, (x_offset, 0))
            x_offset += img.width

    # Save the final concatenated image
    concatenated_image.save(general_path + specific_path)
    
    print('done obsPred')


def plot_bias(dfs, colors, general_path, specific_path):
     """
    Plot model bias across all iterations.

    Parameters:
    dfs (dictionary): Dictionary of dataframes that will each be run through the RF
    colors (dictionary): A dictionary of the experiment and associated color for plotting
    general_path (string): For saving output
    specific_path (string): For saving plot and not overwriting output
    
    Returns:
    None
    
    Saves:
    One jpg file of all resulting CDFs from each experiment.
    """
    images = []  # List to store individual images

    for df_name, df in dfs.items():
           
        bias = np.log10(df.tssPred) - np.log10(df.tss)
        
        plt.figure(figsize=(10, 6))

        plt.hist(bias, bins=1000, color=colors[df_name])
        plt.xlim(-250,250)
        #plt.xticks(range(-10, 10))
        plt.xlabel(df_name + ' Bias', fontsize=25)
        plt.ylabel('Count', fontsize=25)
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=15)
        plt.tight_layout()

        # Save the plot as an image
        image_path = f"{df_name}_biasplot.png"
        plt.tight_layout()
        plt.savefig(general_path + image_path)
        plt.close()

        # Open the saved image and append to the list
        images.append(Image.open(general_path + image_path))

    # Concatenate images horizontally
    concatenated_image = Image.new('RGB', (sum(img.width for img in images), max(img.height for img in images)))
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the final concatenated image
    concatenated_image.save(general_path + specific_path)
    
    print('done bias')
