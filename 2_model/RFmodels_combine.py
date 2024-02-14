#Combine all 100 iterations to single output with all data sources (matchup, wqp, predictions in one netCDF)

#module imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import glob
import joblib
import xarray as xr


def createBasin2(df):
    
    df['basin2'] = df['basin'].astype(str).str.rstrip('0').str.rstrip('.')
    df['basin4'] = df['basin'].astype(str).str.rstrip('0').str.rstrip('.')

    # specify the desired length
    length8 = 8
    length4 = 4
    length2 = 2
    # define lambda function to add a zero in front of values that aren't the desired length
    add_zero8 = lambda x: x if len(x) == length8 else '0'*(length8-len(x))+x
    add_zero2 = lambda x: x if len(x) == length2 else '0'*(length2-len(x))+x
    add_zero4 = lambda x: x if len(x) == length4 else '0'*(length4-len(x))+x

    # apply the lambda function to the column
    df['basin2'] = df['basin2'].apply(add_zero8)
    df['basin2'] = df['basin'].apply(lambda x: int(str(x)[:2]))
    df['basin2'] = df['basin2'].astype(str).apply(add_zero2)
    df['basin4'] = df['basin4'].apply(add_zero8)
    df['basin4'] = df['basin'].apply(lambda x: int(str(x)[:4]))
    df['basin4'] = df['basin4'].astype(str).apply(add_zero4)

    return df
# Function to map dates to seasons
def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'




def allTSS_preprocess(df):
    
    colors = {'WQP': 'red', 'Predicted Fusion': 'darkgreen', 'Matchup': 'orange', 'Predicted LS2': 'limegreen'} #, 'MatchupA': 'purple'
    df['color'] = df['dataSource'].map(colors)
    
    # Apply the function to create the 'season' column
    df['season'] = df['date'].apply(get_season)

    return df

#Draw in data sources, harmonize and clean
def read_harmonize(file_path, keyword, date, dataSource):

    file_list = [file for file in os.listdir(file_path) if keyword in file and file.endswith('.csv')]

    concatenated_df = pd.concat(
        [pd.read_csv(os.path.join(file_path, file), low_memory=False) for file in file_list],
        ignore_index=True)
    
    if dataSource == 'WQP':
        concatenated_df = concatenated_df.loc[(concatenated_df['units'] == 'mg/l')]
        concatenated_df = concatenated_df.drop_duplicates(subset = ['SiteID','date', 'tss'])
        concatenated_df['dataSource'] = dataSource
        
    if dataSource == "Predicted LS2" or dataSource == "Predicted Fusion": 
        concatenated_df['tss'] = concatenated_df['tssPred']
        concatenated_df['dataSource'] = dataSource
        
    if dataSource == 'Test Set':
        concatenated_df['tssObs'] = concatenated_df['tss']
        concatenated_df['tss'] = concatenated_df['tssPred']
        #concatenated_df['dataSource'] = 'Predicted ' + concatenated_df['Matchup'].astype(str)
        concatenated_df['dataSource'] = dataSource

    if dataSource == 'Matchup':
        concatenated_df['dataSource'] = dataSource
        concatenated_df = concatenated_df.drop_duplicates(['SiteID', 'date'])
    
    #After preprocessing
    concatenated_df = concatenated_df[['SiteID', 'tss', 'date', 'dataSource']] #'basin'
    concatenated_df['date'] = [x[:10] for x in concatenated_df['date']]
    concatenated_df['date'] = pd.to_datetime(concatenated_df['date'], format='ISO8601')
    concatenated_df = concatenated_df.loc[(concatenated_df['date'] >= date)]
    concatenated_df['year'] = concatenated_df.date.dt.year

    
    # Group by 'SiteID' and 'date', calculate mean and std for 'tss'
    grouped_df = concatenated_df.groupby(['SiteID', 'date', 'dataSource', 'year'])['tss'].agg(['mean', 'std', 'median','min', 'max']).reset_index() 
    
    return grouped_df


#################
#Read in datasets
#################
experiment = 'directory'
test = 'directory'
stage = 'directory' 

date = '2000-01-01'
#model = 'LS2Fusion' #unit test
models = ['S2', 'Landsat', 'Fusion', 'LS2Fusion']

print('Experiment:\n', experiment)
print('Test:\n', test)

general_path = '/yourPath/' + experiment +'/' + stage + '/'+ test

for model in models:
    print(model)
    ls2Pred_filePath = general_path + '/modelPreds_ls2/'
    dataSource = 'Predicted LS2'
    ls2pred = read_harmonize(ls2Pred_filePath, model, date, dataSource)


    fusionPred_filePath = general_path + '/modelPreds_fusion/'
    dataSource = 'Predicted Fusion'
    fusionpred = read_harmonize(fusionPred_filePath, model, date, dataSource)


    wqp_filePath = '/yourPathNoMatchups/'
    wqpKeyword = 'wqpNoMatch'
    dataSource = 'WQP'
    wqp = read_harmonize(wqp_filePath, wqpKeyword, date, dataSource)


    matchup_filePath = '/yourPath/'
    matchupKeyword = 'Matchups'
    dataSource = 'Matchup'
    matchups = read_harmonize(matchup_filePath, matchupKeyword, date, dataSource)

    #Put them all together
    allTss = pd.concat([ls2pred, fusionpred, matchups, wqp]).reset_index(drop=True) #testSet
    allTss = allTSS_preprocess(allTss)
    allTss['Experiment'] = model 

    #Convert DataFrame to xarray.Dataset
    ds = xr.Dataset.from_dataframe(allTss)

    # Save the dataset to a NetCDF file
    ds.to_netcdf(general_path + '/timeseries/' + model + '_timeseries.nc')


    #PLOT
    allYearMatchup = allTss.groupby(['year', 'dataSource'])['mean'].count().unstack()#.fillna(0)
    allYearMatchup.plot(kind='bar', stacked=True)
    plt.ylabel('Count', fontsize = '20')
    plt.xlabel('Year', fontsize = '20')
    plt.yticks(fontsize = 18)
    plt.xticks(fontsize=18)
    plt.title('Matchup Source by Year', pad = 10)
    #plt.tick_params(bottom=False, labelbottom=False)
    plt.legend(bbox_to_anchor=(0.8, 0.62), fontsize='14')


    # Save the figure as a JPEG file with the specified file path
    plt.savefig(general_path + '/' + model + 'plot.jpg',dpi=300, bbox_inches = "tight")

