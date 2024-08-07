# -*- coding: utf-8 -*-
"""
Author: Elisa Friedmann (efriedmann@umass.edu); credit to Ty Nietupski (Nietupski et al. 2021)

This script shows an example of how the functions in the prep_functions,
get_paired_collections, and core_functions scripts can be applied to predict
all images from a specified timeframe at many individual scene locations. To
use this code as a module, place the GEE_ImageFusion_Fxns directory in the same directory as
the main script.

Note that this script is for any images 2000-2023
Note that it is best to split pulls before and after Landsat 8 launch (2013-03-18) to prevent loading so many image collections at once
Note that you import geemap, which can be installed at: https://geemap.org/installation/
Note that some date ranges do not return enough resolved water pixels, there are no images between pairs after filtering, or user memory limit can be exceeeded during export etc.

General outline:
    1. Define global vars
    2. Slice sites into smaller groups and define date ranges with overlap to avoid exceeding the memory limit
    2. Organize dataset into nested lists
    3. Predict images.
        1. Register Landsat and MODIS
        2. Mask Landsat and MODIS and modify format for fusion (select similar
        pixels and convert images to 'neighborhood' images)
        3. Calculate the spatial and spectral distance to neighboring pixels
        and use to determine individual pixel weight.
        4. Perform regression over selected pixels to determine conversion
        coefficient.
        5. Use weights and conversion coefficient to predict images between
        image pairs.
    4. Export images as many small feature collections (.csv files) to local/Cloud storage
    
Required Inputs:
    1. GEE feature collection with the site name entitled 'SiteID' and its associated geometry
    2. Modify generalFilePath and fileIdentifier variables for export

"""

################################
#Imports
################################
import ee
ee.Authenticate()
ee.Initialize(project='yourProject')
import geemap
import os
import time
from datetime import datetime, timedelta

#Optional for if executing in colab cells
from GEE_ImageFusion_Fxns.core_functions import *
from GEE_ImageFusion_Fxns.get_paired_collections import *
from GEE_ImageFusion_Fxns.prep_functions import *



##########################################
#Scaling, Prediction, and Export Functions
##########################################

def overlappingDateRanges(start_date, end_date, time_span, overlap):
    """
    Generate a list of tuples with overlapping date ranges.

    Parameters:
    - start_date (str): Start date in the format 'YYYY-MM-DD'.
    - end_date (str): End date in the format 'YYYY-MM-DD'.
    - time_span (int): Number of days for each time span.
    - overlap (int): Number of days for overlap.

    Returns:
    - list: List of tuples with overlapping date ranges.
    """
    # Convert input strings to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Initialize list to store tuples of overlapping date ranges
    overlapping_ranges = []
    
    # Iterate through the date range with the specified time span
    while start_date < end_date:
        # Calculate the end date of the current time span
        time_span_end = start_date + timedelta(days=time_span)
        
        # Calculate the start date of the next time span
        next_start_date = time_span_end - timedelta(days=overlap)
        
        # Adjust the end date if it's past the end_date variable
        if time_span_end > end_date:
            time_span_end = end_date
        
        # Append tuple of overlapping date range to the list
        overlapping_ranges.append((start_date.strftime('%Y-%m-%d'), time_span_end.strftime('%Y-%m-%d')))
        
        # Update start date for next iteration
        start_date = next_start_date
    
    return overlapping_ranges


def batchSites(siteList, chunkSize):

    """
    Split a list into chunks of specified size.

    Parameters:
    - input_list (list): The input list to split into chunks.
    - chunk_size (int): The size of each chunk.

    Returns:
    - list: A list of chunks.
    """
    chunks = []
    for i in range(0, len(ee.List(siteList).getInfo()), chunkSize):
        chunk = siteList.slice(i,i + chunkSize)
        chunks.append(chunk)
    return chunks


def sites_processing(startDate, endDate):
    """
    Sort and organize all images and call the prediction functon.

    Parameters:
    - startDate (str): Start date in the format 'YYYY-MM-DD'.
    - endDate (str): End date in the format 'YYYY-MM-DD'.
    
    Returns:
    - ee.FeatureCollection: Feature Collection with SiteID and median band reflectances along with std. deviation.
    """
    def sortCols(site):
                
        region = ee.Feature(site).geometry()

        sortedImages = getPaired(startDate, endDate, ls789, ls57, 
              s2_sr_col, CLOUD_FILTER,
              landsatBands57, landsatBands89, bandNamesLandsat,
              S2Bands, bandNamesS2,
              modisCollection,
              commonBandNames,
              region, threshold) #s2_cloudless_col,

        #Create list of list of lists of groups of images for prediction
        subs = makeSubcollections(sortedImages)

        #Remove lists with no prediction images
        subs = subs.map(lambda el: ee.List(el).map(lambda subEl: ee.List(subEl).remove('null')).remove([]))

        #Turn into feature collection with size property to differentiate from complete and null image groups
        subsFeat = ee.FeatureCollection(ee.List(subs).map(lambda el: ee.Feature(None, {'subs': ee.List(el), 'subSize': ee.List(el).size()})))

        #Filter and remove null groups and extra properties
        subsFeat = subsFeat.filter(ee.Filter.gt('subSize', 2)).map(lambda feat: removeProperty(feat, 'subSize'))
                
        #return subsFeat
        return prediction(subsFeat, site) #, subList
    
    return sortCols


def prediction(subsFeat, site):
    """
    Three functions: Predict images, mask for water, and prepare as feature collection for export.

    Parameters:
    - subsFeat (ee.Feature Collection): A GEE feature collection built from the lists of groups of images to predict.
    - site (feature): The feature and coordinate location for generating fusion images.

    Returns:
    - ee.FeatureCollection: Feature Collection with SiteID and median band reflectances along with std. deviation.
    """
    
    def subPred(el): # list of lists of group of images to predict
        

        el = ee.List(el.toDictionary().values())
        sdist = ee.Feature(site).geometry().buffer(500)
        
        pred_group = ee.List(ee.List(ee.List(ee.List(el)).get(0)).get(0)) 
        #print(pred_group.getInfo())
        landsat_t01 = ee.List(ee.List(ee.List(ee.List(el)).get(0)).get(0))
        #print(landsat_t01.getInfo())
        modis_t01 = ee.List(ee.List(ee.List(ee.List(el)).get(0)).get(1))
        #print(landsat_t01.getInfo())
        modis_tp = ee.List(ee.List(ee.List(ee.List(el)).get(0)).get(2))
        #print(modis_tp.get(0).getInfo())

        origFine = ee.ImageCollection(ee.List(landsat_t01)).first()   

        # start and end day of group
        doys = landsat_t01 \
            .map(lambda img: ee.String(ee.Image(img).get('DOY')).cat('_'))

        # register images
        landsat_t01, modis_t01, modis_tp = registerImages(landsat_t01,
                                                          modis_t01,
                                                          modis_tp)

        # prep landsat imagery (mask and format)
        maskedLandsat, pixPositions, pixBN = prepLandsat(landsat_t01,
                                                         kernel,
                                                         numPixels,
                                                         commonBandNames,
                                                         doys,
                                                         coverClasses)  

        # prep modis imagery (mask and format)
        modSorted_t01, modSorted_tp = prepMODIS(modis_t01, modis_tp, kernel,
                                                numPixels, commonBandNames,
                                                pixBN)

        # calculate spectral distance
        specDist = calcSpecDist(maskedLandsat, modSorted_t01,
                                numPixels, pixPositions)

        # calculate spatial distance
        spatDist = calcSpatDist(pixPositions)

        # calculate weights from the spatial and spectral distances
        weights = calcWeight(spatDist, specDist)

        # calculate the conversion coefficients
        coeffs = calcConversionCoeff(maskedLandsat, modSorted_t01,
                                     doys, numPixels, commonBandNames)


        # predict all modis images in modis tp collection
        prediction = ee.List(modSorted_tp) \
            .map(lambda image:
                 predictLandsat(landsat_t01, modSorted_t01,
                                doys, ee.List(image),
                                weights, coeffs,
                                commonBandNames, numPixels)) 
        
        predsColl = ee.ImageCollection(ee.List(prediction)).map(lambda img: img.toFloat())
        

        def waterOnly(img):
            
            img = ee.Image(img)
            
            #Band quality thresholds
            white_pixels = 2000 #white pixels #2500 is max
            
                        
            #create band theshold masks
            bright_pixels = img.select('red').lt(white_pixels).And(img.select('blue').lt(white_pixels))\
                                .And(img.select('green').lt(white_pixels))

            #mask the image collection to keep only water pixels
            road = ee.FeatureCollection('TIGER/2016/Roads').filterBounds(ee.Feature(site).geometry().buffer(250)).geometry().buffer(30)
            roadMask = ee.Image(0).paint(road,1).Not()

            d = Dswe(img).select('dswe')

            masked = img \
                    .updateMask(roadMask)\
                    .addBands(d).updateMask(d.eq(1).Or(d.eq(2)))\
                    .updateMask(bright_pixels)\
                    .set({'date': img.date().format('YYYY-MM-dd')}).toInt16() 
            

            reducers = ee.Reducer.median().combine(ee.Reducer.stdDev(), "", True)
            previous = ee.List(['blue_stdDev', 'green_stdDev', 'red_stdDev', 'nir_stdDev', 'swir1_stdDev', 
                                'swir2_stdDev',
            'blue_median', 'green_median', 'red_median', 'nir_median', 'swir1_median', 'swir2_median']) #
            to = ee.List(['blue_sd', 'green_sd', 'red_sd', 'nir_sd', 'swir1_sd', 'swir2_sd',
             'blue', 'green', 'red', 'nir', 'swir1', 'swir2']) #

            reduceRegion = masked.reduceRegion(**{'reducer' : reducers, 
                                                  'geometry' : sdist,
                                                  'scale' : 30,
                                                  'crs': origFine.projection(),
                                                  'tileScale': 8}).rename(previous, to)


            output = ee.Feature(site)\
                        .set({"blue": reduceRegion.get('blue')})\
                        .set({"green": reduceRegion.get('green')})\
                        .set({"red": reduceRegion.get('red')})\
                        .set({"nir": reduceRegion.get('nir')})\
                        .set({"swir1": reduceRegion.get('swir1')})\
                        .set({"swir2": reduceRegion.get('swir2')})\
                        .set({"swir1_sd": reduceRegion.get('swir1_sd')})\
                        .set({"blue_sd": reduceRegion.get('blue_sd')})\
                        .set({"green_sd": reduceRegion.get('green_sd')})\
                        .set({"red_sd": reduceRegion.get('red_sd')})\
                        .set({"nir_sd": reduceRegion.get('nir_sd')})\
                        .set({"swir2_sd": reduceRegion.get('swir2_sd')})\
                        .set({"date": masked.get('date')})\
                        .set({"pixelCount": masked.reduceRegion(ee.Reducer.count(), 
                                sdist, 30).get('red')}) #  .set({'dswe': reduceRegion.get('dswe')})\
                        #.set({'dswe_sd': reduceRegion.get('dswe_sd')})\
            
            return ee.FeatureCollection(output).map(removeGeometry)
        
        
        return ee.ImageCollection(predsColl).map(waterOnly).flatten() 
    
    return ee.FeatureCollection(subsFeat).map(subPred).flatten().filter(ee.Filter.gt('pixelCount', 0))\
                .filter(ee.Filter.gt('swir2', 0))\
                .filter(ee.Filter.gt('red', 0))\
                .filter(ee.Filter.gt('blue', 0))\
                .filter(ee.Filter.gt('green', 0))\
                .filter(ee.Filter.gt('nir', 0))\
                .filter(ee.Filter.gt('swir1', 0))


def fusionExport(fcList, dates, generalFilePath, identifier, siteNum, chunkCount):
    """
    Generate a csv of median band reflectances per date per site and export

    Parameters:
    - fcList (list): The site of interest input as a list of features.
    - dates (tuple): The dates of interest to run fusion over.
    - generalFilePath (str): File path to save the prepared csv.
    - identifier (str): The name of the run (i.e. Mississippi_Basin).
    - siteNum (int): The number of sites in each batch (removes a .getInfo() call).
    - chunkCount (int): The batch number to prevent file overwrite.

    Returns:
    - Feature Collection of median band reflectances and their standard deviations for a single date range per site    
    
    """
    for d in dates:
        for i in range(0, siteNum): 
                       
            startDate = str(d[0])
            endDate = str(d[1])
            site = ee.Feature(fcList.get(i))

            try:
                #Call sorting and predicting functions
                results = ee.FeatureCollection(site).map(sites_processing(startDate=startDate, endDate=endDate)).flatten()

                #print(results.getInfo())

                filepath =  generalFilePath + str(i) + '_' + str(chunkCount) + '_' + startDate + '-' + endDate + '_' + str(identifier) + '.csv'
                geemap.ee_export_vector(ee_object = ee.FeatureCollection(results), filename = filepath, selectors = ['SiteID','date','blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixelCount',
                                    'blue_sd', 'green_sd', 'red_sd', 'nir_sd', 'swir1_sd', 'swir2_sd'])
            
            except Exception as error:
                print("An error occurred:", error)
                pass
#Without geemap:        
#             task = ee.batch.Export.table.toDrive(
#                         collection = ee.FeatureCollection(results), 
#                         description = str(i) + '_' + startDate + '-' + endDate + str(identifier) + str(siteSpan), 
#                         folder = 'MasterTest' + str(identifier), 
#                         fileFormat = 'csv',
#                         selectors = ['SiteID','date',
#                                     'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixelCount',
#                                     'blue_sd', 'green_sd', 'red_sd', 'nir_sd', 'swir1_sd', 'swir2_sd'])

            #maximum tasks function(max running jobs, waiting period between checking to submit more)
            #import time
            #maximum_no_of_tasks(2999, 45)

            #Start the task
            #task.start()
            #while task.active():
                #print('Polling for task (id: {}).'.format(task.id), i, startDate, endDate, str(identifier), str(siteSpan))
                #time.sleep(5)
            
            print('done', i, startDate, endDate, str(identifier))
            time.sleep(5)


#################################
#Global Variables and Collections
#################################

# define sites of interest
#Example: 
#https://code.earthengine.google.com/?asset=projects/fusion-353005/assets/fusionSites2/fusionSitesAll_01232024


# define special start and end dates for collections
startDate_ls7 = ee.String('2000-01-01')
endDate_ls7 = ee.String('2003-05-31'); # End date of ls7 full images #option 1
endDate_ls7_S2 = ee.String('2021-10-31'); # End date of ls7 images due to Landsat 9 - option 2
endDate_ls5 = ee.String('2012-05-05')

# Cloud cover threshold for region (percent)
threshold = ee.Number(5)
thresholdS = ee.String('5')

#For DSWE water function
road = ee.FeatureCollection("TIGER/2016/Roads")


#  landsat band names including qc band for masking
bandNamesLandsat = ee.List(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'qa'])

landsatBands57 = ee.List(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'])
landsatBands89 = ee.List(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'])

# Sentinel Band Names
S2Bands = ['B2','B3','B4','B8','B11','B12', 'QA60']
bandNamesS2 = ['blue','green','red','nir','swir1','swir2', 'QA60']

#  modis band names
bandNamesModisT = ee.List(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'qa'])
modisBandsT = ee.List(['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06', 
                       'sur_refl_b07', 'state_1km'])
bandNamesModisA = ee.List(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'qa'])
modisBandsA = ee.List(['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06', 
                       'sur_refl_b07', 'state_1km'])

commonBandNames = ee.List(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']) #'r_gb', 


#Call in feature collections and trim
modisCollection = ee.ImageCollection('MODIS/061/MOD09GA')\
    .select(modisBandsT, bandNamesModisT)\
    .merge(ee.ImageCollection("MODIS/061/MYD09GA").select(modisBandsA, bandNamesModisA))\
    .filterBounds(conusSites.geometry())

ls9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")\
        .filterBounds(conusSites.geometry())\
        .select(landsatBands89, bandNamesLandsat)

ls8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')\
        .filterBounds(conusSites.geometry())\
        .select(landsatBands89, bandNamesLandsat)

ls7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')\
    .filterBounds(conusSites.geometry())\
    .filterDate(startDate_ls7, endDate_ls7_S2)\
    .select(landsatBands57, bandNamesLandsat)
    
ls5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')\
        .filterBounds(conusSites.geometry())\
        .select(landsatBands57, bandNamesLandsat)

ls789 = ee.ImageCollection(ls9.merge(ls8).merge(ls7))
ls57 = ee.ImageCollection(ls5.merge(ls7))

s2_sr_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")


#S2 Cloudless params
CLOUD_FILTER = 80
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 75

# radius of moving window
# Note: Generally, larger windows are better but as the window size increases,
# so does the memory requirement and we quickly will surpass the memory
# capacity of a single node (in testing 13 was max size for single band, and
# 10 was max size for up to 6 bands)
kernelRadius = ee.Number(8)
kernel = ee.Kernel.square(kernelRadius)
numPixels = kernelRadius.add(kernelRadius.add(1)).pow(2)

# number of land cover classes in scene
coverClasses = 6


##########################################
#Batching and Function Call
##########################################

#Sites to list (called from global vars above)
sitesList = conusSites.toList(8000).slice(0,50) #convert to list for scaling

#Where to save your csv files
generalFilePath = '/yourPath/'


# Chunk size/Number of sites in each batch, define for computational feasibility
batchSiteNum = 23

# Example dates:
start_date = '2021-05-01'
end_date = '2021-11-15'
time_span = 60
overlap = 16
fileIdentifier = 'TEST' #prevents file overwrites


# Generate list of overlapping date ranges
overlappingDates = overlappingDateRanges(start_date=start_date, end_date=end_date, time_span=time_span, overlap=overlap)

# Split the list into chunks
chunks = batchSites(siteList=sitesList, chunkSize = batchSiteNum)

#CALL FUSION PREDICTION FUNCTIONS AND EXPORT
for count, chunk in enumerate(chunks):
    fusionExport(fcList = chunk, dates = overlappingDates, generalFilePath = generalFilePath, identifier = fileIdentifier, siteNum = batchSiteNum, chunkCount = count)

