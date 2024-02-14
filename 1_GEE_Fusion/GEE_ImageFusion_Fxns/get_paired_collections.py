# -*- coding: utf-8 -*-
import ee

# MASKING, INDEX CALCULATION, L5 & L7 TO L8 & L9 HARMONIZATION


#  Landsat8 SR scaling function. Applies scaling factors.
def applyScaleFactors(image):
    image = ee.Image(image)
    opticalBands = image.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).multiply(0.0000275).add(-0.2).multiply(10000)
    return image.addBands(opticalBands, None, True).toFloat() \
                .copyProperties(image) #.multiply(10000)

# S2 SR scaling function
def applyScaleS2(image):
    image = ee.Image(image)
    opticalBands = image.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).divide(10000).multiply(10000)
    return image.addBands(opticalBands, None, True).toFloat() \
                .copyProperties(image) #.multiply(10000)


#  Landsat


def fmask89(region):

    def allMasks89(img):
        # Bit 0 - Fill
        # Bit 1 - Dilated Cloud
        # Bit 2 - Unused
        # Bit 3 - Cloud
        # Bit 4 - Cloud Shadow
        # Bit 5 - Snow
        cloudShadowBitMask = 1 << 3
        cloudsBitMask = 1 << 5
        cirrusBitMask = 1 << 2
        #snowBitMask = 1 << 4
        
        
          # Get the pixel QA band.
        qa = img.select('QA_PIXEL')
        site = region.buffer(1000)

        #mask the image collection to keep only water pixels
#         water = gsw.select('occurrence').gte(10);
#         wateronly = water.clip(site)
        
            # Flags should be set to zero, indicating clear conditions.
        mask = (qa.bitwiseAnd(cloudShadowBitMask).eq(0)) \
              .And(qa.bitwiseAnd(cloudsBitMask).eq(0)) \
              .And(qa.bitwiseAnd(cirrusBitMask).eq(0)) #\
              #.And(qa.bitwiseAnd(snowBitMask).eq(0))


        #  mask the mask with the mask...
        maskedMask = mask.updateMask(mask)

        #  count the number of nonMasked pixels
        maskedCount = maskedMask.select(['QA_PIXEL']) \
            .reduceRegion(reducer=ee.Reducer.count(),
                          geometry=site,
                          scale=ee.Number(30),
                          maxPixels=ee.Number(4e10))

        #  count the total number of pixels
        origCount = img.select(['blue']) \
            .reduceRegion(reducer=ee.Reducer.count(),
                          geometry=site,
                          scale=ee.Number(30),
                          maxPixels=ee.Number(4e10))

        #  calculate the percent of masked pixels
        percent = ee.Number(origCount.get('blue')) \
            .subtract(maskedCount.get('QA_PIXEL')) \
            .divide(origCount.get('blue')) \
            .multiply(100) \
            .round()

        #  Return the masked image with new property and time stamp
        return img.toFloat() \
            .set('CloudSnowMaskedPercent', percent) \
            .set('OrigPixelCount', origCount.get('blue'))\
            .set({'MEAN_SOLAR_AZIMUTH_ANGLE': img.get('SUN_AZIMUTH')})\
            .set({'MEAN_SOLAR_ZENITH_ANGLE': img.get('SUN_ELEVATION')})\
            .set({'Mission': img.get('SPACECRAFT_ID')})\
            .set({'date': img.get('DATE_PRODUCT_GENERATED')})\
            .set({'Scene_ID': img.get('LANDSAT_SCENE_ID')})
            #.copyProperties(img) #.updateMask(mask)
    
    return allMasks89

# Mask cloud, cloud shadow, and snow in Landsat 5, 7 images
def fmask57(region):

    def allMasks57(img):
        # Bit 0 - Fill
        # Bit 1 - Dilated Cloud
        # Bit 2 - Unused
        # Bit 3 - Cloud
        # Bit 4 - Cloud Shadow
        # Bit 5 - Snow
        cloudShadowBitMask = 1 << 3
        cloudsBitMask = 1 << 5
        cirrusBitMask = 1 << 2
        #snowBitMask = 1 << 4
        
          # Get the pixel QA band.
        qa = img.select('QA_PIXEL')

          # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
              .And(qa.bitwiseAnd(cloudsBitMask).eq(0)) \
              .And(qa.bitwiseAnd(cirrusBitMask).eq(0)) #\
              #.And(qa.bitwiseAnd(snowBitMask).eq(0))

        site = region.buffer(1000)

        #  mask the mask with the mask...
        maskedMask = mask.updateMask(mask)

        #  count the number of nonMasked pixels
        maskedCount = maskedMask.select(['QA_PIXEL']) \
            .reduceRegion(reducer=ee.Reducer.count(),
                          geometry=site,
                          scale=ee.Number(30),
                          maxPixels=ee.Number(4e10))

        #  count the total number of pixels
        origCount = img.select(['blue']) \
            .reduceRegion(reducer=ee.Reducer.count(),
                          geometry=site,
                          scale=ee.Number(30),
                          maxPixels=ee.Number(4e10))

        #  calculate the percent of masked pixels
        percent = ee.Number(origCount.get('blue')) \
            .subtract(maskedCount.get('QA_PIXEL')) \
            .divide(origCount.get('blue')) \
            .multiply(100) \
            .round()

        #  Return the masked image with new property and time stamp
        return img.toFloat()\
            .set('CloudSnowMaskedPercent', percent) \
            .set('OrigPixelCount', origCount.get('blue'))\
            .set({'MEAN_SOLAR_AZIMUTH_ANGLE': img.get('SUN_AZIMUTH')})\
            .set({'MEAN_SOLAR_ZENITH_ANGLE': img.get('SUN_ELEVATION')})\
            .set({'Mission': img.get('SPACECRAFT_ID')})\
            .set({'date': img.get('DATE_PRODUCT_GENERATED')})\
            .set({'Scene_ID': img.get('LANDSAT_SCENE_ID')})\
            #.copyProperties(img) #.updateMask(mask)

    return allMasks57

#
#  Sentinel
#

# Combine s2 and s2 cloudless collections
def get_s2_sr_cld_col(aoi, start_date, end_date, CLOUD_FILTER):
    #  Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    #  Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date))

    #  Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

# Add cloud band to combined S2 collection
def add_cloud_bands(img):
    #  Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    #  Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    #  Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

# Add shadow band to combined S2 collection
def add_shadow_bands(img):
    #  Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    #  Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    #  Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    #  Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10) \
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100}) \
        .select('distance') \
        .mask() \
        .rename('cloud_transform'))

    #  Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    #  Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

# Assemble all cloud and cloud shadow components for final mask
def add_cld_shdw_mask(img):
    #  Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    #  Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    #  Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    #  Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    #  20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20) \
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20}) \
        .rename('cloudmask'))

    #  Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

# # Define a funciton to apply the cloud mask to each image in the collection - incorporated in maskS2clouds
# function apply_cld_shdw_mask(img) {
#     # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
#     not_cld_shdw = img.select('cloudmask').Not()

#     #  Subset reflectance bands and update their masks, return the result.
#     return img.select(['B.*', 'QA60']).updateMask(not_cld_shdw)
# }

#   Function to mask clouds using the Sentinel-2 QA band
#   @param {ee.Image} image Sentinel-2 image
#   @return {ee.Image} cloud masked Sentinel-2 image

def maskS2clouds(region):
    def allMasksS2(img):

        # Buffer site
        site = region.buffer(1000)
        qa = img.select('QA60')

        #  Bits 10 and 11 are clouds and cirrus, respectively.
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11

        #  Both flags should be set to zero, indicating clear conditions.
        mask = (qa.bitwiseAnd(cloudBitMask).eq(0)) \
           .And(qa.bitwiseAnd(cirrusBitMask).eq(0))

        # # Incorporate s2_cloudless functions
        #add_cld_shdw_bands = add_cld_shdw_mask(img)

        #not_cld_shdw = add_cld_shdw_bands.select('cloudmask').Not()


        #  mask the mask with the mask...
        maskedMask = mask.updateMask(mask)#.updateMask(not_cld_shdw)

        #  count the number of nonMasked pixels
        maskedCount = maskedMask.select(['QA60']) \
            .reduceRegion(reducer = ee.Reducer.count(),
                        geometry = site,
                        scale = ee.Number(30),
                        maxPixels = ee.Number(4e10))

        #  count the total number of pixels
        origCount = img.select(['B2']) \
            .reduceRegion(reducer = ee.Reducer.count(),
                          geometry = site,
                          scale= ee.Number(30),
                          maxPixels= ee.Number(4e10))

        #  calculate the percent of masked pixels
        percent = ee.Number(origCount.get('B2')) \
            .subtract(maskedCount.get('QA60')) \
            .divide(origCount.get('B2')) \
            .multiply(100) \
            .round()
        
        return img.select(['B.*', 'QA60']).toFloat() \
            .set('CloudSnowMaskedPercent', percent) \
            .set('OrigPixelCount', origCount.get('B2'))\
            .set({'Mission': img.get('SPACECRAFT_NAME')})\
            .set({'date': img.get('GENERATION_TIME')})\
            .set({'Scene_ID': img.get('GRANULE_ID')})\
            .set({'WRS_PATH': img.get('MGRS_TILE')})\
            .set({'WRS_ROW': img.get('MGRS_TILE')})#\
            #.copyProperties(img) #.updateMask(not_cld_shdw).updateMask(mask)
    
    return allMasksS2

#
#  MODIS
# 

def maskMODIS(region):
    def allMasksModis(img):
        # Buffer site
        site = region.buffer(1000)

        shadowBitMask = 4;
        adjacentBitMask = 1 << 13;
        cloudInternalBitMask = 1 << 10;
        #snowBitMask = 1 << 15;
    
        # Get the pixel QA band.
        qa = img.select('QA_PIXEL')
  
        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(shadowBitMask).eq(0)\
            .And(qa.bitwiseAnd(adjacentBitMask).eq(0))\
            .And(qa.bitwiseAnd(cloudInternalBitMask).eq(0))#\
            #.And(qa.bitwiseAnd(snowBitMask).eq(0))
        
        #  mask the mask with the mask...
        maskedMask = mask.updateMask(mask)
        
        #  count the number of nonMasked pixels
        maskedCount = maskedMask.select(['QA_PIXEL']) \
            .reduceRegion(reducer = ee.Reducer.count(),
                        geometry = site,
                        scale = ee.Number(500),
                        maxPixels = ee.Number(4e10))

        #  count the total number of pixels
        origCount = img.select(['blue']) \
            .reduceRegion(reducer = ee.Reducer.count(),
                          geometry = site,
                          scale= ee.Number(500),
                          maxPixels= ee.Number(4e10))

        #  calculate the percent of masked pixels
        percent = ee.Number(origCount.get('blue')) \
            .subtract(maskedCount.get('QA_PIXEL')) \
            .divide(origCount.get('blue')) \
            .multiply(100) \
            .round()
        
        return img \
            .updateMask(mask) \
            .set('CloudSnowMaskedPercent', percent) \
            .set('Scene_ID', img.get('system:time_start'))\
            .set('OrigPixelCount', origCount.get('blue'))\
            .copyProperties(img)
    
    return allMasksModis



# Function combine, and sort fine and coarse images
def getPaired(startDate, endDate, ls89, ls57, 
              s2_sr_col, CLOUD_FILTER, 
              landsatBands57, landsatBands89, bandNamesLandsat,S2Bands, bandNamesS2,
              modisCollection,
              commonBandNames,region, threshold): # s2_cloudless_col,

   
    if startDate >= '2013-03-18':
        #Landsat 8 and Landsat 9
        col89 = ee.ImageCollection(ls89)\
                    .filterDate(startDate, endDate) \
                    .filterBounds(region)\
                    .map(fmask89(region)) \
                    .filter(ee.Filter.lte('CloudSnowMaskedPercent', threshold))\
                    .filter(ee.Filter.gt('OrigPixelCount', 0))\
                    .map(lambda image: image \
                         .setMulti({\
                             'system:time_start':\
                                 ee.Date(image.date().format('y-M-d'))\
                                 .millis(),\
                             'DOY': image.date().format('D')\
                             })) \
                    .select(commonBandNames) \
                    .map(lambda image: image.clip(region.buffer(1000)))\
                    .map(applyScaleFactors)\
                    .map(lambda image: image.toFloat())\
                    .sort('system:time_start') #.map(applyScaleFactors)\ .map(addR_GB)\
        
        #print(col89.first().getInfo())
        
        #Sentinel-2
    
        
        #s2_sr_cloudless = get_s2_sr_cld_col(region, startDate, endDate, CLOUD_FILTER)
        
        s2 = ee.ImageCollection(s2_sr_col)\
            .filterBounds(region)\
            .filterDate(startDate, endDate)\
            .map(maskS2clouds(region)) \
            .select(S2Bands, bandNamesS2) \
            .filter(ee.Filter.lte('CloudSnowMaskedPercent', threshold))\
            .filter(ee.Filter.gt('OrigPixelCount', 0))\
            .map(lambda image: image.setDefaultProjection(**{'crs': (col89.filterBounds(region).first().projection()), 
                                                  'scale': ee.Number(30)})\
                .setMulti({
                     'system:time_start':
                         ee.Date(image.date().format('y-M-d')).millis(),
                     'DOY': image.date().format('D')
                     })) \
            .select(commonBandNames)\
            .map(lambda image: image.clip(col89.first().geometry()).toFloat()) \
            .sort('system:time_start') #.map(addR_GB)\
        
        #print(s2.first().getInfo())
        
        fine = ee.ImageCollection(col89).merge(ee.ImageCollection(s2)).sort('system:time_start', True) #.merge(ee.ImageCollection(s2))

        #  For some tiles, Sentinel images from 2A and 2B can occur on the same day, or concurrent with Landsat
        #  which must be filtered out and if taken near the same time, act as validation
        fineDistinct = fine.distinct('system:time_start')

        #print('ORIG Fine # images', fineDistinct.size().getInfo())

        # Define an equals filter to match collections
        dateFilter = ee.Filter.equals(**{
            'leftField': 'system:time_start',
            'rightField': 'system:time_start'
              })

        # Define inverted join to separate and save only the duplicates
        invertedJoin = ee.Join.inverted()

        # Apply the join
        fineDup = ee.ImageCollection(invertedJoin.apply(fine, fineDistinct, dateFilter))

        #  get modis images
        modis = ee.ImageCollection(ee.ImageCollection(modisCollection)\
                  .filterDate(startDate, endDate) \
                  .filterBounds(region)\
                  .map(maskMODIS(region))\
                  .filter(ee.Filter.lte('CloudSnowMaskedPercent', threshold))\
                  .filter(ee.Filter.gt('OrigPixelCount', 0))\
                  .map(lambda image: image.clip(fine.first().geometry())) \
                  .map(lambda image: image.set('DOY', image.date().format('D'))) \
                  .select(commonBandNames).distinct('system:time_start')) \
                  .sort('system:time_start') # .map(addR_GB)\            


        #print('ORIG MODIS # images', modis.first().getInfo())

        #  filter the two collections by the date property
        dayfilter = ee.Filter.equals(**{'leftField':'system:time_start',
                                     'rightField':'system:time_start'})

        #  define simple join
        pairedJoin = ee.Join.simple()
        #  define inverted join to find modis images without landsat pair
        invertedJoin = ee.Join.inverted()

        #  create collections of paired landsat and modis images
        landsatPaired = pairedJoin.apply(fineDistinct, modis, dayfilter)
        modisPaired = pairedJoin.apply(modis, fineDistinct, dayfilter)

        #Filter collection for non-consecutive paired landsat/modis images and store as a validation collection
        def dateDiffProp(img):
            index = lst.indexOf(img)
            img = ee.Image(img)
            imgDate = img.date().format('yyyy-MM-dd')
            previousIndex = ee.Algorithms.If(index.eq(0), index, index.subtract(1))
            prevImgDate = ee.Image(lst.get(previousIndex)).date().format('yyyy-MM-dd')
            diff = ee.Number(ee.Date(imgDate).difference(ee.Date(prevImgDate), 'day'))
            diffProp = ee.Image(img).set('DateDiff', diff)
            return diffProp

        sortImgLength_0 = landsatPaired.size()
        lst = landsatPaired.toList(100000)
        dateProp_0 = ee.ImageCollection(lst.map(dateDiffProp))

        fineImgCons = dateProp_0.filter(ee.Filter.eq('DateDiff', 1))
        landsatPaired = dateProp_0.filter(ee.Filter.neq('DateDiff', 1))

        sortImgLength_1 = modisPaired.size()
        lst = modisPaired.toList(100000)
        dateProp_1 = ee.ImageCollection(lst.map(dateDiffProp))

        coarseImgCons = dateProp_1.filter(ee.Filter.eq('DateDiff', 1))
        modisPaired = dateProp_1.filter(ee.Filter.neq('DateDiff', 1))

        # Merge duplicates (fine) and consecutives (fine and coarse) to one image collection
        #valImgs = fineImgCons.merge(coarseImgCons).merge(fineDup)

        # Create collection of unpaired modis images
        # Use the list of distinct, non-consecutive fine images
        fineDistinctLength = fineDistinct.size()
        lst = fineDistinct.toList(100000)
        fineDist_dateProp = ee.ImageCollection(lst.map(dateDiffProp))

        fineDist_dateProp = fineDist_dateProp.filter(ee.Filter.neq('DateDiff', 1))

        # Create collection of unpaired modis images between the fine/coarse pairs
        modisUnpaired = invertedJoin.apply(modis, fineDist_dateProp, dayfilter)


    else:
        col57 = ee.ImageCollection(ls57)\
                    .filterDate(startDate, endDate) \
                    .filterBounds(region)\
                    .map(lambda image: image.clip(region.buffer(1000)))\
                    .map(fmask89(region)) \
                    .filter(ee.Filter.lte('CloudSnowMaskedPercent', threshold))\
                    .filter(ee.Filter.gt('OrigPixelCount', 0))\
                    .map(lambda image: image \
                         .setMulti({\
                             'system:time_start':\
                                 ee.Date(image.date().format('y-M-d'))\
                                 .millis(),\
                             'DOY': image.date().format('D')\
                             })) \
                    .select(commonBandNames) \
                    .map(applyScaleFactors)\
                    .sort('system:time_start') #.map(applyScaleFactors)\ #.map(addR_GB)\
          
        fine = col57  
    
        
        #Get extra images if any
        fineDistinct = fine.distinct('system:time_start')
    
        #print('ORIG Fine # images', fineDistinct.size().getInfo())
    
                # Define an equals filter to match collections
        dateFilter = ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
              })

        # Define inverted join to separate and save only the duplicates
        invertedJoin = ee.Join.inverted()

        # Apply the join
        fineDup = ee.ImageCollection(invertedJoin.apply(fine, fineDistinct, dateFilter))

            
            
                    #  get modis images
        modis = ee.ImageCollection(ee.ImageCollection(modisCollection)\
                  .filterDate(startDate, endDate) \
                  .filterBounds(fine.first().geometry())\
                  .map(maskMODIS(region))\
                  .filter(ee.Filter.lte('CloudSnowMaskedPercent', threshold))\
                  .filter(ee.Filter.gt('OrigPixelCount', 0))\
                  .map(lambda image: image.clip(fine.first().geometry())) \
                  .map(lambda image: image.set('DOY', image.date().format('D'))) \
                  .select(commonBandNames).distinct('system:time_start')) \
                  .sort('system:time_start') #.map(addR_GB)\


        #print('ORIG MODIS # images', modis.size().getInfo())

                #  filter the two collections by the date property
        dayfilter = ee.Filter.equals(**{'leftField':'system:time_start',
                                     'rightField':'system:time_start'})

        #  define simple join
        pairedJoin = ee.Join.simple()
        #  define inverted join to find modis images without landsat pair
        invertedJoin = ee.Join.inverted()

        #  create collections of paired landsat and modis images
        landsatPaired = pairedJoin.apply(fineDistinct, modis, dayfilter)
        modisPaired = pairedJoin.apply(modis, fineDistinct, dayfilter)

        #Filter collection for non-consecutive paired landsat/modis images and store as a validation collection
        def dateDiffProp(img):
            index = lst.indexOf(img)
            img = ee.Image(img)
            imgDate = img.date().format('yyyy-MM-dd')
            previousIndex = ee.Algorithms.If(index.eq(0), index, index.subtract(1))
            prevImgDate = ee.Image(lst.get(previousIndex)).date().format('yyyy-MM-dd')
            diff = ee.Number(ee.Date(imgDate).difference(ee.Date(prevImgDate), 'day'))
            diffProp = ee.Image(img).set('DateDiff', diff)
            return diffProp

        sortImgLength_0 = landsatPaired.size()
        lst = landsatPaired.toList(1000000)
        dateProp_0 = ee.ImageCollection(lst.map(dateDiffProp))

        fineImgCons = dateProp_0.filter(ee.Filter.eq('DateDiff', 1))
        landsatPaired = dateProp_0.filter(ee.Filter.neq('DateDiff', 1))

        sortImgLength_1 = modisPaired.size()
        lst = modisPaired.toList(1000000)
        dateProp_1 = ee.ImageCollection(lst.map(dateDiffProp))

        coarseImgCons = dateProp_1.filter(ee.Filter.eq('DateDiff', 1))
        modisPaired = dateProp_1.filter(ee.Filter.neq('DateDiff', 1))

        # Merge duplicates (fine) and consecutives (fine and coarse) to one image collection
        #valImgs = fineImgCons.merge(coarseImgCons).merge(fineDup)

        # Create collection of unpaired modis images
        # Use the list of distinct, non-consecutive fine images
        fineDistinctLength = fineDistinct.size()
        lst = fineDistinct.toList(1000000)
        fineDist_dateProp = ee.ImageCollection(lst.map(dateDiffProp))

        fineDist_dateProp = fineDist_dateProp.filter(ee.Filter.neq('DateDiff', 1))

        # Create collection of unpaired modis images between the fine/coarse pairs
        modisUnpaired = invertedJoin.apply(modis, fineDist_dateProp, dayfilter)
        
    sortedImages = [landsatPaired, modisPaired, modisUnpaired]

    return sortedImages




# CREATE SUBCOLLECTIONS FOR EACH SET OF LANDSAT/MODIS PAIRS



def getDates(image, empty_list):
    """
    Get date from image and append to list.

    Parameters
    ----------
    image : image.Image
        Any earth engine image.
    empty_list : ee_list.List
        Earth engine list object to append date to.

    Returns
    -------
    updatelist : ee_list.List
        List with date appended to the end.

    """
    # get date and update format
    date = ee.Image(image).date().format('yyyy-MM-dd')

    # add date to 'empty list'
    updatelist = ee.List(empty_list).add(date)

    return updatelist

# def subsFilter(sub):
    
#     modUnpaired = ee.List(sub).get(2)
    
#     subUpdate = ee.Algorithms.If(ee.String(ee.Image(ee.List(modUnpaired).get(0)).get('DOY')).equals(ee.String('366')),\
#                                        ee.List(sub).remove((ee.List(modUnpaired))), ee.List(sub))
    
#     return subUpdate

def makeSubcollections(sortedImages):
    """
    Reorganize the list of collections into a list of lists of lists. Each\
    list within the list will contain 3 lists. The first of these three will\
    have the earliest and latest Landsat images. The second list will have the\
    earliest and latest MODIS images. The third list will have all the MODIS\
    images between the earliest and latest pairs.\
    (e.g. L8 on 05/22/2017 & 06/23/2017, MOD 05/23/2017 & 06/23/2017,\
     MOD 05/23/2017 through 06/22/2017).

    Parameters
    ----------
    paired : python List
        List of image collections. 1. Landsat pairs, 2. MODIS pairs, and\
        3. MODIS between each of the pairs.

    Returns
    -------
    ee_list.List
        List of lists of lists.

    """
    

    
    def getSub(ind):
        """
        Local function to create individual subcollection.

        Parameters
        ----------
        ind : int
            Element of the list to grab.

        Returns
        -------
        ee_list.List
            List of pairs lists for prediction 2 pairs and images between.

        """
        # get modis images between these two dates
        mod_p = sortedImages[2] \
            .filterDate(ee.List(dateList).get(ind),
                        ee.Date(ee.List(dateList).get(ee.Number(ind).add(1)))\
                            .advance(1, 'day'))
        
        mod_test =ee.Algorithms.If(mod_p.size().neq(0), mod_p.toList(10000), ee.List(['null']))

        #mod_p = mod_p.toList(mod_p.size())

        # get landsat images
        lan_01 = sortedImages[0] \
            .filterDate(ee.List(dateList).get(ind),
                        ee.Date(ee.List(dateList).get(ee.Number(ind).add(1)))\
                            .advance(1, 'day')) \
            .toList(2)
        # get modis paired images
        mod_01 = sortedImages[1] \
            .filterDate(ee.List(dateList).get(ind),
                        ee.Date(ee.List(dateList).get(ee.Number(ind).add(1)))\
                            .advance(1, 'day')) \
            .toList(2)

        # combine collections to one object
        subcollection = ee.List([lan_01, mod_01, mod_test]) #mod_p

        

        return subcollection

    # empty list to store dates
    empty_list = ee.List([])

    # fill empty list with dates
    dateList = sortedImages[0].iterate(getDates, empty_list)
    dateList1 = sortedImages[1].iterate(getDates, empty_list)
    dateList2 = sortedImages[2].iterate(getDates, empty_list)
    
#     print('fine', ee.List(dateList).getInfo())
#     print('coarse', ee.List(dateList1).getInfo())
#     print('coarseUnpaired', ee.List(dateList2).getInfo())
    

    # filter out sub collections from paired and unpaired collections
    subcols = ee.List.sequence(0, ee.List(dateList).length().subtract(2))\
        .map(getSub)#.map(subsFilter)
    
    #print(subcols.getInfo())
    return subcols.distinct()

#Remove certain properties in featur collections
def removeProperty(feat, prop):
    properties = ee.Feature(feat).propertyNames()
    selectProperties = properties.filter(ee.Filter.neq('item', prop))
    return feat.select(selectProperties)




#DSWE Post Processing
### These functions all go into calculating the USGS Dynamic water m

def AddFmask(image):
    ndvi = ee.Image(image).normalizedDifference(['nir', 'red'])
    nir = ee.Image(image).select(['nir']).multiply(0.0001);  # change DF
    fwater = ndvi.lt(0.01).And(nir.lt(0.11)).Or(ndvi.lt(0.1).And(nir.lt(0.05)))
    fmask = fwater.rename(['fmask'])
    ##mask the fmask so that it has the same footprint as the quality (BQA) band

    return(ee.Image(image).addBands(fmask))

def Mndwi(image):
    return image.normalizedDifference(['green', 'swir1']).rename('mndwi')

def Mbsrv(image):
    return image.select(['green']).add(image.select(['red'])).rename('mbsrv')

def Mbsrn(image):
    return image.select(['nir']).add(image.select(['swir1'])).rename('mbsrn')

def Ndvi(image):
    return image.normalizedDifference(['nir', 'red']).rename('ndvi')

def Awesh(image):
    return (image.addBands(Mbsrn(image)) \
    .expression('blue + 2.5 * green + (-1.5) * mbsrn + (-0.25) * swir2', {
      'blue': image.select(['blue']),
      'green': image.select(['green']),
      'mbsrn': Mbsrn(image).select(['mbsrn']),
      'swir2': image.select(['swir2'])
      }))

def Dswe(i):
    mndwi = Mndwi(i)
    mbsrv = Mbsrv(i)
    mbsrn = Mbsrn(i)
    awesh = Awesh(i)
    swir1 = i.select(['swir1'])
    nir = i.select(['nir'])
    ndvi = Ndvi(i)
    blue = i.select(['blue'])
    swir2 = i.select(['swir2'])

    t1 = mndwi.gt(0.124)
    t2 = mbsrv.gt(mbsrn)
    t3 = awesh.gt(0)
    t4 = mndwi.gt(-0.44) \
    .And(swir1.lt(900)) \
    .And(nir.lt(1500)) \
    .And(ndvi.lt(0.7))
    t5 = mndwi.gt(-0.5) \
    .And(blue.lt(1000)) \
    .And(swir1.lt(3000)) \
    .And(swir2.lt(1000)) \
    .And(nir.lt(2500))

    t = t1.add(t2.multiply(10)).add(t3.multiply(100)).add(t4.multiply(1000)).add(t5.multiply(10000))

    noWater = t.eq(0) \
    .Or(t.eq(1)) \
    .Or(t.eq(10)) \
    .Or(t.eq(100)) \
    .Or(t.eq(1000))
    hWater = t.eq(1111) \
    .Or(t.eq(10111)) \
    .Or(t.eq(11011)) \
    .Or(t.eq(11101)) \
    .Or(t.eq(11110)) \
    .Or(t.eq(11111))
    mWater = t.eq(111) \
    .Or(t.eq(1011)) \
    .Or(t.eq(1101)) \
    .Or(t.eq(1110)) \
    .Or(t.eq(10011)) \
    .Or(t.eq(10101)) \
    .Or(t.eq(10110)) \
    .Or(t.eq(11001)) \
    .Or(t.eq(11010)) \
    .Or(t.eq(11100))
    pWetland = t.eq(11000)
    lWater = t.eq(11) \
    .Or(t.eq(101)) \
    .Or(t.eq(110)) \
    .Or(t.eq(1001)) \
    .Or(t.eq(1010)) \
    .Or(t.eq(1100)) \
    .Or(t.eq(10000)) \
    .Or(t.eq(10001)) \
    .Or(t.eq(10010)) \
    .Or(t.eq(10100))

    iDswe = noWater.multiply(0) \
    .add(hWater.multiply(1)) \
    .add(mWater.multiply(2)) \
    .add(pWetland.multiply(3)) \
    .add(lWater.multiply(4))

    return(iDswe.rename('dswe'))



##Function for limiting the max number of tasks sent to
#earth engine at one time to avoid time out errors
import time
def maximum_no_of_tasks(MaxNActive, waitingPeriod):
    ##maintain a maximum number of active tasks
    time.sleep(10)
    ## initialize submitting jobs
    ts = list(ee.batch.Task.list())

    NActive = 0
    for task in ts:
        if ('RUNNING' in str(task) or 'READY' in str(task)):
            NActive += 1
    ## wait if the number of current active tasks reach the maximum number
    ## defined in MaxNActive
    while (NActive >= MaxNActive):
        time.sleep(waitingPeriod) # if reach or over maximum no. of active tasks, wait for 2min and check again
        ts = list(ee.batch.Task.list())
        NActive = 0
        for task in ts:
            if ('RUNNING' in str(task) or 'READY' in str(task)):
                NActive += 1
    return()
def removeGeometry(feat):
    #remove geometry of a feature
    
    return ee.Feature(feat).setGeometry(None)