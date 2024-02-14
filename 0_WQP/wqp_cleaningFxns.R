#This code roughly cleans and then examines the temporal data distribution by basin

library(tzdb)
library(withr)
library(readr)
library(dplyr)
library(ggplot2)
library(purrr)
library(labeling)
library(farver)
library(vroom)
library(lubridate)
library(sf)
library(mapview)



setwd('~/Documents/SNiP/WQP_data')




##Call WQP Data
##MS Basin
df <- readRDS("path_to_file.rds")
fileString = 'test'

#Quick map if lat and long columns are specified
mapview(df, xcol = "long", ycol = "lat", crs = 4269, grid = FALSE)


##########
#Functions
##########
#Built for TSS

#Select and ROUGHLY adapt to same Characteristic Names and parameters as Aquasat
#Went ahead and removed na values that Ross et al 2019 has sprinkled throughout his code.
wqp.tss.renamer <- function(df){
  TSS_forms <- c("Total suspended solids", "Suspended Sediment Concentration (SSC)", "Suspended sediment Concentration (SSC)", "Fixed Suspended solids", "Sediment")
  TSS_fractionsNOT <- c('Bedload', 'Bed Sediment', 'Dissolved')
  simple.names <- df %>%
    dplyr::select(date=ActivityStartDateTime,
                  parameter=CharacteristicName,
                  units=ResultMeasure.MeasureUnitCode,
                  SiteID=MonitoringLocationIdentifier,
                  huc=HUCEightDigitCode,
                  org=OrganizationFormalName,
                  org_id=OrganizationIdentifier,
                  time=(ActivityStartDateTime),
                  value=ResultMeasureValue,
                  sample_method=SampleCollectionMethod.MethodName,
                  analytical_method=ResultAnalyticalMethod.MethodName,
                  particle_size=ResultParticleSizeBasisText,
                  date_time=(ActivityStartDateTime),
                  media=ActivityMediaName,
                  sample_depth=ActivityDepthHeightMeasure.MeasureValue,
                  sample_depth_unit=ActivityDepthHeightMeasure.MeasureUnitCode,
                  fraction=ResultSampleFractionText,
                  status=ResultStatusIdentifier,
                  hydroCondition = HydrologicCondition,
                  hydroEvent = HydrologicEvent,
                  lat=LatitudeMeasure,
                  long=LongitudeMeasure,
                  datum=HorizontalCoordinateReferenceSystemDatumName) %>%
    #Remove trailing white space in labels
    mutate(units = trimws(units)) %>%
    mutate(time = as.character(format(as_datetime(date_time), '%H:%M:%S'))) %>%
    mutate(date = as.character(format(as_datetime(date_time), '%Y-%m-%d'))) %>%
    filter(parameter %in% TSS_forms) %>%
    filter(!fraction %in% TSS_fractionsNOT) %>%
    filter(media=='Water') %>% #Must be in a river or stream
    filter(value > 0.01) %>% #Must be somewhat detectable
    filter(value < 100000) %>% #Must be reasonable
    filter(is.na(sample_depth) | sample_depth <= 3) %>% #can't throw away all values/surfacae water samples
    filter(!is.na(value)) %>% #Must exist
    filter(!is.na(date)) %>%
    filter(!is.na(lat)) %>% #Exclude sites with missing geospatial data
    filter(!is.na(long)) %>%
    distinct(SiteID,value, date, .keep_all = TRUE) #Remove duplicates - rough - did not average between multiples at same site/time
  dropped = nrow(df)-nrow(simple.names)
  print(paste('we dropped',dropped,'samples'))
  print(paste('nrows',nrow(simple.names)))
  return(simple.names)
}

#Function for nonsensical methods for TSS
nonsense.methods.fxn <- function(df){
  #There are a lot of parameter codes so we are just going to use a grepl command with key words that definitely disqualify the sample
  non.sensical.tss.methods <- df %>%
    filter(grepl("Oxygen|Nitrogen|Ammonia|Metals|E. coli|Carbon|Anion|Cation|Phosphorus|Silica|PH|HARDNESS|Nutrient|Turbidity|Temperature|Nitrate|Conductance|Alkalinity|Chlorophyll",analytical_method,ignore.case=T))
  
  
  tss.filtered.method <- df %>%
    filter(!analytical_method %in% non.sensical.tss.methods$analytical_method)
  
  print(paste('We dropped', round(nrow(non.sensical.tss.methods)/nrow(df)*100,2),'% of samples, because the method used did not make sense. These methods are:'))
  print(unique(non.sensical.tss.methods$analytical_method),wrap='',sep=' - ') #Nice function for printing long vectors horizontally separated by dash
  return(tss.filtered.method)
}

#Function to filter and synchronize units for TSS

unit.harmony.fxn <- function(df){
  #Select only units for % and filter to keep only the sand fraction data ('< 0.0625 mm','sands')
  tss.p <- df %>%
    filter(units =='%' & (particle_size %in%  c('< 0.0625 mm','sands'))) %>%
    mutate(conversion=NA,
           parameter_name='p.sand',
           harmonized_value=value,
           harmonized_unit='%')
  
  
  #Make a tss lookup table
  tss.lookup <- tibble(units=c('mg/l','g/l','ug/l','ppm'),
                       conversion = c(1,1000,1/1000,1))
  
  #Join to the lookup table and harmonize units
  
  tss.harmonized <- df %>%
    inner_join(tss.lookup,by='units') %>%
    mutate(parameter_name = 'tss',
           harmonized_value=as.numeric(value) * as.numeric(conversion),
           harmonized_unit='mg/l') 
  
  #Combine p.sand and tss dataframes
  #remove conversion/original columns and rename harmonized columns
  tss.harmonized.p <- rbind(tss.p, tss.harmonized) %>%
    select(-value, -units, -conversion,
           value = harmonized_value,
           units = harmonized_unit)
  dropped = nrow(df)-nrow(tss.harmonized.p)
  print(paste('we dropped',dropped,'samples with unknown/unusable units'))
  print(nrow(tss.harmonized.p))
  return(tss.harmonized.p)
}

#Fxn to harmonize depth units to m, filters for surface samples
depth.harmony.fxn <- function(df) {
  #Define a depth lookup table to convert all depth data to meters. 
  depth.lookup <- tibble(sample_depth_unit=c('cm','feet','ft','in','m','meters','None'),
                         depth_conversion=c(1/100,.3048,.3048,0.0254,1,1,NA)) 
  
  #Join depth lookup table to tss data
  tss.depth <- df %>%
    dplyr::mutate(sample_depth = as.numeric(sample_depth)) %>%
    left_join(depth.lookup,by='sample_depth_unit') %>%
    #Some depth measurements have negative values (assume that is just preference)
    #I also added .01 meters because many samples have depth of zero assuming they were
    # taken directly at the surface
    #remove original/conversion columns and rename harmonized columns
    mutate(harmonized_depth=abs(sample_depth*depth_conversion)+.01) %>%
    select(-depth_conversion, -sample_depth, -sample_depth_unit,
           sample_depth = harmonized_depth) %>%
    mutate(sample_depth_unit = 'm') %>%
    filter(is.na(sample_depth) | sample_depth <= 1)
  
  # We lose lots of data by keeping only data with depth measurements
  print(paste('If we only kept samples that had depth information we would lose',round((nrow(df)-nrow(tss.depth))/nrow(df)*100,1),'% of samples'))
  dropped_depth = nrow(df)-nrow(tss.depth)
  print(paste('we dropped an additional',dropped_depth,'samples with converted depths > 1 m'))
  
  return(tss.depth)
}

#Function for date and time 
#Need to add date and time splitter later
date.format.fxn <- function(df){
  date.format <- df %>% 
    dplyr::mutate(time = as.character(format(as_datetime(date_time), '%H:%M:%S'))) %>%
    dplyr::mutate(date = as.character(format(as_datetime(date_time), '%Y-%m-%d'))) %>%
    #dplyr::mutate(date_only= as.character(ifelse(is.na(as_datetime(date_time)) | time == '00:00:00',T,F))) %>%
    #dplyr::mutate(date_unity = as.character(ymd_hms(ifelse(as_datetime(date_only) == T,
    #                                                       paste(date,'00:00:00'),
    #                                                       date),
    #                                                tz='UTC'))) %>%
    #remove any time stamps that are NA
    dplyr::filter(!is.na(date))
  return(date.format)
}


#Function for datum
#From Aquasat:
#Setup a datum table to transform different datums into WGS84 for GEE and to 
#match LAGOS. Note that we make the somewhat reisky decision that others and unknowns
# are mostly NAD83 (since > 60% of the data is originally in that projection)
datum.format.fxn <- function(df){
  print(table(df$datum)) #Alter function if needed to account for misc. datum
  datum_epsg <- tibble(datum=c('NAD27','NAD83','OTHER','UNKWN','WGS84'),
                       epsg=c(4267,4269,4269,4269,4326))
  # Get distinct lat longs and sites
  inv_uniques <- df %>%
    distinct(SiteID,lat,long,datum)
  
  ## project inventory using the above lookup table
  ## Have to loop over each projection
  projected <- list() 
  
  for(i in 1:nrow(datum_epsg)){
    #Select each datum iteratively
    d1 <- datum_epsg[i,]
    # Join inventory to that datum only and then feed it that CRS and then 
    # transform to wgs84
    inv_reproj <- inv_uniques %>%
      filter(datum == d1$datum) %>%
      inner_join(d1, by='datum') %>%
      st_as_sf(.,coords=c('long','lat'),crs=d1$epsg) %>%
      st_transform(4326)
    
    projected[[i]] <- inv_reproj
  }
  
  #print(projected)
  # Note that we lose some sites in GUAM which is intentional
  # Add back in lat longs which are all WGS84 now
  inv_wgs84 <- do.call('rbind', projected) %>%
    mutate(lat = st_coordinates(.)[,2],
           long=st_coordinates(.)[,1]) %>%
    as.data.frame(.) %>%
    select(-geometry,-datum) 
  print(head(inv_wgs84))
  #print(head(df))
  
  #Put back together
  df <- inner_join(df, inv_wgs84, by = 'SiteID')
  
  return(df)
  print('done')
}



###########
#Call Fxns
###########

#Create a cleaned dataset
df_S <- df %>% 
  wqp.tss.renamer() %>%
  nonsense.methods.fxn() %>%
  unit.harmony.fxn() %>%
  depth.harmony.fxn() %>%
  date.format.fxn() %>%
  datum.format.fxn() %>%
  dplyr::select(-lat.x, -long.x) %>%
  dplyr::rename(lat = lat.y, long=long.y)
#rm(tss,tss.depth,tss.filtered,tss.lookup,tss.p,non.sensical.tss.methods,depth.lookup)

names(df_S)

length(unique(df_S$SiteID))
nrow(df_S)

#Quick vis
map_df_S <- df_S %>%
  distinct(SiteID, lat, long)

mapview(map_df_S, xcol = "long", ycol = "lat", crs = 4269, grid = FALSE)


#write out any needed files
df_S <- write.csv(df_S_aquaSNiP, file = paste0('~/your_wd/wqpClean_',fileString,'_tss.csv'), row.names = FALSE)






