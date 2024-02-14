#WQP data retrieval
#Updated to match dataretrieval package update
#Updated with TryCatch() to automatically skip empty basins without breaking fxn
#For query args use this website: https://www.waterqualitydata.us/portal_userguide/

##########
#Get sites
##########
#IF YOU DO NOT HAVE SITES, CAN START HERE querying from more general parameters
#I find that going by site gives prevents missing data as this specific function will break at the first NULL encounter
#Could also do by bounding box/state but not shown here

library(iterators)
library(lubridate)
library(ellipsis)
library(desc)
library(withr)
library(ps)
library(backports)
library(rstudioapi)
library(usethis)
library(curl,lib.loc)
library(devtools)
library(dataRetrieval)  #package for water quality portal
library(foreach)
library(doParallel)
library(readr)
library(dplyr)
library(tidyr)

#Set wd for saving
setwd('/yourPath')

#CONUS HUCs
HUCs01 <- c('0101*','0102*','0103*','0104*','0105*','0106*','0107*','0108*','0109*','0110*','0111*')
HUCs02 <- c('0201*','0202*','0203*','0204*','0205*','0206*','0207*') 
HUCs03 <- c('0301*','0302*','0303*','0304*','0305*','0306*','0307*','0308*','0309*','0310*','0311*','0312*','0313*','0314*','0315*','0316*','0317*','0318*')
HUCs04 <- c('0401*','0402*','0403*','0404*','0405*','0406*','0407*','0408*','0409*','0410*','0411*','0412*','0413*','0414*','0415*')
HUCs09 <- c('0901*','0902*','0903*')
HUCs12 <- c('1201*','1202*','1203*','1204*','1205*','1206*','1207*','1208*','1209*','1210*','1211*') 
HUCs13 <- c('1301*','1302*','1303*','1304*','1305*','1306*','1307*','1308*')
HUCs14 <- c('1401*','1402*','1403*','1404*','1405*','1406*','1407*','1408*')
HUCs15 <- c('1501*','1502*','1503*','1504*','1505*','1506*','1507*','1508*')
HUCs16 <- c('1601*','1602*','1603*','1604*','1605*','1606*')
HUCs17 <- c('1701*','1702*','1703*','1704*','1705*','1706*','1707*','1708*','1709*','1710*','1711*','1712*')
HUCs18 <- c('1801*','1802*','1803*','1804*','1805*','1806*','1807*','1808*','1809*','1810*')
HUCs19 <- c('1901*','1902*','1903*','1904*','1905*','1906*')
HUCs20 <- c('2001*','2002*','2003*','2004*','2005*','2006*','2007*','2008*','2009*')
HUCs21 <- c('2101*', '2102*','2103*') 

#MS Basin
HUCs05 <- c('0501*','0502*','0503*','0504*','0505*','0506*','0507*','0508*','0509*','0510*','0511*','0512*','0513*','0514*')
HUCs06 <- c('0601*','0602*','0603*','0604*')
HUCs07 <- c('0701*','0702*','0703*','0704*','0705*','0706*','0707*')
HUCs08 <- c('0801*','0802*','0803*','0804*','0805*','0806*','0807*','0808*','0809*')
HUCs10 <- c('1002*','1003*','1004*','1005*','1006*','1007*','1008*','1009*','1010*','1011*','1012*','1013*','1014*','1015*','1016*','1017*','1018*','1019*','1020*','1021*','1022*','1023*','1024*','1025*','1026*','1027*','1028*','1029*','1030*')
HUCs11 <- c('1101*','1102*','1103*','1104*','1105*','1106*','1107*','1108*','1109*','1110*','1111*','1112*','1113*','1114*')

#Combine
HUCs <- list(HUCs04, HUCs05,HUCs06,HUCs07, HUCs08, HUCs09, HUCs10, HUCs11, HUCs12, HUCs13, 
             HUCs14, HUCs15, HUCs16, HUCs17, HUCs18, HUCs19, HUCs20, HUCs21)

## Add to a list (this funciton is built for large queries), here is a short example
HUCs <- list(HUCs09)


for (i in HUCs) {
  siteListStr = paste0('HUC',substr(as.character(i)[1], 1,2))
  print(siteListStr)
  
  #Define characteristicNames of interest
  S_Forms <- c("Sediment", "Turbidity", "Total suspended solids", "Suspended sediment concentration (SSC)", "Suspended Sediment Concentration (SSC)")
  siteType <- 'Stream'
  startDate = '2021-01-01'
  endDate = '2023-03-01'
  
  
  #Define the site function, asks the WQP for site and site metadata only
  getSites = function(HUC) {
    data_resultFiltered <- whatWQPdata(huc = HUC,
                                       siteType = siteType,
                                       characteristicName = S_Forms,
                                       startDate=startDate,
                                       endDate=endDate,
                                       convertType = TRUE) %>%
      dplyr::select(MonitoringLocationIdentifier, resultCount, siteUrl, StateName, CountyName)
    
    #Filter data by number of results per site if needed
    data_filtered <- data_resultFiltered %>% 
      dplyr::filter(resultCount >= 1)
    
    return (data_filtered) 
  }
  
  
  #Create a dataframe and weed out null locations or sub-basins
  WQPsite_df <- as.data.frame(do.call(rbind, lapply(i, getSites))) %>%
    dplyr::filter(!is.na(MonitoringLocationIdentifier))
  
  #Extract sites of interest
  siteList <- unlist(list(WQPsite_df$MonitoringLocationIdentifier))
  print(length(siteList)) #[1:5])
  
  ###############
  #Master Function
  ################
  
  #Separate into chunks of 1000
  chunk_size <- 1000
  num_chunks <- ceiling(length(siteList) / chunk_size)
  chunked_lists <- lapply(1:num_chunks, function(i) {
    start_idx <- (i - 1) * chunk_size + 1
    end_idx <- min(i * chunk_size, length(siteList))
    return(siteList[start_idx:end_idx])
  })
  
  #Define function to pull WQP data by sites (sorted by site)
  extract_WQP_data = function(site) {
    library(iterators)
    library(lubridate)
    library(ellipsis)
    library(desc)
    library(withr)
    library(ps)
    library(backports)
    library(rstudioapi)
    library(usethis)
    library(curl,lib.loc)
    library(devtools)
    library(dataRetrieval)  #package for water quality portal
    library(foreach)
    library(doParallel)
    library(readr)
    library(dplyr)
    library(tidyr)
    
    #Define fxn in tryCatch  
    out <- tryCatch({
      #Define constituent parameters (CharacteristicName) needed to grab from multiple databases
      S_Forms <- c("Sediment", "Turbidity", "Total suspended solids", "Suspended sediment concentration (SSC)", "Suspended Sediment Concentration (SSC)")

      #Extract site data by HUC with selected columns
      data_resultFiltered <- whatWQPdata(siteid = site,
                                         siteType = siteType,
                                         characteristicName = S_Forms,
                                         startDate=startDate,
                                         endDate=endDate,
                                         convertType = TRUE) %>%
        dplyr::select(MonitoringLocationIdentifier, resultCount, siteUrl, StateName, CountyName)
      
      #Filter data by number of results per site if needed
      data_filtered <- data_resultFiltered %>% 
        dplyr::filter(resultCount >= 1)
      #Name filtered data to a new variable to call result data
      sites <- data_filtered$MonitoringLocationIdentifier
      
      data_sites <- whatWQPsites(siteNumbers = sites,
                                 siteType = siteType,
                                 characteristicName = S_Forms,
                                 startDate=startDate,
                                 endDate=endDate,
                                 convertType = TRUE) %>%
      dplyr::select(MonitoringLocationIdentifier, HUCEightDigitCode, DrainageAreaMeasure.MeasureValue, DrainageAreaMeasure.MeasureUnitCode, ContributingDrainageAreaMeasure.MeasureValue, ContributingDrainageAreaMeasure.MeasureUnitCode, VerticalMeasure.MeasureValue, VerticalMeasure.MeasureUnitCode, VerticalAccuracyMeasure.MeasureValue, VerticalAccuracyMeasure.MeasureUnitCode, VerticalCollectionMethodName, LatitudeMeasure, LongitudeMeasure, HorizontalCoordinateReferenceSystemDatumName)
      
      #Call actual data by filtered sites
      data_retrieval <- readWQPdata(siteNumbers = sites,
                                    siteType = siteType,
                                    characteristicName = S_Forms,
                                    startDate=startDate, 
                                    endDate=endDate,
                                    tz = 'UTC',
                                    convertType = TRUE)%>%
        dplyr::filter(!is.na(ResultMeasureValue))%>%
        dplyr::select(MonitoringLocationIdentifier, OrganizationIdentifier, OrganizationFormalName, ActivityDepthHeightMeasure.MeasureValue, ActivityDepthHeightMeasure.MeasureUnitCode, ActivityIdentifier, ActivityMediaName, ActivityStartDate, ActivityStartDateTime, ActivityStartTime.Time, ActivityStartTime.TimeZoneCode, MonitoringLocationIdentifier, HydrologicCondition, HydrologicEvent, CharacteristicName, ResultAnalyticalMethod.MethodName, ResultParticleSizeBasisText, ResultSampleFractionText, ResultMeasureValue, ResultMeasure.MeasureUnitCode, ResultParticleSizeBasisText, ResultSampleFractionText, ResultStatusIdentifier, SampleCollectionMethod.MethodName, USGSPCode)

      #Merge location data and results
      All_WQP_Data <- data_retrieval %>%
        left_join(data_filtered, by ='MonitoringLocationIdentifier') %>%
        left_join(data_sites, by='MonitoringLocationIdentifier')
      
    },
    
    #Handle an error
    error = function(e) {
      message(paste('Reading a site caused an error:', site))
      message('Here is the original error message:')
      message(e)
      return(NA)
    },
    
    #Handle a warning
    warning = function(cond) {
      message(paste('Reading a site caused a warning:', site))
      message('Here is the original warning message:')
      message(cond)
      return(NA)
    },
    
    #Define what should happen after
    finally = {
      message(paste('Processed:', site))
    }
    
    
    
    )
    return(out) 
  }
  
  
  #Create a dataframe and weed out null locations or sub-basins
  WQPdata_frame <- as.data.frame(do.call(rbind, lapply(chunked_lists, extract_WQP_data))) #%>%
    #dplyr::filter(!is.na(MonitoringLocationIdentifier))
  print(head(WQPdata_frame))

  #Save as RDS
  WQPdata <- saveRDS(WQPdata_frame, file = paste0("WQPdatarawTEST_", siteListStr,".rds"))
  
}
