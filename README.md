# FusionCONUS_TSS
Use imagery information from Landsat and Sentinel-2 to 'teach' MODIS 30 m resolution, using machine learning to model river total total suspended sediment (TSS) concentration from fused and original image reflectance across the Continental United States.

# **OVERVIEW**

This manuscript broadly can be divided into three phases: data collection and cleaning from the Water Quality Portal (WQP), processing Landsat-5,7,8,9 and Sentinel-2 and Fusion imagery, and lastly model matchup prep, training, and validation. All resulting reflectance/TSS datasets are available on Zenodo. All scripts are executed using python except for the WQP phase, which uses R due to the capabilities of the USGS dataRetrieval package.

## **FILE DESCRIPTIONS**

### 0_WQP

**wqp_dataRetrival_SiteFxn.R** - Script to pull WQP data at the CONUS scale for river TSS by HUC 2-digit code.

**wqp_cleaningFxns.R** - Script to harmonize and clean all data based on Ross et al. (2019)

### 1_GEE_Fusion

**LS2_Pull.ipynb** : Code to calculate and download median reflectance and other image information from Landsat and Sentinel-2 with credit to Dr. Dongmei Feng.

**siteAttributes_GEE.ipynb** - Jupyter notebook for attaching basin characteristics and width to sites from HydroATLAS level 06 basins (Linke et al. 2019) and GRWL (Allen & Pavelsky, 2018).

GEE_ImageFusion_Fxns - Module containing three scripts (Nietupski et al. 2021). Place in same directory or into the anaconda site-packages directory within the working environent.

1.  **get_paired_collections.py** - Functions to retrieve, filter, mask, sort, and organize the Landsat and MODIS data.

2.  **prep_functions.py** - Functions to preprocess the Landsat and MODIS imagery including perform a co-registration step, determine and mask similar pixels, and convert images to 'neighborhood' images with bands that are sorted in the necessary order for the core functions.

3.  **core_functions.py** - Main functions needed to perform image fusion. If all images have been preprocessed and formatted correctly then these functions can be run to predict new images at times when only MODIS is available. Functions include spectral distance to similar pixels, spatial distance, weight calculation, conversion coefficient calculation, and prediction.

**fusionDemo.py -** Script with highly customizable scaling functions and an example of the workflow needed to use all the GEE_ImageFusion_Fxns module.


### 2_model

**matchupsCONUS.ipynb -** A jupyter notebook that collates and matches +/- 1 day WQP measurements to an LS2 or Fusion image, cleans the data, and prepares the matchups for model training.

**0_RFmodelFunctions.py -** All functions needed to run the RF model with reflectance-TSS matchup data including preprocessing, formatting, model training, skill score export, predictions, plots, and evaluation across 100 iterations. Formatted to input all model iterations as separate files.

**1_RFmodels.py -** Requires matchup, unmatched LS2, and unmatched Fusion dataframes.

**2_RFmodels_evaluate.py -** This script takes the outputs from 1_RFmodels.py and outputs figures to evaluate model skill across all experiments. Outputs figures in jpg files.

**RFmodels_combine.py** - Combine all 100 iterations to single output with all data sources (matchup, wqp, predictions in one netCDF) for further analysis.

