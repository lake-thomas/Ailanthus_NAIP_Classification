# This script filters and deduplicates species occurrence data from iNaturalist, GBIF, and state/federal agencies for Ailanthus.
# It combines the datasets and removes duplicates based on the resolution of Worldclim climate data, as is typical for species distribution modeling.
# After deduplication and filtering, we sample background points throughout the US for modeling.
# Then saves the cleaned data with occurrences and background points to a .csv file for use in species distribution modeling.
# With those cleaned data, we lastly sample NAIP Imagery from presences and background records for model training.

# Author: Thomas Lake, November 2025

# Imports
import os

# Set up GDAL and PROJ environment variables for geopandas compatibility
conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

# Imports after setting environment variables
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point

####################################
#        Collate Occurrences       # 
####################################

# Load .csv datasets from iNaturalist and GBIF
# Accessed Nov 4, 2025
inat_df = pd.read_csv(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\GBIF_iNat_Ailanthus\iNat_Ailanthus_Raw_Nov4_2025\observations-638670.csv') # iNaturalist data
gbif_df = pd.read_csv(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\GBIF_iNat_Ailanthus\GBIF_Ailanthus_Raw_Nov4_2025\GBIF_Ailanthus_Raw_Nov4_2025.csv') # GBIF data
print(inat_df.shape, gbif_df.shape)
print("iNat and GBIF datasets loaded.")

# Load shapefile datasets from State, Federal Spotted Lanternfly/ Ailanthus Surveys
# SLF Surveys: 2023 Cumulative ('Host' = "Ailanthus altissima"). Keep column 'State' to track source agency user.
# Contains approx 600,000 surveys largely in PA, NJ, DE, MD, VA, WV, and some surrounding states.
slf_2023_cumulative = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\Pest_Pathogen Occurrence\Spotted_Lanternfly\SLF_Update_2025\slf_2023_cumulative.shp', encoding='latin1')
print("SLF 2023 Cumulative dataset loaded.")
print(slf_2023_cumulative.head())
print(slf_2023_cumulative.shape)

# Visual Surveys for SLF and Ailanthus from INDNR, NCDA, NYSAGM, OHDA, PPQ KY, and PPQ TN. Each layer has a different query to filter for Ailanthus hosts.
# PPQ_SLF_Missing_Visual_Surveys_2023 — INDNR_Visual_Surveys_2023 (where host_tree == "Ailanthus"). Add column 'agency' = 'INDNR' to track source agency user.
ailanthus_visual_survey_INDNR_2023 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\INDNR_PPQ_SLF_Visual_Survey_2023.shp')
print("Ailanthus Visual Survey INDNR 2023 dataset loaded.")
print(ailanthus_visual_survey_INDNR_2023.head())
print(ailanthus_visual_survey_INDNR_2023.shape)

# PPQ_SLF_Missing_Visual_Surveys_2023 — NCDA_Visual_Surveys_2023 (where host == "Ailanthus altissima"). Add column 'agency' = 'NCDA' to track source agency user.
ailanthus_visual_survey_NCDA_2023 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\NCDA_PPQ_SLF_Visual_Survey_2023.shp')
print("Ailanthus Visual Survey NCDA 2023 dataset loaded.")
print(ailanthus_visual_survey_NCDA_2023.head())
print(ailanthus_visual_survey_NCDA_2023.shape)  

# PPQ_SLF_Missing_Visual_Surveys_2023 — NYSAGM_Visual_Surveys_2023 (where 'ALI_DBH_GT' > 0 OR 'ALI_DBH_LT' > 0 OR 'ALI_WHIP_D' > 0). Add column 'agency' = 'NYSAGM' to track source agency user.
ailanthus_visual_survey_NYSAGM_2023 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\NYSAGM_PPQ_SLF_Visual_Survey_2023.shp')
print("Ailanthus Visual Survey NYSAGM 2023 dataset loaded.")
print(ailanthus_visual_survey_NYSAGM_2023.head())
print(ailanthus_visual_survey_NYSAGM_2023.shape)

# PPQ_SLF_Missing_Visual_Surveys_2023 — OHDA_Visual_Surveys_2023 (where host == "Ailanthus altissima"). Add column 'agency' = 'OHDA' to track source agency user.
ailanthus_visual_survey_2023_OHDA = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\OHDA_PPQ_SLF_Visual_Survey_2023.shp')
print("Ailanthus Visual Survey OHDA 2023 dataset loaded.")
print(ailanthus_visual_survey_2023_OHDA.head())

# PPQ_SLF_Missing_Visual_Surveys_2023 — PPQ_KY_Visual_Surveys_2023 (where host == "Ailanthus altissima"). Add column 'agency' = 'PPQ_KY' to track source agency user.
ailanthus_visual_survey_2023_KY = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\KY_PPQ_SLF_Visual_Survey_2023.shp')
print("Ailanthus Visual Survey KY 2023 dataset loaded.")
print(ailanthus_visual_survey_2023_KY.head())
print(ailanthus_visual_survey_2023_KY.shape)

# PPQ_SLF_Missing_Visual_Surveys_2023 — PPQ_TN_Visual_Surveys_2023 (where host == "Ailanthus altissima"). Add column 'agency' = 'PPQ_TN' to track source agency user.
ailanthus_visual_survey_2023_TN = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\TN_PPQ_SLF_Visual_Survey_2023.shp')
print("Ailanthus Visual Survey TN 2023 dataset loaded.")
print(ailanthus_visual_survey_2023_TN.head())
print(ailanthus_visual_survey_2023_TN.shape)

# Treatment points for Ailanthus (each point is Ailanthus) treated with herbicide or insecticide in PA in 2019. Keep column 'created_us' to track source agency user.
ailanthus_trees_treatment_points_2019 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\2019_AilanthusTreesTreatmentsPoints.shp')
print("Ailanthus Trees Treatment Points 2019 dataset loaded.")
print(ailanthus_trees_treatment_points_2019.head())
print(ailanthus_trees_treatment_points_2019.shape)

# Surveys for Ailanthus in maryland, north carolina, west virginia, new jersey, deleware, virginia, and by USDA APHIS PPQ
# Select points where HostStatus == 'Ailanthus altissima' and keep column 'created_us' to track source agency user
allstates_visual_surveys_2019 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\2019_AllStates_Visual_SurveyDownload_11_04_2019.shp')
print("All States Visual Surveys 2019 dataset loaded.")
print(allstates_visual_surveys_2019.head())
print(allstates_visual_surveys_2019.shape)

# Surveys of Ailanthus from PDA, FIA, and LCC (various years)
# Select points where 'Species' == 'Ailanthus' and keep column 'DataSource' to track source agency.
ailanthus_combined_pda_fia_lcc = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\Ailanthus_Combined_PDA_FIA_LCC.shp')
print("Ailanthus Combined PDA, FIA, LCC dataset loaded.")
print(ailanthus_combined_pda_fia_lcc.head())
print(ailanthus_combined_pda_fia_lcc.shape)

# Individual state datasets for Ailanthus occurrences
# Connecticut 2020 survey of Ailanthus occurrences. Select where 'Host' == "Ailanthus altissima" and keep column 'Agency' to track source agency.
ailanthus_ct_2020 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\CT_2020.shp')
print("Ailanthus CT 2020 dataset loaded.")
print(ailanthus_ct_2020.head())
print(ailanthus_ct_2020.shape)

# Delaware 2020 survey of Ailanthus occurrences. Select where 'Host' == "Ailanthus altissima" and keep column 'Agency' to track source agency.
ailanthus_de_2020 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\DE_2020_1.shp')
print("Ailanthus DE 2020 dataset loaded.")
print(ailanthus_de_2020.head())
print(ailanthus_de_2020.shape)

# Maryland 2020 survey of Ailanthus occurrences. Select where 'Host' == "Ailanthus altissima" and keep column 'Agency' to track source agency.
ailanthus_md_2020 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\MD_2020_1.shp')
print("Ailanthus MD 2020 dataset loaded.")
print(ailanthus_md_2020.head())
print(ailanthus_md_2020.shape)

# Maryland 2020 survey of Ailanthus occurrences (file 2). Select where 'HostStatus' == "Ailanthus altissima" and keep column 'Agency' to track source agency.
ailanthus_md_2020_2 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\MD_2020_2.shp')
print("Ailanthus MD 2020 (file 2) dataset loaded.")
print(ailanthus_md_2020_2.head())
print(ailanthus_md_2020_2.shape)

# New Jersey 2020 survey of Ailanthus occurrences. Select where 'Host' == "Ailanthus altissima" and keep column 'Agency' to track source agency.
ailanthus_nj_2020 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\NJ_2020_1.shp')
print("Ailanthus NJ 2020 dataset loaded.")
print(ailanthus_nj_2020.head())
print(ailanthus_nj_2020.shape)

# New Jersey 2020 survey of Ailanthus occurrences (file 2). Select where 'HostStatus' == "Ailanthus altissima" and keep column 'Agency' to track source agency.
ailanthus_nj_2020_2 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\NJ_2020_2.shp')
print("Ailanthus NJ 2020 (file 2) dataset loaded.")
print(ailanthus_nj_2020_2.head())
print(ailanthus_nj_2020_2.shape)

# New York 2020 survey of Ailanthus occurrences. Select where 'Host' == "Ailanthus altissima" and keep column 'Agency' to track source agency.
ailanthus_ny_2020 = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\NY_2020.shp')
print("Ailanthus NY 2020 dataset loaded.")
print(ailanthus_ny_2020.head())
print(ailanthus_ny_2020.shape)

# NYSAGM (NY State Dept of Agriculture and Markets) survey of Ailanthus. Select where 'host' == "Ailanthus_altissima" and add column 'Agency' = 'NYSAGM' to track source agency.
ailanthus_nysagm = gpd.read_file(r'Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_Spotted_Lanternfly_Surveys\Spotted_Lanternfly\NYSAGM_2020.shp')
print("Ailanthus NYSAGM dataset loaded.")
print(ailanthus_nysagm.head())
print(ailanthus_nysagm.shape)


print("All iNat, GBIF, State, and Federal Ailanthus survey datasets loaded.")

###############################
#   Filter Occurrences        #
###############################

# iNaturalist filtering for quality
# 'observed_on' between 2015 and 2025
# positional_accuracy <= 125 meters
# geoprivacy != 'obscured'
# coordinates_obscured = False
# quality_grade = 'research'  

inat_df_filtered = inat_df[
    (pd.to_datetime(inat_df['observed_on']).dt.year >= 2015) &
    (pd.to_datetime(inat_df['observed_on']).dt.year <= 2025) &
    (inat_df['positional_accuracy'] <= 125) &
    (inat_df['geoprivacy'] != 'obscured') &
    (inat_df['coordinates_obscured'] == False) & 
    (inat_df['quality_grade'] == 'research')
].copy()

print("iNaturalist dataset filtered from ", inat_df.shape, "to", inat_df_filtered.shape, "records.")
print(inat_df_filtered.head())
print(inat_df_filtered.shape)

# GBIF filtering for quality
# year between 2015 and 2025
# coordinateUncertaintyInMeters <= 125

gbif_df_filtered = gbif_df[
    (gbif_df['year'] >= 2015) &
    (gbif_df['year'] <= 2025) &
    (gbif_df['coordinateUncertaintyInMeters'] <= 125)
].copy()

print("GBIF dataset filtered from ", gbif_df.shape, "to", gbif_df_filtered.shape, "records.")
print(gbif_df_filtered.head())
print(gbif_df_filtered.shape)

# State and Federal datasets filtering for quality and Ailanthus
# Filtering individual visual survey and treatment datasets for Ailanthus host
slf_2023_cumulative_filtered = slf_2023_cumulative[slf_2023_cumulative['Host'] == 'Ailanthus altissima'].copy()
print("SLF 2023 Cumulative dataset filtered from ", slf_2023_cumulative.shape, "to", slf_2023_cumulative_filtered.shape, "records.")
print(slf_2023_cumulative_filtered.head())
print(slf_2023_cumulative_filtered.shape)

ailanthus_visual_survey_INDNR_2023_filtered = ailanthus_visual_survey_INDNR_2023[ailanthus_visual_survey_INDNR_2023['host_tree'] == 'Ailanthus'].copy()
print("Ailanthus Visual Survey INDNR 2023 dataset filtered from ", ailanthus_visual_survey_INDNR_2023.shape, "to", ailanthus_visual_survey_INDNR_2023_filtered.shape, "records.")
print(ailanthus_visual_survey_INDNR_2023_filtered.head())
print(ailanthus_visual_survey_INDNR_2023_filtered.shape)

ailanthus_visual_survey_NCDA_2023_filtered = ailanthus_visual_survey_NCDA_2023[ailanthus_visual_survey_NCDA_2023['host'] == 'Ailanthus altissima'].copy()
print("Ailanthus Visual Survey NCDA 2023 dataset filtered from ", ailanthus_visual_survey_NCDA_2023.shape, "to", ailanthus_visual_survey_NCDA_2023_filtered.shape, "records.")
print(ailanthus_visual_survey_NCDA_2023_filtered.head())
print(ailanthus_visual_survey_NCDA_2023_filtered.shape)

ailanthus_visual_survey_NYSAGM_2023_filtered = ailanthus_visual_survey_NYSAGM_2023[
    (ailanthus_visual_survey_NYSAGM_2023['ALI_DBH_GT'] > 0) |
    (ailanthus_visual_survey_NYSAGM_2023['ALI_DBH_LT'] > 0) |
    (ailanthus_visual_survey_NYSAGM_2023['ALI_WHIP_D'] > 0)].copy()
print("Ailanthus Visual Survey NYSAGM 2023 dataset filtered from ", ailanthus_visual_survey_NYSAGM_2023.shape, "to", ailanthus_visual_survey_NYSAGM_2023_filtered.shape, "records.")
print(ailanthus_visual_survey_NYSAGM_2023_filtered.head())
print(ailanthus_visual_survey_NYSAGM_2023_filtered.shape)

ailanthus_visual_survey_2023_OHDA_filtered = ailanthus_visual_survey_2023_OHDA[ailanthus_visual_survey_2023_OHDA['host'] == 'Ailanthus altissima'].copy()
print("Ailanthus Visual Survey OHDA 2023 dataset filtered from ", ailanthus_visual_survey_2023_OHDA.shape, "to", ailanthus_visual_survey_2023_OHDA_filtered.shape, "records.")
print(ailanthus_visual_survey_2023_OHDA_filtered.head())
print(ailanthus_visual_survey_2023_OHDA_filtered.shape)

ailanthus_visual_survey_2023_KY_filtered = ailanthus_visual_survey_2023_KY[ailanthus_visual_survey_2023_KY['host'] == 'Ailanthus altissima'].copy()
print("Ailanthus Visual Survey KY 2023 dataset filtered from ", ailanthus_visual_survey_2023_KY.shape, "to", ailanthus_visual_survey_2023_KY_filtered.shape, "records.")
print(ailanthus_visual_survey_2023_KY_filtered.head())
print(ailanthus_visual_survey_2023_KY_filtered.shape)

ailanthus_visual_survey_2023_TN_filtered = ailanthus_visual_survey_2023_TN[ailanthus_visual_survey_2023_TN['host'] == 'Ailanthus altissima'].copy()
print("Ailanthus Visual Survey TN 2023 dataset filtered from ", ailanthus_visual_survey_2023_TN.shape, "to", ailanthus_visual_survey_2023_TN_filtered.shape, "records.")
print(ailanthus_visual_survey_2023_TN_filtered.head())
print(ailanthus_visual_survey_2023_TN_filtered.shape)

ailanthus_trees_treatment_points_2019_filtered = ailanthus_trees_treatment_points_2019.copy()  # All points are Ailanthus treated trees
print("Ailanthus Trees Treatment Points 2019 dataset filtered from ", ailanthus_trees_treatment_points_2019.shape, "to", ailanthus_trees_treatment_points_2019_filtered.shape, "records.")
print(ailanthus_trees_treatment_points_2019_filtered.head())
print(ailanthus_trees_treatment_points_2019_filtered.shape)

allstates_visual_surveys_2019_filtered = allstates_visual_surveys_2019[allstates_visual_surveys_2019['HostStatus'] == 'Ailanthus altissima'].copy()
print("All States Visual Surveys 2019 dataset filtered from ", allstates_visual_surveys_2019.shape, "to", allstates_visual_surveys_2019_filtered.shape, "records.")
print(allstates_visual_surveys_2019_filtered.head())
print(allstates_visual_surveys_2019_filtered.shape)

ailanthus_combined_pda_fia_lcc_filtered = ailanthus_combined_pda_fia_lcc[ailanthus_combined_pda_fia_lcc['Species'] == 'Ailanthus'].copy()
print("Ailanthus Combined PDA, FIA, LCC dataset filtered from ", ailanthus_combined_pda_fia_lcc.shape, "to", ailanthus_combined_pda_fia_lcc_filtered.shape, "records.")
print(ailanthus_combined_pda_fia_lcc_filtered.head())
print(ailanthus_combined_pda_fia_lcc_filtered.shape)

ailanthus_ct_2020_filtered = ailanthus_ct_2020[ailanthus_ct_2020['Host'] == 'Ailanthus altissima'].copy()
print("Ailanthus CT 2020 dataset filtered from ", ailanthus_ct_2020.shape, "to", ailanthus_ct_2020_filtered.shape, "records.")
print(ailanthus_ct_2020_filtered.head())
print(ailanthus_ct_2020_filtered.shape)

ailanthus_de_2020_filtered = ailanthus_de_2020[ailanthus_de_2020['Host'] == 'Ailanthus altissima'].copy()
print("Ailanthus DE 2020 dataset filtered from ", ailanthus_de_2020.shape, "to", ailanthus_de_2020_filtered.shape, "records.")
print(ailanthus_de_2020_filtered.head())
print(ailanthus_de_2020_filtered.shape)

ailanthus_md_2020_filtered = ailanthus_md_2020[ailanthus_md_2020['Host'] == 'Ailanthus altissima'].copy()
print("Ailanthus MD 2020 dataset filtered from ", ailanthus_md_2020.shape, "to", ailanthus_md_2020_filtered.shape, "records.")
print(ailanthus_md_2020_filtered.head())
print(ailanthus_md_2020_filtered.shape)

ailanthus_md_2020_2_filtered = ailanthus_md_2020_2[ailanthus_md_2020_2['HostStatus'] == 'Ailanthus altissima'].copy()
print("Ailanthus MD 2020 (file 2) dataset filtered from ", ailanthus_md_2020_2.shape, "to", ailanthus_md_2020_2_filtered.shape, "records.")
print(ailanthus_md_2020_2_filtered.head())
print(ailanthus_md_2020_2_filtered.shape)

ailanthus_nj_2020_filtered = ailanthus_nj_2020[ailanthus_nj_2020['Host'] == 'Ailanthus altissima'].copy()
print("Ailanthus NJ 2020 dataset filtered from ", ailanthus_nj_2020.shape, "to", ailanthus_nj_2020_filtered.shape, "records.")
print(ailanthus_nj_2020_filtered.head())
print(ailanthus_nj_2020_filtered.shape)

ailanthus_nj_2020_2_filtered = ailanthus_nj_2020_2[ailanthus_nj_2020_2['HostStatus'] == 'Ailanthus altissima'].copy()
print("Ailanthus NJ 2020 (file 2) dataset filtered from ", ailanthus_nj_2020_2.shape, "to", ailanthus_nj_2020_2_filtered.shape, "records.")
print(ailanthus_nj_2020_2_filtered.head())
print(ailanthus_nj_2020_2_filtered.shape)

ailanthus_ny_2020_filtered = ailanthus_ny_2020[ailanthus_ny_2020['Host'] == 'Ailanthus altissima'].copy()
print("Ailanthus NY 2020 dataset filtered from ", ailanthus_ny_2020.shape, "to", ailanthus_ny_2020_filtered.shape, "records.")
print(ailanthus_ny_2020_filtered.head())
print(ailanthus_ny_2020_filtered.shape)

ailanthus_nysagm_filtered = ailanthus_nysagm[ailanthus_nysagm['host'] == 'Ailanthus_altissima'].copy()
print("Ailanthus NYSAGM dataset filtered from ", ailanthus_nysagm.shape, "to", ailanthus_nysagm_filtered.shape, "records.")
print(ailanthus_nysagm_filtered.head())
print(ailanthus_nysagm_filtered.shape)

print("All State and Federal Ailanthus survey datasets filtered for Ailanthus host.")

#########################################
#  Standardize and Combine Occurrences  #
#########################################

print("Standardizing and combining all occurrence datasets...")

def gdf_from_coords(df, loncol, latcol, crs="EPSG:4326"): 
    # Make a GeoDataFrame from latitude/longitude columns
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[loncol], df[latcol])], crs=crs)
    return gdf

final_gdfs = []

########### 1. iNat ###########
inat_occ = inat_df_filtered.copy()
inat_occ['source'] = "iNaturalist"
inat_occ['host'] = "Ailanthus altissima"
inat_occ = gdf_from_coords(inat_occ, loncol="longitude", latcol="latitude")
inat_occ['latitude'] = inat_occ.geometry.y
inat_occ['longitude'] = inat_occ.geometry.x
inat_occ = inat_occ[['source', 'host', 'latitude', 'longitude', 'geometry']]

# final_gdfs.append(inat_occ)

########### 2. GBIF ###########
gbif_occ = gbif_df_filtered.copy()
gbif_occ['source'] = "GBIF"
gbif_occ['host'] = "Ailanthus altissima"
gbif_occ = gdf_from_coords(gbif_occ, loncol="decimalLongitude", latcol="decimalLatitude")
gbif_occ['latitude'] = gbif_occ.geometry.y
gbif_occ['longitude'] = gbif_occ.geometry.x
gbif_occ = gbif_occ[['source', 'host', 'latitude', 'longitude', 'geometry']]

# final_gdfs.append(gbif_occ)

########### 3+. State/Fed Datasets ###########
# Helper for common pattern: assign 'source', 'host', extract lat/lon from geometry, keep cols

def standardize_gdf(gdf, source_name):
    gdf2 = gdf.copy()
    gdf2['source'] = source_name
    gdf2['host'] = "Ailanthus altissima"
    # Reproject if needed to EPSG:4326
    if gdf2.crs and gdf2.crs.to_epsg() != 4326:
        gdf2 = gdf2.to_crs(epsg=4326)
    gdf2['latitude'] = gdf2.geometry.y
    gdf2['longitude'] = gdf2.geometry.x
    return gdf2[['source', 'host', 'latitude', 'longitude', 'geometry']]

# Example for each dataset; repeat for all!
final_gdfs.append(standardize_gdf(slf_2023_cumulative_filtered, "SLF_PA_Surveys"))
final_gdfs.append(standardize_gdf(ailanthus_visual_survey_INDNR_2023_filtered, "INDNR_2023"))
final_gdfs.append(standardize_gdf(ailanthus_visual_survey_NCDA_2023_filtered, "NCDA_2023"))
final_gdfs.append(standardize_gdf(ailanthus_visual_survey_NYSAGM_2023_filtered, "NYSAGM_2023"))
final_gdfs.append(standardize_gdf(ailanthus_visual_survey_2023_OHDA_filtered, "OHDA_2023"))
final_gdfs.append(standardize_gdf(ailanthus_visual_survey_2023_KY_filtered, "PPQ_KY_2023"))
final_gdfs.append(standardize_gdf(ailanthus_visual_survey_2023_TN_filtered, "PPQ_TN_2023"))
final_gdfs.append(standardize_gdf(ailanthus_trees_treatment_points_2019_filtered, "PA_Treatment_2019"))
final_gdfs.append(standardize_gdf(allstates_visual_surveys_2019_filtered, "Allstates_visual_2019"))
final_gdfs.append(standardize_gdf(ailanthus_combined_pda_fia_lcc_filtered, "PDA_FIA_LCC"))
# ... and rest of the state files
final_gdfs.append(standardize_gdf(ailanthus_ct_2020_filtered, "CT_2020"))
final_gdfs.append(standardize_gdf(ailanthus_de_2020_filtered, "DE_2020"))
final_gdfs.append(standardize_gdf(ailanthus_md_2020_filtered, "MD_2020"))
final_gdfs.append(standardize_gdf(ailanthus_md_2020_2_filtered, "MD_2020"))
final_gdfs.append(standardize_gdf(ailanthus_nj_2020_filtered, "NJ_2020"))
final_gdfs.append(standardize_gdf(ailanthus_nj_2020_2_filtered, "NJ_2020"))
final_gdfs.append(standardize_gdf(ailanthus_ny_2020_filtered, "NY_2020"))
final_gdfs.append(standardize_gdf(ailanthus_nysagm_filtered, "NYSAGM"))

# Concatenate all
occurrences_all = pd.concat(final_gdfs, ignore_index=True)
print(occurrences_all.head())

# To GIS file or CSV:
# occurrences_all.to_file("Y:/Ailanthus_NAIP_SDM/ailanthus_combined_occurrences.gpkg", driver="GPKG")
# occurrences_all.drop(columns="geometry").to_csv("Y:/Ailanthus_NAIP_SDM/ailanthus_combined_occurrences.csv", index=False)

#####################################
#    Deduplicate Occurrences        #
#####################################

print("Thinning occurrences to match resolution of WorldClim data...")

# Load WorldClim raster (any 1 km variable will work for cell ID reference)
worldclim_path = r"Y:\Ailanthus_NAIP_SDM\Env_Data\Worldclim\wc2.1_30s_bio_1.tif"
with rasterio.open(worldclim_path) as src:
    affine = src.transform
    raster_crs = src.crs
    width, height = src.width, src.height

# Reproject occurrences to WorldClim CRS if necessary
if occurrences_all.crs != raster_crs:
    occurrences_all = occurrences_all.to_crs(raster_crs)

# Get each point’s raster cell index (row, col)
rows, cols = rasterio.transform.rowcol(affine,
                                       occurrences_all.geometry.x,
                                       occurrences_all.geometry.y)

# Create a unique cell identifier
occurrences_all["cell_id"] = [f"{r}_{c}" for r, c in zip(rows, cols)]

# Deduplicate by cell_id, keeping one random occurrence per cell
occurrences_dedup = (
    occurrences_all.groupby("cell_id")
    .apply(lambda x: x.sample(1))
    .reset_index(drop=True)
)

print(f"Deduplicated from {len(occurrences_all)} to {len(occurrences_dedup)} occurrences.")

# To GIS file or CSV:
# occurrences_dedup.to_file("Y:/Ailanthus_NAIP_SDM/ailanthus_combined_occurrences_thin_wc1km.gpkg", driver="GPKG")
# occurrences_dedup.drop(columns="geometry").to_csv("Y:/Ailanthus_NAIP_SDM/ailanthus_combined_occurrences_thin_wc1km.csv", index=False)

################################
#   Sample Background Points   #
################################

# Sample background points equal to number of occurrences
n_background = len(occurrences_dedup)
print(f"Sampling {n_background} background points...")

# NAIP tiles US shapefile with URL paths for .tif files
naip_tiles_us_polygons = gpd.read_file(r"Y:\Ailanthus_NAIP_SDM\NAIP_Imagery_Tile_Indices\NAIP_US_State_Tile_Indices_URL_Paths_Jan25.shp")

if naip_tiles_us_polygons.crs is None:
    naip_tiles_us_polygons.set_crs(epsg=4326, inplace=True)

# Randomly sample background points within NAIP footprint
def random_points_in_polygon(polygon, n):
    """Uniform random sampling of n background points within polygon bounds."""
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < n:
        random_points = [
            Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            for _ in range(int(n * 2.0))  # oversample now to remove later
        ]
        inside = [pt for pt in random_points if polygon.contains(pt)]
        points.extend(inside)
        if len(points) > n:
            points = points[:n]
    return points

# Create NAIP footprint polygon (union)
# This will be used to sample random background points within NAIP coverage
print("Creating NAIP union polygon...")
naip_union = naip_tiles_us_polygons.unary_union

# Generate candidate points (oversample to filter later) within NAIP footprint
print("Generating candidate background points within NAIP footprint...")
candidate_background_points = random_points_in_polygon(naip_union, int(n_background * 2))
print(f"Generated {len(candidate_background_points)} candidate points.")

candidate_background_gdf = gpd.GeoDataFrame(geometry=candidate_background_points, crs="EPSG:4326")

# Reproject to WorldClim CRS
candidate_background_gdf = candidate_background_gdf.to_crs(raster_crs)

##########################################
# Assign each candidate background point #
# to a WorldClim cell ID                 #
##########################################

rows_bg, cols_bg = rasterio.transform.rowcol(
    affine,
    candidate_background_gdf.geometry.x,
    candidate_background_gdf.geometry.y
)
candidate_background_gdf["cell_id"] = [f"{r}_{c}" for r, c in zip(rows_bg, cols_bg)]

# Remove duplicates (retain only one background point per cell)
background_unique = (
    candidate_background_gdf.groupby("cell_id")
    .apply(lambda x: x.sample(1))
    .reset_index(drop=True)
)

# Remove background cells overlapping with presence cells
presence_cells = set(occurrences_dedup["cell_id"])
background_filtered = background_unique[~background_unique["cell_id"].isin(presence_cells)]
print(f"Filtered to {len(background_filtered)} unique background cells (no overlap with presences).")

# Randomly select background points equal to number of presences, in case we sampled too many
if len(background_filtered) > n_background:
    background_sampled = background_filtered.sample(n=n_background, random_state=42)
else:
    background_sampled = background_filtered.copy()
print(f"Selected {len(background_sampled)} background points for modeling.")

################################
#   Link to NAIP tile URLs     #
################################

# Reproject to NAIP CRS for spatial join
background_sampled = background_sampled.to_crs(naip_tiles_us_polygons.crs)
occurrences_dedup = occurrences_dedup.to_crs(naip_tiles_us_polygons.crs)

# Spatial join with NAIP tiles
print("Linking background and presence points to NAIP tile URLs...")
background_with_naip_tiles = gpd.sjoin(
    background_sampled, naip_tiles_us_polygons[["geometry", "url"]], how="left", predicate="within")

presences_with_naip_tiles = gpd.sjoin(
    occurrences_dedup, naip_tiles_us_polygons[["geometry", "url"]], how="left", predicate="within")

################################
#   Export combined dataset    #
################################

# Add columns for presence flag and source
presences_with_naip_tiles["presence"] = 1
presences_with_naip_tiles["source"] = occurrences_dedup.get("source", "unknown")

background_with_naip_tiles["presence"] = 0
background_with_naip_tiles["source"] = "background"

# Reproject to WGS84 for export
presences_with_naip_tiles = presences_with_naip_tiles.to_crs(epsg=4326)
background_with_naip_tiles = background_with_naip_tiles.to_crs(epsg=4326)

# Add latitude/longitude
presences_with_naip_tiles["lat"] = presences_with_naip_tiles.geometry.y
presences_with_naip_tiles["lon"] = presences_with_naip_tiles.geometry.x
background_with_naip_tiles["lat"] = background_with_naip_tiles.geometry.y
background_with_naip_tiles["lon"] = background_with_naip_tiles.geometry.x

# Reorder columns for export
presences_export_gdf = presences_with_naip_tiles[["lat", "lon", "url", "source", "geometry", "presence"]]
background_export_gdf = background_with_naip_tiles[["lat", "lon", "url", "source", "geometry", "presence"]]

# Combine presences and background points
presences_background_gdf = pd.concat([presences_export_gdf, background_export_gdf], ignore_index=True)

# Get length of final dataset for presences and background points
print(f"Final dataset contains {len(presences_background_gdf)} records (presences + background).")

# Num presences
print(f" - Presences: {presences_background_gdf['presence'].sum()}")
# Num background points
print(f" - Background points: {len(presences_background_gdf) - presences_background_gdf['presence'].sum()}")

# Export to shapefile and CSV
output_fp = r"Y:\Ailanthus_NAIP_SDM\Ailanthus_US_Occurrences\Ailanthus_US_APHIS_State_P_B_Filtered_Sampled_NAIP_Tiles.shp"
# presences_background_gdf.to_file(output_fp)
# presences_background_gdf.drop(columns="geometry").to_csv(output_fp.replace(".shp", ".csv"), index=False)

print(f"✅ Exported presences and background points to:")
print(f"   {output_fp}")
print(f"   {output_fp.replace('.shp', '.csv')}")

# EOF