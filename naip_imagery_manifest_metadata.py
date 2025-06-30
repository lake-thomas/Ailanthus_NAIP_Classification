# This script processes a manifest file from AWS NAIP S3 bucket to extract metadata about NAIP imagery.
# https://github.com/awslabs/open-data-docs/tree/main/docs/naip
# June 30, 2025
# Thomas Lake; Felipe Sanchez

# Imports
import pandas as pd

# AWS NAIP S3 bucket manifest
# This script reads a manifest text file from an AWS NAIP S3 bucket, processes it, and extracts relevant information.

#Read the manifest text file
manifest_file = r'C:/Users/talake2/Desktop/Ailanthus_NAIP_Classification/naip_manifest_aws_june3025.txt'

# Read the text file into a DataFrame
df = pd.read_csv(manifest_file, header=None, delimiter='\t')

# Split the values in the first column based on the '/' delimiter and put them into separate columns
df = df[0].str.split('/', expand=True)

# Select the first six columns if there are more than six columns
df = df.iloc[:, :6]

# Assign column names
df.columns = ['State', 'Year', 'Resolution', 'File_Type', 'Index', 'Quad']

#print(df)

#         State  Year Resolution  File_Type  Index                              Quad
# 0          al  2011      100cm       fgdc  30085    m_3008501_ne_16_1_20110815.txt
# 1          al  2011      100cm       fgdc  30085    m_3008501_nw_16_1_20110815.txt
# 2          al  2011      100cm       fgdc  30085    m_3008502_ne_16_1_20110815.txt
# 3          al  2011      100cm       fgdc  30085    m_3008502_nw_16_1_20110815.txt
# 4          al  2011      100cm       fgdc  30085    m_3008503_ne_16_1_20110815.txt
# ...       ...   ...        ...        ...    ...                               ...
# 7000580    wa  2021       60cm  rgbir_cog  49119  m_4911962_sw_11_060_20211003.tif
# 7000581    wa  2021       60cm  rgbir_cog  49119  m_4911963_se_11_060_20210627.tif
# 7000582    wa  2021       60cm  rgbir_cog  49119  m_4911963_sw_11_060_20210627.tif
# 7000583    wa  2021       60cm  rgbir_cog  49119  m_4911964_se_11_060_20210627.tif
# 7000584    wa  2021       60cm  rgbir_cog  49119  m_4911964_sw_11_060_20210627.tif

# # Check for rows with unexpected values in the 'State' column (e.g., readme.html)
invalid_state_rows = df[~df['State'].str.match(r'^[a-zA-Z]{2}$')]

if not invalid_state_rows.empty:
    print("Rows with invalid states:")
    print(invalid_state_rows)
    
    # Remove rows with invalid states from the DataFrame
    df = df.drop(invalid_state_rows.index)
    print("Rows with invalid states removed.")
else:
    print("No rows with invalid states found.")

# # Remove rows with Hawaii
hawaii_rows = df[df['State'] == 'hi']
df = df.drop(hawaii_rows.index)

# Convert 'Year' to numeric type
df.loc[:, 'Year'] = pd.to_numeric(df['Year'])

# # Group by 'State' and find the maximum 'Year' for each group
most_recent_years = df.groupby('State')['Year'].max()

print(most_recent_years)

# Extract the corresponding 'Resolution' for the most recent year for each state
for state, max_year in most_recent_years.items():
    recent_entries = df[(df['State'] == state) & (df['Year'] == max_year)]
    if not recent_entries.empty:
        resolution = recent_entries['Resolution'].iloc[0]
        print(f"For {state}, the most recent year is {max_year} with resolution {resolution}")
    else:
        print(f"No data found for {state}")


# EOF