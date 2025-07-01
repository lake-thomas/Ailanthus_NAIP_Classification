# A script to download species occurrence data from GBIF using pygbif.
# This script filters for Ailanthus altissima occurrences in the USA from 2015 to 2025.
# Author: Thomas Lake
# Date: June 30, 2025

# Imports
from pygbif import occurrences as occ

# Specify Taxon for Downloading
taxon_key = 3190653  # https://www.gbif.org/species/3190653 for Ailanthus altissima (Tree of Heaven)

# Load GBIF credentials from file
def load_gbif_credentials(filepath="gbif_creds.txt"):
    creds = {}
    with open(filepath, 'r') as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                creds[key.strip()] = value.strip()
    return creds

# Load credentials
creds = load_gbif_credentials(filepath=r"C:/Users/tomla/OneDrive/Desktop/Ailanthus_NAIP_Classification/credentials/gbif_creds.txt")
gbif_user = creds["GBIF_USER"]
gbif_pwd  = creds["GBIF_PWD"]
gbif_email = creds["GBIF_EMAIL"]

# Build the search predicate
# https://pygbif.readthedocs.io/en/latest/modules/occurrence.html
predicates = [
    {"type": "equals",          "key": "TAXON_KEY",                      "value": str(taxon_key)},
    {"type": "equals",          "key": "GADM_GID",                       "value": "USA.34_1"},
    {"type": "equals",          "key": "BASIS_OF_RECORD",                "value": "HUMAN_OBSERVATION"},
    {"type": "equals",          "key": "HAS_COORDINATE",                 "value": "true"},
    {"type": "equals",          "key": "HAS_GEOSPATIAL_ISSUE",           "value": "false"},
    {"type": "equals",          "key": "OCCURRENCE_STATUS",              "value": "present"},
    {
        "type": "and",
        "predicates": [
            {"type": "greaterThanOrEquals", "key": "COORDINATE_UNCERTAINTY_IN_METERS", "value": "0"},
            {"type": "lessThanOrEquals",    "key": "COORDINATE_UNCERTAINTY_IN_METERS", "value": "120"},
        ]
    },
    {
        "type": "and",
        "predicates": [
            {"type": "greaterThanOrEquals", "key": "YEAR", "value": "2015"},
            {"type": "lessThanOrEquals",    "key": "YEAR", "value": "2025"},
        ]
    }
]

search_predicate = {
    "type": "and",
    "predicates": predicates
}

# Submit download request (simple CSV)
key = occ.download(
    queries=search_predicate,
    user=gbif_user,
    pwd=gbif_pwd,
    email=gbif_email,
    format="SIMPLE_CSV"
)
print("Download key:", key)
print(f"Monitor status at: https://www.gbif.org/occurrence/download/{key}")
# Note: The download will be processed asynchronously. You can check the status using the provided link.
# EOF








