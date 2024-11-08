# -*- coding: utf-8 -*-
"""data_acquisition_processing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ULFCdhBgLb3WP-Xnx80CVQmzGk0jimMU
"""

import os
import requests
import time
import pandas as pd
import argparse
import json
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer

# API endpoints
url_gda = "https://api.disgenet.com/api/v1/gda/summary"
url_disease = "https://api.disgenet.com/api/v1/entity/disease"

# Function to handle API requests with rate-limiting handling
def make_request(url, params, headers):
  retries = 0
  while retries < 5:
    try:
      response = requests.get(url, params=params, headers=headers, timeout=10)
      # If rate-limited (HTTP 429), retry after waiting
      if response.status_code == 429:
        wait_time = int(response.headers.get('x-rate-limit-retry-after-seconds', 60))
        print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
        time.sleep(wait_time)
        retries += 1
      else:
        return response  # Return response if successful or error other than 429
    except requests.exceptions.RequestException as e:
      print(f"Request error: {e}")
      retries += 1
      time.sleep(2)  # Wait before retrying

  return None  # Return None if retries are exhausted

# Function to get the maximum number of pages for the data
def get_max_pages(url, params, headers):
  response = make_request(url, params, headers)
  if response and response.ok:
    response_json = response.json()
    total_results = response_json.get("paging", {}).get("totalElements", 0)
    results_in_page = response_json.get("paging", {}).get("totalElementsInPage", 0)
    max_pages = min((total_results + results_in_page - 1) // results_in_page, 100)
  else:
    max_pages = 100
    print("Request failed, returned max_pages=100")
  return max_pages

# Function to retrieve disease IDs based on disease type
def get_disease_ids(disease_type, headers):
  disease_ids = []
  params = {"page_number": 0, "type": "disease", "disease_free_text_search_string": disease_type}

  max_page = get_max_pages(url_disease, params, headers)
  for page in range(max_page):
    params['page_number'] = str(page)
    response_disease = make_request(url_disease, params, headers)
    if response_disease and response_disease.ok:
      response_disease_json = response_disease.json()
      data = response_disease_json.get("payload", [])
      for item in data:
        for code_info in item.get("diseaseCodes", []):
          if code_info.get("vocabulary") == "MONDO":
              disease_ids.append(f'MONDO_{code_info.get("code")}')
    else:
      print(f"Failed to fetch data for page {page}. Status code: {response_disease_json.status_code}")
      break
  return list(set(disease_ids))

# Function to download GDA data for a list of disease IDs
def download_gda(disease_ids, headers):
  gda_data = []
  params = {"page_number": 0, "type": "disease", "disease": disease_ids}
  max_page = get_max_pages(url_gda, params, header)
  for page in range(max_page):
    params['page_number'] = str(page)
    response_gda = make_request(url_gda, params, headers)
    if response_gda and response_gda.ok:
      response_json = response_gda.json()
      data = response_json.get("payload", [])
      gda_data.extend(data)
    else:
      print(f"Failed to fetch data for page {page}. Status code: {response_json.status_code}")
      break

  return gda_data

# Function to download all GDA data in chunks
def download_all_gda(headers, ids, chunk_size=100):
  all_data = []
  for i in range(0, len(ids), chunk_size):
    ids_chunk = ids[i:i + chunk_size]
    ids_string = '"' + ', '.join(ids_chunk) + '"'
    chunk_data = download_gda(ids_string, headers)
    all_data.extend(chunk_data)
  df_gda = pd.DataFrame(all_data)
  return df_gda

def clean_and_encode_data(GDA_df):
  # Replace '[]' with NaN and encode diseaseUMLSCUI
  GDA_df = GDA_df.applymap(lambda x: np.nan if x == '[]' else x)
  label_encoder = LabelEncoder()
  GDA_df['diseaseUMLSCUI_encoded'] = label_encoder.fit_transform(GDA_df['diseaseUMLSCUI'])

  # Drop duplicates based on assocID
  GDA_df = GDA_df.drop_duplicates(subset=['assocID']).reset_index(drop=True)

  # Select specific columns for further processing
  columns = [
      'geneNcbiID', 'geneDSI', 'geneDPI', 'geneNcbiType',
      'diseaseUMLSCUI_encoded', 'diseaseClasses_MSH',
      'diseaseClasses_UMLS_ST', 'assocID', 'score'
  ]
  GDA_df = GDA_df[columns]
  return GDA_df

def one_hot_encode_gene_type(GDA_df):
  # One-hot encode geneNcbiType column
  enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
  encoded_geneNcbiType = enc.fit_transform(GDA_df[['geneNcbiType']])
  columns = ['geneType_' + col.split('_')[-1] for col in enc.get_feature_names_out(['geneNcbiType'])]
  encoded_df = pd.DataFrame(encoded_geneNcbiType, columns=columns)
  GDA_df = pd.concat([GDA_df.reset_index(drop=True), encoded_df], axis=1).drop('geneNcbiType', axis=1)
  return GDA_df

# Keep only IDs for simplicity
def clean_classes(entry):
  if isinstance(entry, (str, bytes)):
    return [match.strip() for match in re.findall(r'\((.*?)\)', entry)]
  else:
    return []

def process_and_encode_disease_classes(GDA_df):
    # Apply cleaning to disease class columns
    GDA_df['diseaseClasses_UMLS_ST'] = GDA_df['diseaseClasses_UMLS_ST'].apply(clean_classes)
    GDA_df['diseaseClasses_MSH'] = GDA_df['diseaseClasses_MSH'].apply(clean_classes)

    # Combine the two lists into a new column for handling missing values in diseaseClasses_MSH
    GDA_df['diseaseClass'] = GDA_df.apply(
        lambda row: list(set(row['diseaseClasses_UMLS_ST'] + row['diseaseClasses_MSH'])),
        axis=1
    )

    # Multi-label binarize combined disease class column
    mlb = MultiLabelBinarizer()
    encoded_diseaseClass = mlb.fit_transform(GDA_df['diseaseClass'])
    enc_df = pd.DataFrame(encoded_diseaseClass, columns=['diseaseClass_' + cols for cols in mlb.classes_])
    GDA_df = pd.concat([GDA_df.reset_index(drop=True), enc_df], axis=1)

    # Drop unnecessary columns
    GDA_df = GDA_df.drop(['diseaseClasses_UMLS_ST', 'diseaseClasses_MSH', 'diseaseClass'], axis=1)
    return GDA_df

def rename_and_index(GDA_df):
  GDA_df.rename(columns={'geneNcbiID': 'geneID', 'diseaseUMLSCUI_encoded': 'diseaseID'}, inplace=True)

  # Create unique indices for geneID and diseaseID
  unique_gene_ids = GDA_df['geneID'].unique()
  unique_disease_ids = GDA_df['diseaseID'].unique()

  # geneIds 0 to len(unique_gene_ids) and diseaseIds len(unique_gene_ids) to len(unique_gene_ids) + len(unique_disease_ids)
  gene_id_to_idx = {id: idx for idx, id in enumerate(unique_gene_ids)}
  disease_id_to_idx = {id: idx + len(unique_gene_ids) for idx, id in enumerate(unique_disease_ids)}

  GDA_df['geneID'] = GDA_df['geneID'].map(gene_id_to_idx)
  GDA_df['diseaseID'] = GDA_df['diseaseID'].map(disease_id_to_idx)
  GDA_df['assocID'] = range(0, len(GDA_df))

  return GDA_df

def process_data(GDA_df):
  GDA_df = clean_and_encode_data(GDA_df)
  GDA_df = one_hot_encode_gene_type(GDA_df)
  GDA_df = process_and_encode_disease_classes(GDA_df)
  GDA_df = rename_and_index(GDA_df)

  return GDA_df

def get_data(api_key, disease_type):
    headers = {
        'Authorization': api_key,
        'accept': 'application/json'
    }

    # Retrieve disease IDs
    disease_ids = get_disease_ids(disease_type, headers)
    print(f"Number of unique disease IDs retrieved: {len(disease_ids)}")

    GDA_df = download_all_gda(headers, disease_ids)
    print("Data acquisition completed. DataFrame created.")

    GDA_df = process_data(GDA_df)

    return GDA_df


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Download and process GDA data for a specified disease type.")
  parser.add_argument("api_key", type=str, help="API key for accessing the DisGeNET API")
  parser.add_argument("disease_type", type=str, help="Type of disease to download data for (e.g., 'cancer')")
  args = parser.parse_args()

  df = get_data(args.api_key, args.disease_type)