import requests
import time
import pandas as pd
import json
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer

# API endpoints
url_gda = "https://api.disgenet.com/api/v1/gda/summary"
url_disease = "https://api.disgenet.com/api/v1/entity/disease"

def make_request(url, params, headers):
    """Makes a request to the specified URL with retries for handling rate limits."""
    retries = 0
    while retries < 5:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 429: # Rate limit
                wait_time = int(response.headers.get('x-rate-limit-retry-after-seconds', 60))
                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                return response

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            retries += 1
            time.sleep(2)  # Wait before retrying

    return None

def get_max_pages(url, params, headers):
    response = make_request(url, params, headers)
    if response and response.ok:
        response_json = response.json()
        total_results = response_json.get("paging", {}).get("totalElements", 0)
        results_in_page = response_json.get("paging", {}).get("totalElementsInPage", 0)
        max_pages = min((total_results + results_in_page - 1) // results_in_page, 100)
    else:
        max_pages = 100
        print("The total exceeds 100 pages."
              "Returning 100 pages as it is the maximum that can be processed.")
    return max_pages

def get_disease_ids(disease_type, headers):
    """Retrieves disease IDs for a specified disease type."""
    disease_ids = []
    params = {"page_number": 0, "type": "disease", "disease_free_text_search_string": disease_type}
    max_page = get_max_pages(url_disease, params, headers)

    for page in range(max_page):
        params['page_number'] = str(page)
        response = make_request(url_disease, params, headers)
        if response and response.ok:
          response_json = response.json()
          data = response_json.get("payload", [])
          for item in data:
              for code_info in item.get("diseaseCodes", []):
                  if code_info.get("vocabulary") == "MONDO":
                      disease_ids.append(f'MONDO_{code_info.get("code")}')
        else:
            print(f"Failed to fetch data for page {page}. Status code: {response_json.status_code}")
            break
    return list(set(disease_ids))

def download_gda(disease_ids, headers):
    """Downloads GDA data for the provided disease IDs."""
    gda_data = []
    params = {"page_number": 0, "type": "disease", "disease": disease_ids}
    max_page = get_max_pages(url_gda, params, headers)
    
    for page in range(max_page):
        params['page_number'] = str(page)
        response = make_request(url_gda, params, headers)
        if response and response.ok:
            response_json = response.json()
            data = response_json.get("payload", [])
            gda_data.extend(data)
        else:
            print(f"Failed to fetch data for page {page}. Status code: {response_json.status_code}")
            break

    return gda_data

def download_all_gda(headers, ids, chunk_size=100):
    """Downloads all GDA data in chunks to handle API limits."""
    all_data = []

    for i in range(0, len(ids), chunk_size):
        ids_chunk = ids[i:i + chunk_size]
        ids_string = '"' + ', '.join(ids_chunk) + '"'
        chunk_data = download_gda(ids_string, headers)
        all_data.extend(chunk_data)
    return pd.DataFrame(all_data)

def clean_and_encode_data(df):
    """Cleans and encodes the raw GDA data."""
    df.replace('[]', np.nan, inplace=True)
    label_encoder = LabelEncoder()
    df['diseaseUMLSCUI'] = label_encoder.fit_transform(df['diseaseUMLSCUI'])

    # Drop duplicates based on assocID
    df = df.drop_duplicates(subset=['assocID']).reset_index(drop=True)

    # Select specific columns for further processing
    columns = ['geneNcbiID', 'geneDSI', 'geneDPI',
              'geneNcbiType', 'diseaseUMLSCUI', 'diseaseClasses_MSH',
              'diseaseClasses_UMLS_ST', 'diseaseType', 'assocID', 'score']
    df = df[columns]
    return df

def one_hot_encode_gene_type(df):
    """One-hot encodes the 'geneNcbiType' column."""
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = enc.fit_transform(df[['geneNcbiType']])
    encoded_columns = ['geneType_' + col.split('_')[-1] for col in enc.get_feature_names_out(['geneNcbiType'])]
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns)
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1).drop('geneNcbiType', axis=1)
    return df

def clean_classes(entry):
    """Extracts IDs from class strings."""
    if isinstance(entry, (str, bytes)):
        return [match.strip() for match in re.findall(r'\((.*?)\)', entry)]
    else:
        return []

def process_and_encode_disease_classes(df):
    """Processes and encodes disease class information."""
    df['diseaseClasses_UMLS_ST'] = df['diseaseClasses_UMLS_ST'].apply(clean_classes)
    df['diseaseClasses_MSH'] = df['diseaseClasses_MSH'].apply(clean_classes)

    # Combine the two lists into a new column for handling missing values in diseaseClasses_MSH
    df['diseaseClass'] = df.apply(
        lambda row: list(set(row['diseaseClasses_UMLS_ST'] + row['diseaseClasses_MSH'])),
        axis=1
    )

    # Multi-label binarize combined disease class column
    mlb = MultiLabelBinarizer()
    encoded_diseaseClass = mlb.fit_transform(df['diseaseClass'])
    enc_df = pd.DataFrame(encoded_diseaseClass, columns=['diseaseClass_' + cols for cols in mlb.classes_])
    df = pd.concat([df.reset_index(drop=True), enc_df], axis=1)

    # Drop unnecessary columns
    df = df.drop(['diseaseClasses_UMLS_ST', 'diseaseClasses_MSH', 'diseaseClass'], axis=1)
    df.rename(columns={'geneNcbiID': 'geneID', 'diseaseUMLSCUI': 'diseaseID'}, inplace=True)
    return df


def process_data(df):
    """Cleans, encodes, and processes the GDA DataFrame."""
    df = clean_and_encode_data(df)
    df = one_hot_encode_gene_type(df)
    df = process_and_encode_disease_classes(df)

    return df

def get_data(api_key, disease_type):
    """Main function to retrieve and process GDA data."""
    headers = {
        'Authorization': api_key,
        'accept': 'application/json'
    }

    # Retrieve disease IDs
    disease_ids = get_disease_ids(disease_type, headers)
    print(f"Number of unique disease IDs retrieved: {len(disease_ids)}")

    # Limit the number of disease IDs if it exceeds 2000
    if len(disease_ids) > 2000:
        print(f"Number of disease IDs exceeds 2000. Limiting to the first 2000 IDs.")
        disease_ids = disease_ids[:2000]

    raw_df = download_all_gda(headers, disease_ids)
    print("Data acquisition completed. Processing data...")

    processed_df = process_data(raw_df)
    print("Data processing complete.")
    print(f"Number of unique gene IDs: {len(processed_df['geneID'].unique())}")
    print(f"Number of unique disease IDs: {len(processed_df['diseaseID'].unique())}")
    print(f"Number of unique assocIDs: {len(processed_df['assocID'].unique())}")

    return processed_df