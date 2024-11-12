from models import GCN_DP, GCN_MLP, GraphSAGE_MLP, GIN_MLP
import pandas as pd
import requests
import re
import torch
import gradio as gr
import pickle

def fetch_gene_features(gene_id, api_key):
    url = "https://api.disgenet.com/api/v1/entity/gene"
    params = {"gene_ncbi_id": gene_id}
    headers = {'Authorization': api_key, 'accept': 'application/json'}
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
          response_json = response.json()
          data = response_json.get("payload", [])
          records = []
          for item in data:
              records.append({
                  "geneDSI": item.get("dsi", 0),
                  "geneDPI": item.get("dpi", 0),
                  "geneNcbiType": item.get("ncbi_type", "unknown"),
              })
          return pd.DataFrame(records)

    else:
        raise ValueError(f"Error fetching gene features: {response.status_code}")
    
def fetch_disease_features(disease_id, api_key):
    url = "https://api.disgenet.com/api/v1/entity/disease"
    params = {"disease": disease_id}
    headers = {'Authorization': api_key, 'accept': 'application/json'}
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
          response_json = response.json()
          data = response_json.get("payload", [])
          records = []
          for item in data:
              records.append({
                  "diseaseType": item.get("type", "unknown"),
                  "diseaseClasses_MSH": item.get("diseaseClasses_MSH", "unknown"),
                  "diseaseClasses_UMLS_ST": item.get("diseaseClasses_UMLS_ST", "unknown"),
              })

          return pd.DataFrame(records)

    else:
        raise ValueError(f"Error fetching gene features: {response.status_code}")
    
def extract_parentheses_values(entry):
    """Extract values enclosed in parentheses from a string or list of strings."""
    if isinstance(entry, str):
        return re.findall(r'\((.*?)\)', entry)
    elif isinstance(entry, list):
        result = []
        for item in entry:
            result.extend(re.findall(r'\((.*?)\)', item))
        return result
    return []

def process_gene_disease_features(df, disease_features, gene_features):

    gene_columns = ['geneDSI', 'geneDPI'] + [col for col in df.columns if col.startswith('geneType')]
    disease_columns = [col for col in df.columns if col.startswith('diseaseClass') or col.startswith('diseaseType')]

    all_columns = gene_columns + disease_columns + ['nodetype']
    base_template = {col: 0 for col in all_columns}

    # Process the gene features
    processed_gene_features = base_template.copy()
    processed_gene_features['geneDSI'] = gene_features['geneDSI'].iloc[0]
    processed_gene_features['geneDPI'] = gene_features['geneDPI'].iloc[0]

    # Dynamically add the one-hot encoded gene type
    gene_type_column = f"geneType_{gene_features['geneNcbiType'].iloc[0]}"
    if gene_type_column in processed_gene_features:
        processed_gene_features[gene_type_column] = 1

    processed_gene_features['nodetype'] = 1

    processed_disease_features = base_template.copy()

    # Dynamically add the one-hot encoded disease type
    disease_type_column = f"diseaseType_{disease_features['diseaseType'].iloc[0]}"
    if disease_type_column in processed_disease_features:
        processed_disease_features[disease_type_column] = 1

    msh_classes = extract_parentheses_values(disease_features.get('diseaseClasses_MSH', ""))
    umls_classes = extract_parentheses_values(disease_features.get('diseaseClasses_UMLS_ST', ""))
    disease_classes = set(msh_classes + umls_classes)

    for disease_class in disease_classes:
        disease_class_column = f"diseaseClass_{disease_class}"
        if disease_class_column in processed_disease_features:
            processed_disease_features[disease_class_column] = 1

    # Add the nodetype indicator for diseases
    processed_disease_features['nodetype'] = 0  # 0 for diseases

    # Convert processed features to DataFrames for consistency
    processed_gene_df = pd.DataFrame([processed_gene_features])
    processed_disease_df = pd.DataFrame([processed_disease_features])

    gene_node_feature = torch.tensor(processed_gene_df.values, dtype=torch.float)
    disease_node_feature = torch.tensor(processed_disease_df.values, dtype=torch.float)

    return gene_node_feature, disease_node_feature

def check_and_assign_ids(graph_data, gene_node_feature, disease_node_feature):
    """
    Check if the provided gene and disease features exist in the graph.
    If not, add them to the graph, ensuring gene nodes are added after the current gene nodes
    and disease nodes after the current disease nodes. Assign IDs accordingly.
    """
    node_features = graph_data.x
    nodetype_indicator = node_features[:, -1]

    gene_features = node_features[nodetype_indicator == 1]
    disease_features = node_features[nodetype_indicator == 0]

    # Check for gene node existence
    gene_id = None
    if (gene_features == gene_node_feature).all(dim=1).any():
        gene_id = torch.where((gene_features == gene_node_feature).all(dim=1))[0][0].item()
    else:
        gene_id = gene_features.size(0)
        node_features = torch.cat([node_features, gene_node_feature], dim=0)

    # Check for disease node existence
    disease_id = None
    if (disease_features == disease_node_feature).all(dim=1).any():
        disease_id = torch.where((disease_features == disease_node_feature).all(dim=1))[0][0].item() + gene_features.size(0)
    else:
        disease_id = node_features.size(0)  # Append disease after all existing nodes
        node_features = torch.cat([node_features, disease_node_feature], dim=0)

    graph_data.x = node_features

    return graph_data, gene_id, disease_id

# Define global variables
GDA_df = pd.read_csv('GDA_df.csv')
df = GDA_df.copy()

# Load the graph and model
with open("graph_data.pkl", "rb") as f:
    graph_data = pickle.load(f)

API_KEY = "ad6669df-65b6-45f9-8e02-7ba74e788acd"

# Define a function to load models dynamically
def load_model(model_name):
    checkpoint = torch.load(f"{model_name}.pth")
    if model_name == "GIN_MLP":
        model = GIN_MLP(input_dim=48, hidden_dim=128, output_dim=64)
    elif model_name == "GCN_DP":
        model = GCN_DP(input_dim=48, hidden_dim=128, output_dim=64)
    elif model_name == "GCN_MLP":
        model = GCN_MLP(input_dim=48, hidden_dim=128, output_dim=64)
    elif model_name == "GraphSAGE_MLP":
        model = GraphSAGE_MLP(input_dim=48, hidden_dim=128, output_dim=64)
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    best_threshold = checkpoint['best_threshold']
    return model, best_threshold

# Prediction function
def gradio_predict(gene_id: int, disease_id: str, model_name: str):
    try:
        # Strip whitespace from inputs
        gene_id = gene_id.strip()
        disease_id = disease_id.strip()

        # Load the selected model
        model, best_threshold = load_model(model_name)

        # Fetch features for the gene and disease
        gene_features = fetch_gene_features(gene_id, API_KEY)
        disease_features = fetch_disease_features(disease_id, API_KEY)

        # Process the fetched features into node features
        processed_gene_feature, processed_disease_feature = process_gene_disease_features(
            df, disease_features, gene_features
        )

        # Check and assign IDs in the graph
        updated_graph, gene_idx, disease_idx = check_and_assign_ids(
            graph_data, processed_gene_feature, processed_disease_feature
        )

        # Prepare edge indices for prediction
        edge_label_index = torch.tensor([[gene_idx], [disease_idx]], dtype=torch.long)

        # Predict association
        with torch.no_grad():
            z = model.encode(updated_graph.x, updated_graph.edge_index)  # Encode node embeddings
            raw_score = model.decode(z, edge_label_index).sigmoid().item()  # Get the sigmoid of the predicted score

        # Apply the best threshold
        prediction = 1 if raw_score >= best_threshold else 0

        # Return results
        result = f"Prediction: {'Exists' if prediction == 1 else 'Does not exist'}"
        raw_score_str = f"Raw Score: {raw_score:.4f}"
        return result, raw_score_str

    except Exception as e:
        return f"Error: {str(e)}", ""

# Gradio interface
model_selector = gr.Radio(
    choices=["GIN_MLP", "GCN_DP", "GCN_MLP", "GraphSAGE_MLP"],
    label="Select a Model",
    value="GIN_MLP"  # Default model
)

interface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Textbox(label="Gene ID, the Entrez Id from Disgenet (e.g., 7124)", placeholder="Enter Gene ID here"),
        gr.Textbox(label="Disease ID (e.g., MONDO_0000728)", placeholder="Enter Disease ID here"),
        model_selector
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Raw Score")
    ],
    title="Gene-Disease Association Prediction",
    description="Enter Gene and Disease IDs to predict the association using the trained model."
)

# Launch Gradio interface
interface.launch(server_name="0.0.0.0")