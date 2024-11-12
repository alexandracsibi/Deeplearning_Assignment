import numpy as np
import torch
import torch_geometric as pyg

def homogeneous_node_features(df):
        '''Preprocess and construct node features for genes and diseases for homogeneous graph'''
        # Extract unique rows for genes and diseases
        gene_rows = df[
            ['geneID', 'geneDSI', 'geneDPI'] +
            [col for col in df.columns if col.startswith('geneType')]]
        gene_rows = gene_rows.drop_duplicates(subset=['geneID']).drop(columns=['geneID'])

        disease_rows = df[
            ['diseaseID'] +
            [col for col in df.columns if col.startswith('diseaseClass')] +
            [col for col in df.columns if col.startswith('diseaseType')]]
        disease_rows = disease_rows.drop_duplicates(subset=['diseaseID']).drop(columns=['diseaseID'])

        # Fill missing columns with zeros where needed
        gene_rows = gene_rows.assign(**{col: 0 for col in disease_rows.columns if col not in gene_rows.columns})
        disease_rows = disease_rows.assign(**{col: 0 for col in gene_rows.columns if col not in disease_rows.columns})

        # Convert features to numpy arrays and add node type indicator
        gene_features = np.hstack([gene_rows.values, np.ones((gene_rows.shape[0], 1))]) # 1 indicates gene
        disease_features = np.hstack([disease_rows.values, np.zeros((disease_rows.shape[0], 1))]) # 0 indicates disease

        # Combine gene and disease features into a single matrix and return as tensor
        node_features = np.vstack([gene_features, disease_features])

        return torch.tensor(node_features, dtype=torch.float)

def prepare_homogeneous_graph(df):
        '''Prepare a homogeneous graph for PyTorch Geometric'''
        # Map IDs to separate index ranges
        unique_gene_ids = df['geneID'].unique()
        unique_disease_ids = df['diseaseID'].unique()

        # geneIds 0 to len(unique_gene_ids) and diseaseIds len(unique_gene_ids) to len(unique_gene_ids) + len(unique_disease_ids)
        gene_id_to_idx = {id: idx for idx, id in enumerate(unique_gene_ids)}
        disease_id_to_idx = {id: idx + len(unique_gene_ids) for idx, id in enumerate(unique_disease_ids)}

        df['geneID'] = df['geneID'].map(gene_id_to_idx)
        df['diseaseID'] = df['diseaseID'].map(disease_id_to_idx)

        # Construct node features
        node_features = homogeneous_node_features(df)

        # Create edge indices
        edge_index = torch.tensor(np.array([df['geneID'].values, df['diseaseID'].values]), dtype=torch.long)
        edge_index = pyg.utils.to_undirected(edge_index) # Make edges bidirectional

        # Homogeneous Graph
        graph_data = pyg.data.Data(
            x = node_features,
            edge_index = edge_index,
        )

        return graph_data