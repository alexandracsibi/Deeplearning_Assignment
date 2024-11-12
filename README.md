# DeepLearning_Project
**Name and Neptun code:** <br>
- Csibi Alexandra, GPVFEV

**Project description** <br>
*Disease-gene interaction prediction with graph neural networks* <br>
The goal of this project is to create a graph neural network for predicting disease-gene associations. Working with DisGeNET, a comprehensive database of these associations, you'll apply deep learning to an important challenge of bioinformatics. By choosing this project, you'll gain experience in the intersection of deep learning and bioinformatics while extracting valuable insights from real-world data.

**Related works (papers, GitHub repositories, blog posts, etc)** <br>
- [Related GitHub repository](https://github.com/pyg-team/pytorch_geometric)
- [Related GitHub repository](https://github.com/sujitpal/pytorch-gnn-tutorial-odsc2021)
- [Related YouTube video](https://www.youtube.com/watch?v=-UjytpbqX4A&list=LL&index=1)
- [Dataset](https://www.disgenet.org/)

Welcome to the Deep Learning Assignment repository. This repository contains code, data, and resources for building and evaluating graph neural network (GNN) models to predict gene-disease associations. Below is a detailed overview of the repository's structure and its components.

**Repository Structure** <br>

<details>
<summary><strong>1. Data</strong></summary>

- **`GDA_df.csv`**: Contains preprocessed data for gene-disease associations, ready to be used by the graph models.  
- **`graph_data.pkl`**: A serialized file containing the homogeneous graph data used by the models.  

</details>

---

<details>
<summary><strong>2. Models</strong></summary>

The repository includes four graph neural network (GNN) models:

### GCN_DP
- **Architecture**: Two `GCNConv` layers for encoding, followed by a simple dot product for classification.  
- **Use Case**: Basic model for link prediction.  

### GCN_MLP
- **Architecture**: Two `GCNConv` layers for encoding, followed by a Multilayer Perceptron (MLP) for decoding.  
- **Use Case**: Improved model with more complex decoding.  

### GraphSAGE_MLP
- **Architecture**: Two `SAGEConv` layers for encoding, followed by an MLP for decoding.  
- **Use Case**: Handles larger graphs by sampling neighbors during training.  

### GIN_MLP
- **Architecture**: Two `GINConv` layers for encoding, followed by an MLP for decoding.  
- **Use Case**: Captures structural properties of the graph more effectively.  

**Model Weights**:  
The best-performing configurations are saved in the following files:  
- `GCN_DP.pth`  
- `GCN_MLP.pth`  
- `GraphSAGE_MLP.pth`  
- `GIN_MLP.pth`  

</details>

---

<details>
<summary><strong>3. Code Components</strong></summary>

### Data Handling
#### `data_acquisition_processing.py`
- Downloads and processes raw data to prepare it for GNN models.  
- Provides the `get_data` function:  
  - **Parameters**: `disgenet_api_key`, `disease_type`.  
  - Fetches and processes associations based on disease type.  

#### `graph_preparation.py`
- Prepares the homogeneous graph for training:  
  - **Node Features**: Combines gene and disease features, along with a node type indicator (1 for genes, 0 for diseases).  
  - **Edge Index**: Assigns unique IDs to nodes and creates edges based on associations. Reverse edges are added for better embeddings.  

### Model Training
#### `trainer.py`
Implements a `Trainer` class for training, evaluating, and testing models:
- **Training**:  
  - Encodes node embeddings and decodes edge predictions.  
  - Computes loss for positive and negative edges using `BCEWithLogitsLoss`.  
  - Supports learning rate scheduling (`CosineAnnealingLR`) and early stopping.  
- **Evaluation**:  
  - Computes AUC, F1 score, and confusion matrix.  
  - Dynamically determines the best threshold for binary classification.  
- **Testing**:  
  - Loads the best model checkpoint to evaluate test performance.  

### Interactive Application
#### `gradio_app.py`
- Provides a Gradio-based interface for user input:  
  - Accepts a gene ID and disease ID.  
  - Fetches features from the DisGeNET API.  
  - Processes input to match the training graph format.  
  - Updates the graph if the gene or disease is missing.  
  - Uses the loaded model to predict associations and outputs binary classification based on the best threshold.  

</details>

---

<details>
<summary><strong>4. Containerization</strong></summary>

### `Dockerfile`
- Sets up the environment for running either JupyterLab (for development) or Gradio (for user interaction).  
- Includes all dependencies for data processing, model training, and serving.  

### `start.sh`
- Script for running the Docker container:  
  - Option to start JupyterLab or the Gradio application.  

</details>

**Installation Instructions** <br>

**Usage** <br>
