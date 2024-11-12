# DeepLearning_Project
**Name and Neptun code:** <br>
- Csibi Alexandra, GPVFEV

## Project description
*Disease-gene interaction prediction with graph neural networks* <br>
The goal of this project is to create a graph neural network for predicting disease-gene associations. Working with DisGeNET, a comprehensive database of these associations, you'll apply deep learning to an important challenge of bioinformatics. By choosing this project, you'll gain experience in the intersection of deep learning and bioinformatics while extracting valuable insights from real-world data.

**Related works (papers, GitHub repositories, blog posts, etc)** <br>
- [Related GitHub repository](https://github.com/pyg-team/pytorch_geometric)
- [Related GitHub repository](https://github.com/sujitpal/pytorch-gnn-tutorial-odsc2021)
- [Related YouTube video](https://www.youtube.com/watch?v=-UjytpbqX4A&list=LL&index=1)
- [Dataset](https://www.disgenet.org/)

**Welcome to the Deep Learning Assignment repository. This repository contains code, data, and resources for building and evaluating graph neural network (GNN) models to predict gene-disease associations. Below is a detailed overview of the repository's structure and its components.** <br>

## Repository Structure

### 1. Data
- **`GDA_df.csv`**: Contains preprocessed data for gene-disease associations, ready to be used by the graph models.  
- **`graph_data.pkl`**: A serialized file containing the homogeneous graph data used by the models.

---

### 2. Models
The repository includes four graph neural network (GNN) models:

- **GCN_DP**  
  - **Architecture**: Two `GCNConv` layers for encoding, followed by a simple dot product for classification.  
  - **Use Case**: Basic model for link prediction.  

- **GCN_MLP**  
  - **Architecture**: Two `GCNConv` layers for encoding, followed by a Multilayer Perceptron (MLP) for decoding.  
  - **Use Case**: Improved model with more complex decoding.  

- **GraphSAGE_MLP**  
  - **Architecture**: Two `SAGEConv` layers for encoding, followed by an MLP for decoding.  
  - **Use Case**: Handles larger graphs by sampling neighbors during training.  

- **GIN_MLP**  
  - **Architecture**: Two `GINConv` layers for encoding, followed by an MLP for decoding.  
  - **Use Case**: Captures structural properties of the graph more effectively.  

**Model Weights**  
The best-performing configurations are saved in the following files:  
- `GCN_DP.pth`  
- `GCN_MLP.pth`  
- `GraphSAGE_MLP.pth`  
- `GIN_MLP.pth`

---

### 3. Code Components

#### Data Handling
- **`data_acquisition_processing.py`**  
  - Downloads and processes raw data to prepare it for GNN models.  
  - Provides the `get_data` function:  
    - **Parameters**: `disgenet_api_key`, `disease_type`.  
    - Fetches and processes associations based on disease type.  

- **`graph_preparation.py`**  
  - Prepares the homogeneous graph for training:  
    - **Node Features**: Combines gene and disease features, along with a node type indicator (1 for genes, 0 for diseases).  
    - **Edge Index**: Assigns unique IDs to nodes and creates edges based on associations. Reverse edges are added for better embeddings.

#### Model Training
- **`trainer.py`**  
  Implements a `Trainer` class for training, evaluating, and testing models:
  - **Training**  
    - Encodes node embeddings and decodes edge predictions.  
    - Computes loss for positive and negative edges using `BCEWithLogitsLoss`.  
    - Supports learning rate scheduling (`CosineAnnealingLR`) and early stopping.  
  - **Evaluation**  
    - Computes AUC, F1 score, and confusion matrix.  
    - Dynamically determines the best threshold for binary classification.  
  - **Testing**  
    - Loads the best model checkpoint to evaluate test performance.

#### Interactive Application
- **`gradio_app.py`**  
  - Provides a Gradio-based interface for user input:  
    - Accepts a gene ID and disease ID.  
    - Fetches features from the DisGeNET API.  
    - Processes input to match the training graph format.  
    - Updates the graph if the gene or disease is missing.  
    - Uses the loaded model to predict associations and outputs binary classification based on the best threshold.

---

### 4. Containerization

- **`Dockerfile`**  
  - Sets up the environment for running either JupyterLab (for development) or Gradio (for user interaction).  
  - Includes all dependencies for data processing, model training, and serving.  

- **`start.sh`**  
  - Script for running the Docker container:  
    - Option to start JupyterLab or the Gradio application.

## Installation Instructions
There are two options to set up the project:

1. **Clone and Build Locally**  
   Download all the files from the GitHub repository and build the lightweight Docker image locally.
   
2. **Pull Prebuilt Image**  
   Pull the publicly available Docker image:
   ```bash
   docker pull alexandracsibi/deeplearning-project
   ```

---

## Run the Project

After obtaining the Docker image, you can either run **JupyterLab** or the **Gradio app**.
If you built the Docker image locally, replace `alexandracsibi/deeplearning-project:latest` with your local image name.

1. **Run JupyterLab**  
   Use JupyterLab to explore the project by testing data acquisition, graph data preparation, and training or evaluating models.  
   Run the following command:
   ```bash
   docker run -p 8888:8888 -it alexandracsibi/deeplearning-project:latest
   ```  
   - After starting the container, connect to JupyterLab at [http://localhost:8888](http://localhost:8888). You will be prompted to enter a password ("kicsikutya").

2. **Run the Gradio App**  
   Use the Gradio app to interact with the pre-trained models for gene-disease link prediction.  
   Run the following command:
   ```bash
   docker run -p 7860:7860 -it alexandracsibi/deeplearning-project:latest 
   ```
   - Once the container is running, connect to the app at [http://localhost:7860](http://localhost:7860).
---

## Model Evaluations

The performance metrics for the models trained and evaluated in this project are summarized below:

| Metric      | GCN_DP   | GCN_MLP  | GraphSAGE_MLP | GIN_MLP   |
|-------------|----------|----------|---------------|-----------|
| **F1**      | 0.8835   | 0.9020   | 0.9217        | 0.9252    |
| **AUC**     | 0.9414   | 0.9640   | 0.9272        | 0.9716    |

### Confusion Matrices

#### **GCN_DP model**
|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | 2890               | 422                |
| **Actual Negative** | 357                | 2955               |

#### **GCN_MLP model**
|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | 3044               | 268                |
| **Actual Negative** | 371                | 2941               |

#### **GraphSAGE_MLP model**
|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | 2781               | 531                |
| **Actual Negative** | 27                 | 3285               |

#### **GIN_MLP model**
|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | 3089               | 223                |
| **Actual Negative** | 269                | 3043               |

---

The GIN model demonstrated the best performance in separating positive and negative edges, as evidenced by its high F1 score, AUC, and confusion matrix on the test dataset. This exceptional edge classification capability translates directly to its effectiveness in the Gradio app for individual gene-disease predictions. Its precision in distinguishing associations makes it the most reliable model for practical applications in real-world scenarios.

---

### Contribution Note

In this project:
- The data fetching process was a collaborative effort between me and my former teammate (Nagypál Márton Péter, Q88P3E).
- All subsequent steps, including data processing, graph preparation, model training, evaluation, and application development, were completed individually by me.

---


