# DeepLearning_Assignment-Disgenet
**Team name:** *CSA_NMP_TEAMS* <br>
**Team members' names and Neptun codes:** <br>
- Csibi Alexandra *(GPVFEV)*
- Nagypál Márton Péter *(Q88P3E)*

**Project description** <br>
*Disease-gene interaction prediction with graph neural networks* <br>
The goal of this project is to create a graph neural network for predicting disease-gene associations. Working with DisGeNET, a comprehensive database of these associations, you'll apply deep learning to an important challenge of bioinformatics. By choosing this project, you'll gain experience in the intersection of deep learning and bioinformatics while extracting valuable insights from real-world data.

**Functions of the files in the repository** <br>
- **ssh forder / ssh_config:**  contains the ssh settings for the machine made from the docker image
- **Dockerfile:** uses an official PyTorch image with CUDA support. It sets some environment variables and creates a working directory for the user. Installs the necessary packages (git, openssh-server, mc), creates a new user, then clones the GitHub repository to the working directory. Finally, it installs the Python dependencies from the requirements.txt file.
- **docker-compose.yml:** defines a service called srserver, which is built based on the local Dockerfile. The service image is deeplearningassdis, and the container name is deeplearningassdis_con. It appends the local directory to the container's user directory and opens the necessary ports: 8899 for Jupyter, 2299 for SSH, and 7860 for Gradio. The container will reboot unless it is shut down and also seizes an NVIDIA GPU to run.
- **requirements.txt:** lists the Python dependencies for the project, including scikit-learn, seaborn, pandas, numpy, matplotlib, torch-geometric, jupyterlab and related jupyter packages, and the tensorflow and gradio libraries. These packages are provided with different version numbers, ensuring compatibility and functionality needed for the project.
- **Mélytanulás_Beadandó_Csibi_Alexandra,_Nagypál_Márton.ipynb:** contains the Python code for the header solution, tagged
- **disgenet-GDA_cancer.csv & preprocessed_GDA_df_cancer.csv:**

**Related works (papers, GitHub repositories, blog posts, etc)** <br>
- [Related GitHub repository](https://github.com/pyg-team/pytorch_geometric)
- [Related GitHub repository](https://github.com/sujitpal/pytorch-gnn-tutorial-odsc2021)
- [Related YouTube video](https://www.youtube.com/watch?v=-UjytpbqX4A&list=LL&index=1)
- [Dataset](https://www.disgenet.org/)


**How to run it (building and running the container, running your solution within the container)** <br>
0. **Add the NVIDIA-Container-toolkit to OS**
Go to the [Official NVIDIA website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for installing and configure! And install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network).

1. **Build the Docker container:**
    ```bash
    docker-compose build
    ```

2. **Run the Docker container:**
    ```bash
    docker-compose up
    ```

3. **Access JupyterLab:**
    Open your web browser and navigate to `http://localhost:8899`. You should see the JupyterLab interface.

4. **Access the container via SSH:**
    ```bash
    ssh -p 2299 user@localhost
    ```

5. **Run the solution within the container:**
    Open the Jupyter notebook `Mélytanulás_Beadandó_Csibi_Alexandra,_Nagypál_Márton.ipynb` in JupyterLab and execute the cells to run the solution.

**How to run the pipeline?**<br>

To run the pipeline, start by setting up the environment within the Docker container, ensuring that all dependencies, including PyTorch, PyTorch Geometric, CUDA (for GPU support), and other required libraries, are installed. This environment setup is automatically managed when you build the container using `docker-compose`. With the environment ready, perform data acquisition by running the script to retrieve disease-gene associations from the DisGeNET API. In JupyterLab, you can use the following code to acquire data:

```python
from data_acquisition_processing import get_data

df = get_data("your_api_key", "disease_type") # Replace "your_api_key" and "disease_type" with actual values (e.g., "cancer").
df.to_csv("/data/raw/GDA_df_processed.csv", index=False)
```
Alternatively, you can download the processed data directly from GitHub:

```python
url = "https://raw.githubusercontent.com/NagypalMarton/DeepLearning_Assignment-Disgenet/main/GDA_df_processed.csv"
response = requests.get(url)
with open("/data/raw/GDA_df_processed.csv", "wb") as file:
    file.write(response.content)

df = pd.read_csv("GDA_df_processed.csv")
```

To prepare the data in graph format for the model, initialize the dataset with the `GDADataset` class. This will structure the disease-gene data for graph-based analysis:

```python
from data_module import GDADataset
dataset = GDADataset(data_dir='./data/')
```

**How to train the models?**<br>

To train the model, start by setting up the `GDADataModule`, which organizes the data into training, validation, and test sets. Initialize the data module with the following code:

```python
from data_module import GDADataModule
datamodule = GDADataModule(data_dir='./data/', batch_size=32)
datamodule.setup()
```

Next, initialize the graph neural network model (`GCNLinkPredictor`) with the specified input dimensions, hidden layer size, and learning rate:

```python
from model import GCNLinkPredictor
model = GCNLinkPredictor(input_dim=datamodule.train_data.x.shape[1], hidden_dim=64, lr=1e-2)
```

After initializing the model, configure the PyTorch Lightning Trainer with settings such as GPU support, the number of epochs, and a callback for saving the best model checkpoints. Start training with the following code:

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint()
trainer = Trainer(
    max_epochs=20,
    log_every_n_steps=1,
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback]
)
trainer.fit(model, datamodule)
```

**How to evaluate the models?** <br>

After training, evaluate the model’s performance on the test set by running:

```python
trainer.test(model, datamodule)
```

During evaluation, key performance metrics such as **Binary Cross-Entropy Loss** and **AUROC (Area Under Receiver Operating Characteristic Curve)** are calculated. These metrics provide insight into the model's prediction accuracy and its ability to distinguish between true disease-gene associations and false ones, giving a clear measure of its classification effectiveness.
