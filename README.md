# TriDTI
The official code implementation for *TriDTI* from our paper, *"TriDTI: Tri-modal Representation Learning with Cross-Modal Alignment for Drug-Target Interaction Prediction"*. 

# Requirements

## 1. Environment Setup (Recommended)
The recommended environment uses **Python 3.9** and **CUDA 12.1** compatible libraries.

**A. Using `requirements.txt` (Recommended)**

A `requirements.txt` file has been created to simplify dependency installation.

1.  **Create a virtual environment (e.g., using `conda`):**
    ```bash
    conda create -n TriDTI python=3.9
    conda activate TriDTI
    ```
    
2.  **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: This file contains the versions specified below, including PyTorch and DGL, which should install CUDA 12.1 compatible versions automatically if a suitable GPU/environment is detected.)

**B. Alternative/Manual Setup**
If the `requirements.txt` method fails to install the correct CUDA-enabled versions (especially for PyTorch and DGL), please use the following detailed steps:
- `python`=3.9.24
- `pytorch`=2.1.0+cu121
- `dgl`=2.0.0+cu121
- `dgllife`
- `numpy`=1.26.3
- `transformers`=4.50.1
- `fair-esm`
- `rdkit`=2024.9.5

```bash
# 1. Create and activate environment
conda create -n TriDTI python=3.9
conda activate TriDTI

# 2. Install PyTorch (ensure CUDA 12.1 compatibility)
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Install DGL (ensure CUDA 12.1 compatibility)
conda install -c dglteam/label/th21_cu121 dgl

# 4. Install remaining dependencies
pip install dgllife
pip install numpy==1.26.3
pip install transformers==4.50.1
pip install fair-esm
pip install rdkit==2024.9.5
```

# Running code example

### 1. Prepare Raw Data

Unzip the following files in the `/dataset/string_database` directory:
- `filtered_protein_links.zip` → `filtered_protein_links.csv`
- `protein_link.zip` → `protein_link.txt`
- `protein_sequence.zip` → `protein_sequence.fa`

### 2. Run Preprocessing (`preprocessing.py`)

Run `preprocessing.py` to perform **initial data preparation** and **static feature generation**.

**This script generates the core feature files for the entire dataset:**

* **Sequential Features:**
    * Drug sequence embeddings (`drug_embeddings.npy`)
    * Target sequence embeddings (`protein_embeddings.npy`)
    * Corresponding SMILES codes (`drug_smiles.npy`)
    * Target protein sequences (`protein_sequences.npy`)
* **Relational Features (Global Graphs):**
    * Drug-Drug Similarity Matrix (`drug_similarity_matrix.csv`, based on sequence embeddings)
    * STRING PPI Matrix (`ppi_similarity.csv`, containing combined scores for all targets in the dataset)
    * Mapping files for unique Drug and Protein IDs.

You can select the dataset you want to preprocess on line 71. (dataset_name = "DAVIS/BIOSNAP/DrugBank")
```bash
python preprocessing.py
```

### 3. Run Training (`main.py`)
Run 'main.py --config {DATASET}.yaml to strat the training process. The model automatically performs the remaining data structuring steps.

** 3.1 Dynamic Graph Construction & Data Loading
The data loader component (```./utils/data.py```) handles the full preparation of tri-modal inputs:
*** 1. Global Relational Graph Caching:
    * The Drug-Drug Similarity Matrix and STRING PPI Matrix (from step 2) are used to construct sparse, $top\_k$ based Global Relational Graphs (Drug-Drug Graph and Target-Target Graph).
    * These global graphs are created only once per dataset and saved as binary files (```drug_graph.bin, string_graph.bin```) for subsequent automatic loading.
*** 2. Sample-Specific Input Generation:
    * For each DTI pair, the three required feature types are generated:
    ** Structural Feature: The Molecular Graph is generated on-the-fly from the SMILES string (using RDKit/DGL) and padded to ```MAX_ATOM_NODES```.
    ** Relational Features (Subgraphs): $k$-hop neighborhoods centered around the current drug/target are extracted (subsampled) from the cached Global Relational Graphs and padded to ```MAX_DRUG_NODES``` and ```MAX_PROT_NODES```.
    ** Sequential Features: Sequence embeddings and sequence encoding are fetched.

3.2. Execution
Check the configuration details in ```/configs/README.md``` before execution.

```
python main.py --config DAVIS.yaml
python main.py --config BIOSNAP.yaml
python main.py --config DrugBank.yaml
```

### 4. Results
The predicted results for each fold will be saved in the `/dataset/{DATASET}_5fold/result` directory.
