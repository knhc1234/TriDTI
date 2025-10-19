# TriDTI
The official code implementation for *TriDTI* from our paper, *"TriDTI: Tri-modal Representation Learning with Cross-Modal Alignment for Drug-Target Interaction Prediction"*. 

# Requirements

- `python`=3.9
- `pytorch`=2.1.0+cu121
- `dgl`=2.0.0+cu121
- `dgllife`
- `numpy`=1.26.3
- `transformers`=4.50.1
- `fair-esm`
- `rdkit`=2024.9.5
  
# Virtual Environment Setup (using anaconda)

```
conda create -n TriDTI python=3.9
conda activate TriDTI
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c dglteam/label/cu121 dgl
pip install dgllife
pip install numpy==1.26.3
pip install transformers==4.50.1
pip install fair-esm
pip install rdkit==2024.9.5
```

# Running code example

### 1. Unzip the following files in the `/dataset/string_database` directory:
- `filtered_protein_links.zip` → `filtered_protein_links.csv`
- `protein_link.zip` → `protein_link.txt`
- `protein_sequence.zip` → `protein_sequence.fa`

### 2. Check the config file(`/configs/{DATASET}.yaml`) to set the hyperparameters for the model specific to each dataset.
The DATASET variable should be replaced by the name of the dataset being used, such as DAVIS, BIOSNAP, or DrugBank.
The hyperparameters are defined as follows:
- `MAX_ATOM_NODES`: Maximum number of atom nodes for padding drug graph inputs (Structural Feature).
- `MAX_DRUG_NODES`: Maximum number of drug nodes for padding the drug relationship matrix (Relational Feature)
- `MAX_PROT_NODES`: Maximum number of protein nodes for padding the protein relationship matrix (Relational Feature)
- `top_k_d`: The number of top neighbors *k* used in the Drug GATv2 layer to define relational links.
- `top_k_t`: The number of top neighbors *k* used in the Target(Protein) GATv2 layer to define relational links.
- `hidden_dim`: The hidden dimension used for the hidden layer within the projection layer and the intermediate layer of the MLP that performs the final prediction. 
- `mol_dim`: The output embedding dimension of the ChemBERTa model for drug sequences (Sequential Feature).
- `prot_dim`: The output embedding dimension of the ESM2 model for protein sequences (Sequential Feature).
- `atom_dim`: The input feature dimension for molecular GIN (Graph Isomorphism Network) layers.
- `graph_dim`: The hidden/output dimension of the GIN/Drug GATv2/Target GATv2 layers.
- `conv_dim`: The output dimension of the Multi-scale CNN.
- `proj_dim`: The output dimension of the projection layer used in the Modality Alignment (Contrastive Learning) step.
- `LR`: The learning rate for the model optimizer (AdamW).
- `EPOCHS`: The maximum number of training epochs.
- `BATCH_SIZE`: The batch size used for training the model.
- `pos_weight`: The weight applied to the positive class in the Binary Cross-Entropy Loss (used for handling imbalanced datasets).

### 3. run `preprocessing.py` to make preprocessed data.
You can select the dataset you want to preprocess on line 71. (dataset_name = "DAVIS/BIOSNAP/DrugBank")
```
python preprocessing.py
```

### 4. run 'main.py --config {DATASET}.yaml to training model and get results.
```
python main.py --config DAVIS.yaml
python main.py --config BIOSNAP.yaml
python main.py --config DrugBank.yaml
```

### 5. The predicted results for each fold will be saved in the `/dataset/{DATASET}_5fold/result` directory.
