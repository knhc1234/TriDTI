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

### 2. Check the config file(`/configs/{DAVIS, BIOSNAP, DrugBank}.yaml`) to set the ATC level for prediction and configure the model hyperparameters.
The hyperparameters are defined as follows:
- `level`: Specifies the ATC code level to be predicted (1-4).
- `generate_new_negative_target`: Determines whether to generate new negative samples or use existing data (True/False).
- `epochs`: Number of training epochs.
- `early_stopping_count`: Criterion for applying early stopping.
- `batch_size`: Batch size.
- `num_layers`: Number of Transformer blocks in the CT Encoder.
- `tr_hidden`: CNN output dimension and Transformer's hidden dimension.
- `num_head`: Number of attention heads.
- `ff_weight`: Ratio for the feed-forward neural network dimension in the Transformer.
- `dropout`: Dropout rate.
- `learning_rate`: Learning rate for model optimization.
- `adj_paths`:  Specifies the paths to the three drug similarity matrices. 

### 3. run `preprocessing.py` to make preprocessed data.
```
python preprocessing.py
```

### 4. run 'main.py --config {DAVIS/BIOSNAP/DrugBank}.yaml to training model and get results.
```
python main.py --config DAVIS.yaml
python main.py --config BIOSNAP.yaml
python main.py --config DrugBank.yaml
```

### 5. The predicted results for each fold will be saved in the `/dataset/{DAVIS/BIOSNAP/DrugBank}_5fold/result` directory.
