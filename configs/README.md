# Configuration File Hyperparameters (`/configs/{DATASET}.yaml`)

This document provides detailed explanations for the hyperparameters defined in the configuration files (e.g., `DAVIS.yaml`, `BIOSNAP.yaml`, `DrugBank.yaml`).

## Model Hyperparameters

| Parameter | Description | Related Feature/Component |
| :--- | :--- | :--- |
| `MAX_ATOM_NODES` | Maximum number of atom nodes used for padding drug graph inputs. | Structural Feature (Drug Graph) |
| `MAX_DRUG_NODES` | Maximum number of drug nodes for padding the drug relationship matrix. | Relational Feature (Drug GATv2) |
| `MAX_PROT_NODES` | Maximum number of protein nodes for padding the protein relationship matrix. | Relational Feature (Target GATv2) |
| `top_k_d` | The number of top neighbors *k* used in the Drug GATv2 layer to define relational links. | Relational Feature (Drug GATv2) |
| `top_k_t` | The number of top neighbors *k* used in the Target(Protein) GATv2 layer to define relational links. | Relational Feature (Target GATv2) |
| `hidden_dim` | The hidden dimension used in the projection layer and the intermediate layer of the final MLP predictor. | General Model Dimension |
| `mol_dim` | The output embedding dimension of the ChemBERTa model for drug sequences. | Sequential Feature (Drug) |
| `prot_dim` | The output embedding dimension of the ESM2 model for protein sequences. | Sequential Feature (Target) |
| `atom_dim` | The input feature dimension for molecular GIN (Graph Isomorphism Network) layers. | Structural Feature (GIN) |
| `graph_dim` | The hidden/output dimension of the GIN, Drug GATv2, and Target GATv2 layers. | Graph Module Dimension (GIN, GATv2) |
| `conv_dim` | The output dimension of the Multi-scale CNN. | Sequential Feature (CNN) |
| `proj_dim` | The output dimension of the projection layer used in the Modality Alignment (Contrastive Learning) step. | Modality Alignment |

## Training Hyperparameters

| Parameter | Description |
| :--- | :--- |
| `LR` | The learning rate for the model optimizer (AdamW). |
| `EPOCHS` | The maximum number of training epochs. |
| `BATCH_SIZE` | The batch size used for training the model. |
| `pos_weight` | The weight applied to the positive class in the Binary Cross-Entropy Loss (used for handling imbalanced datasets). |
