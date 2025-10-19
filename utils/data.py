import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops, Descriptors

import torch.nn.functional as F
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
#from torch_geometric.data import Data
from functools import partial
#from dgl import graph, add_self_loop, save_graphs, load_graphs, khop_in_subgraph
import dgl
from tqdm import tqdm
import os
import yaml
import argparse
import gc

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load configuration from YAML file.")
    parser.add_argument("--config", type=str, default="BindingDB.yaml", help="Path to the config YAML file")
    args = parser.parse_args()
    return args

def load_config(config_path="config.yaml"):
    config_path = os.path.join("./configs", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_dgl_graph(save_path, similarity_matrix, ids, id_to_embedding, top_k = 30):
    src_list, dst_list, edge_weights = [], [], []
    for i, row in enumerate(similarity_matrix.values):
        top_indices = np.argsort(row)[-top_k:]
        src_list.extend([similarity_matrix.index[i]] * top_k)   # start node
        dst_list.extend(similarity_matrix.index[top_indices])   # end node
        edge_weights.extend(row[top_indices])

    similarity_graph = dgl.graph((src_list, dst_list), num_nodes=len(similarity_matrix))
    similarity_graph.edata['e'] = torch.tensor(edge_weights, dtype=torch.float32)
    similarity_graph = dgl.add_self_loop(similarity_graph)

    node_features = np.array([id_to_embedding[id] for id in ids])
    node_features = torch.tensor(node_features, dtype=torch.float)
        
    similarity_graph.ndata['h'] = node_features

    dgl.save_graphs(save_path, [similarity_graph])
    print(f"Saved similarity graph to {save_path}")
    return similarity_graph

def save_dgl_graph_from_links_final(save_path, link_file, prot_ids, id_to_embedding, top_k=30):
    df_links = pd.read_csv(link_file)

    similarity_matrix = pd.DataFrame(
        0.0,
        index=prot_ids,
        columns=prot_ids,
        dtype=np.float32
    )
    for _, row in df_links.iterrows():
        i, j, score = int(row['ID1']), int(row['ID2']), np.float32(row['combined_score'])
        similarity_matrix.at[i, j] = score
        similarity_matrix.at[j, i] = score  

    src_list, dst_list, edge_weights = [], [], []
    for i, row in enumerate(similarity_matrix.values):
        top_indices = np.argsort(row)[-top_k:]
        src_list.extend([similarity_matrix.index[i]] * top_k)
        dst_list.extend(similarity_matrix.index[top_indices])
        edge_weights.extend(row[top_indices])

    similarity_graph = dgl.graph((src_list, dst_list), num_nodes=len(similarity_matrix))
    similarity_graph.edata['e'] = torch.tensor(edge_weights, dtype=torch.float32)
    similarity_graph = dgl.add_self_loop(similarity_graph)

    node_features = np.array([id_to_embedding[id] for id in prot_ids])
    similarity_graph.ndata['h'] = torch.tensor(node_features, dtype=torch.float)

    dgl.save_graphs(save_path, [similarity_graph])
    print(f"Saved top-{top_k} sparsified similarity graph to {save_path}")
    return similarity_graph

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

def label_sequence(line, pro_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = pro_ch_ind[ch]
    return X

# mol atom feature for mol graph (https://github.com/thinng/GraphDTA/blob/master/create_data.py)
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = np.array([atom_features(atom) / sum(atom_features(atom)) for atom in mol.GetAtoms()], dtype=np.float32)
    
    edges = []
    bond_features = []
    for bond in mol.GetBonds():
        bond_feat = []
        bond_feat.append(bond.GetBondTypeAsDouble())
        bond_feat.append(bond.GetIsAromatic())
        bond_features.append(bond_feat)
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])  
    
    features = torch.from_numpy(features)
    bond_features = torch.tensor(bond_features, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=c_size)
    g.ndata['h'] = features
    g.edata['e'] = bond_features
    return g

def extract_subgraph(protein_graph, protein_id, num_hops=2):
    sg, inverse_indices = dgl.khop_in_subgraph(protein_graph, protein_id, num_hops)
    return sg, inverse_indices[0].item() # subgraph, subgraph's protein_id index

def load_dataset_5fold(data_path, fold, top_k_d, top_k_p):
    train_df = pd.read_csv(f"{data_path}/fold{fold}/train.csv")
    valid_df = pd.read_csv(f"{data_path}/fold{fold}/valid.csv")
    test_df = pd.read_csv(f"{data_path}/fold{fold}/test.csv")

    drug_smiles = np.load(f"{data_path}/drug_smiles.npy")  
    drug_embeddings = np.load(f"{data_path}/drug_embeddings.npy")   
    protein_sequences = np.load(f"{data_path}/protein_sequences.npy")  
    protein_embeddings = np.load(f"{data_path}/protein_embeddings.npy") 

    # Mapping Dictionary
    drug_smiles_to_embedding = dict(zip(drug_smiles, drug_embeddings))
    protein_sequences_to_embedding = dict(zip(protein_sequences, protein_embeddings))

    train_df['drug_embedding'] = train_df['SMILES'].map(drug_smiles_to_embedding)
    valid_df['drug_embedding'] = valid_df['SMILES'].map(drug_smiles_to_embedding)
    test_df['drug_embedding'] = test_df['SMILES'].map(drug_smiles_to_embedding)

    train_df['prot_embedding'] = train_df['Target Sequence'].map(protein_sequences_to_embedding)
    valid_df['prot_embedding'] = valid_df['Target Sequence'].map(protein_sequences_to_embedding)
    test_df['prot_embedding'] = test_df['Target Sequence'].map(protein_sequences_to_embedding)

    drug_id_mapping_df = pd.read_csv(f"{data_path}/drug_ID.csv")
    drug_id_mapping = dict(zip(drug_id_mapping_df['SMILES'], drug_id_mapping_df['ID']))

    drug_id_to_embedding = {drug_id: drug_smiles_to_embedding[seq] 
                           for seq, drug_id in drug_id_mapping.items() if seq in drug_smiles_to_embedding}

    train_df['drug_id'] = train_df['SMILES'].map(drug_id_mapping)
    valid_df['drug_id'] = valid_df['SMILES'].map(drug_id_mapping)
    test_df['drug_id'] = test_df['SMILES'].map(drug_id_mapping)
    
    drug_ids = list(drug_id_mapping.values())
    
    drug_graph_cache_path = f"{data_path}/drug_graph.bin"

    if os.path.exists(drug_graph_cache_path):
        print(f"Loading cached drug similarity graph from {drug_graph_cache_path}")
        drug_similarity_graph = dgl.load_graphs(drug_graph_cache_path)[0][0]
    else:
        print("Creating drug similarity graph from similarity matrix...")
        similarity_matrix = pd.read_csv(f"{data_path}/drug_similarity_matrix.csv", index_col=0)

        similarity_matrix.index = similarity_matrix.index.astype(int)
        similarity_matrix.columns = similarity_matrix.columns.astype(int)

        drug_similarity_graph = save_dgl_graph(drug_graph_cache_path, similarity_matrix, drug_ids, drug_id_to_embedding, top_k = top_k_d)

    prot_id_mapping_df = pd.read_csv(f"{data_path}/protein_ID.csv")
    prot_id_mapping = dict(zip(prot_id_mapping_df['Protein Sequence'], prot_id_mapping_df['ID']))

    prot_id_to_embedding = {prot_id: protein_sequences_to_embedding[seq] 
                           for seq, prot_id in prot_id_mapping.items() if seq in protein_sequences_to_embedding}

    train_df['prot_id'] = train_df['Target Sequence'].map(prot_id_mapping)
    valid_df['prot_id'] = valid_df['Target Sequence'].map(prot_id_mapping)
    test_df['prot_id'] = test_df['Target Sequence'].map(prot_id_mapping)

    prot_ids = list(prot_id_mapping.values())
    
    prot_graph_cache_path = f"{data_path}/string_graph.bin"
    prot_link_path = f"{data_path}/ppi_similarity.csv"

    if os.path.exists(prot_graph_cache_path):
        print(f"Loading cached prot similarity graph from {prot_graph_cache_path}")
        prot_similarity_graph = dgl.load_graphs(prot_graph_cache_path)[0][0]
    else:
        print("Creating drug similarity graph from STRING dataset...")
        prot_similarity_graph = save_dgl_graph_from_links_final(prot_graph_cache_path, prot_link_path, prot_ids, prot_id_to_embedding, top_k = top_k_p)
        
    del drug_smiles, drug_embeddings, drug_id_mapping_df, protein_sequences, protein_embeddings, prot_id_mapping_df, 
    del drug_smiles_to_embedding, drug_id_mapping, drug_id_to_embedding, protein_sequences_to_embedding, prot_id_mapping, prot_id_to_embedding
    gc.collect()
    torch.cuda.empty_cache()
    
    return train_df, valid_df, test_df, drug_similarity_graph, prot_similarity_graph

def get_dataloaders(data_path, fold, top_k_d, top_k_t, max_atom_nodes, max_drug_nodes, max_prot_nodes, batch_size, num_workers):
    train_df, valid_df, test_df, drug_similarity_graph, prot_similarity_graph = load_dataset_5fold(data_path, fold, top_k_d, top_k_t)
    
    train_dataset = CustomDataset_5fold(train_df, drug_similarity_graph, prot_similarity_graph, max_atom_nodes, max_drug_nodes, max_prot_nodes)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    
    val_dataset = CustomDataset_5fold(valid_df, drug_similarity_graph, prot_similarity_graph, max_atom_nodes, max_drug_nodes, max_prot_nodes)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    test_dataset = CustomDataset_5fold(test_df, drug_similarity_graph, prot_similarity_graph, max_atom_nodes, max_drug_nodes, max_prot_nodes)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    del train_df, train_dataset
    del valid_df, val_dataset
    del test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    return train_dataloader, val_dataloader, test_dataloader, test_df, drug_similarity_graph, prot_similarity_graph

def collate_fn(batch):
    mocular_graphs, drug_features, drug_graphs, prot_sequences, prot_features, prot_graphs, drug_id, prot_id, dg_index, pg_index, labels = zip(*batch)
    
    batched_graph1 = list(mocular_graphs)
    batched_graph2 = list(drug_graphs)
    batched_graph3 = list(prot_graphs)
    
    d_id = torch.as_tensor(drug_id, dtype=torch.long)
    p_id = torch.as_tensor(prot_id, dtype=torch.long)
    
    dg_index = torch.as_tensor(dg_index, dtype=torch.long)
    pg_index = torch.as_tensor(pg_index, dtype=torch.long)
    
    drug_features = torch.stack([torch.tensor(drug_feature) for drug_feature in drug_features], dim=0)
    prot_features = torch.stack([torch.tensor(prot_feature) for prot_feature in prot_features], dim=0)
    prot_seq_features = torch.stack([torch.from_numpy(label_sequence(prot_sequence, CHARPROTSET, 1000)).long() for prot_sequence in prot_sequences], dim = 0)
    
    labels = torch.stack(labels, dim=0)
    
    return batched_graph1, batched_graph2, batched_graph3, d_id, p_id, dg_index, pg_index, drug_features, prot_features, prot_seq_features, labels

class CustomDataset_5fold(Dataset):
    def __init__(self, df, drug_graph, prot_graph, max_atom_nodes, max_drug_nodes, max_prot_nodes):   # similarity threshold 0.8: max_prot_nodes = 1254, top_k 30: 655 
        self.smiles = df["SMILES"].tolist()
        self.protein_sequences = df["Target Sequence"].tolist()
        self.drug_id = df["drug_id"].values
        self.prot_id = df["prot_id"].values
        self.drug_features = df["drug_embedding"].tolist()
        self.prot_features = df["prot_embedding"].tolist()
        self.labels = df["Label"].values  
        
        self.drug_graph = drug_graph
        self.prot_graph = prot_graph
        
        self.max_atom_nodes = max_atom_nodes
        self.max_drug_nodes = max_drug_nodes
        self.max_prot_nodes = max_prot_nodes

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        molecule = self.smiles[idx]
        protein_sequence = self.protein_sequences[idx]
        drug_id = self.drug_id[idx]
        prot_id = self.prot_id[idx]
        drug_feature = self.drug_features[idx]
        prot_feature = self.prot_features[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Generate graph
        # Molecular graph
        mol_g = smile_to_graph(molecule)
        node_feats = mol_g.ndata['h']
        num_nodes = node_feats.shape[0]
        num_virtual = self.max_atom_nodes - num_nodes

        node_feats = torch.cat((node_feats, torch.zeros(num_nodes, 1)), dim=1)
        mol_g.ndata['h'] = node_feats
        virtual_feats = torch.cat((torch.zeros(num_virtual, 78), torch.ones(num_virtual, 1)), dim=1)
        mol_g.add_nodes(num_virtual, {'h': virtual_feats})
        mol_g = mol_g.add_self_loop()

        # Drug graph
        drug_g, dg_index = extract_subgraph(self.drug_graph, drug_id)
        node_feats = drug_g.ndata['h']
        num_nodes = node_feats.shape[0]
        num_virtual = self.max_drug_nodes - num_nodes

        node_feats = torch.cat((node_feats, torch.zeros(num_nodes, 1)), dim=1)
        drug_g.ndata['h'] = node_feats
        virtual_feats = torch.cat((torch.zeros(num_virtual, 768), torch.ones(num_virtual, 1)), dim=1)
        drug_g.add_nodes(num_virtual, {'h': virtual_feats})
        drug_g = drug_g.add_self_loop()
        
        # Prot graph
        prot_g, pg_index = extract_subgraph(self.prot_graph, prot_id)
        node_feats = prot_g.ndata['h']
        num_nodes = node_feats.shape[0]
        num_virtual = self.max_prot_nodes - num_nodes

        node_feats = torch.cat((node_feats, torch.zeros(num_nodes, 1)), dim=1)
        prot_g.ndata['h'] = node_feats
        virtual_feats = torch.cat((torch.zeros(num_virtual, 1280), torch.ones(num_virtual, 1)), dim=1)
        prot_g.add_nodes(num_virtual, {'h': virtual_feats})
        prot_g = prot_g.add_self_loop()
        
        return mol_g, drug_feature, drug_g, protein_sequence, prot_feature, prot_g, drug_id, prot_id, dg_index, pg_index, label