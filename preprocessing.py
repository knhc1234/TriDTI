import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils.data import *
from models.model import *
from models.protein_llm_feature import *
from models.drug_llm_feature import *

def get_similarity_matrix(embeddings_array):
    cos_sim_matrix = cosine_similarity(embeddings_array)
    min_value = np.min(cos_sim_matrix)
    max_value = np.max(cos_sim_matrix)
    scaled_cos_sim_matrix = (cos_sim_matrix - min_value) / (max_value - min_value)
    return scaled_cos_sim_matrix

def process_split(df, save_path, drug_llm, protein_llm):
    unique_smiles = list(dict.fromkeys(df["SMILES"].tolist()))
    unique_proteins = list(dict.fromkeys(df["Target Sequence"].tolist()))

    drug_embeddings = drug_llm.get_representations(unique_smiles)
    protein_embeddings = protein_llm.get_representations(unique_proteins)

    drug_embeddings_array = np.array(list(drug_embeddings.values()))
    protein_embeddings_array = np.array(list(protein_embeddings.values()))

    # Save raw embeddings and sequences
    np.save(f"{save_path}/drug_embeddings.npy", drug_embeddings_array)
    np.save(f"{save_path}/protein_embeddings.npy", protein_embeddings_array)
    np.save(f"{save_path}/drug_smiles.npy", np.array(unique_smiles))
    np.save(f"{save_path}/protein_sequences.npy", np.array(unique_proteins))

    drug_ids = [f"{i}" for i in range(len(unique_smiles))]
    protein_ids = [f"{i}" for i in range(len(unique_proteins))]

    pd.DataFrame({'ID': drug_ids, 'SMILES': unique_smiles}).to_csv(f"{save_path}/drug_ID.csv", index=False)
    pd.DataFrame({'ID': protein_ids, 'Protein Sequence': unique_proteins}).to_csv(f"{save_path}/protein_ID.csv", index=False)

    # Compute and save similarity matrix
    drug_sim_matrix = get_similarity_matrix(drug_embeddings_array)
    pd.DataFrame(drug_sim_matrix, index=drug_ids, columns=drug_ids).to_csv(f"{save_path}/drug_similarity_matrix.csv")

    print(f"drug similarity matrices and metadata saved to {save_path}/")

def map_protein_sequences(fasta_file, filtered_link_file):
    protein_sequences = {}
    
    df_links = pd.read_csv(filtered_link_file)
    valid_proteins = set(df_links['protein1']).union(set(df_links['protein2']))
    
    with open(fasta_file, 'r') as f:
        sequence = ""
        protein_id = ""
        for line in f:
            if line.startswith('>'):
                if sequence and protein_id in valid_proteins:  
                    protein_sequences[protein_id] = sequence
                protein_id = line.strip().split()[0][1:]  
                sequence = ""
            else:
                sequence += line.strip()
        if sequence and protein_id in valid_proteins:
            protein_sequences[protein_id] = sequence
    
    return df_links, protein_sequences

def main():
    dataset_name = "DAVIS"    # DAVIS, BIOSNAP, DrugBank 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = f'./dataset/{dataset_name}_5fold'
    
    # Load splits
    train_df = pd.read_csv(f"{save_dir}/fold0/train.csv")
    valid_df = pd.read_csv(f"{save_dir}/fold0/valid.csv")
    test_df = pd.read_csv(f"{save_dir}/fold0/test.csv")
    
    data_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    
    drug_llm = DRUGLLM(device)
    protein_llm = PROTEINLLM(device)
    
    process_split(data_df, save_dir, drug_llm, protein_llm)

    del drug_llm
    del protein_llm
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Create PPI dataset
    fasta_file = "./dataset/string_database/protein_sequence.fa"
    filtered_link_file = "./dataset/string_database/filtered_protein_links.csv"
    
    df_links, protein_sequences = map_protein_sequences(fasta_file, filtered_link_file)
    
    dataset_file = f"{save_dir}/protein_ID.csv"
    output_file = f"{save_dir}/ppi_similarity.csv"
    
    df_dataset = pd.read_csv(dataset_file)
    sequence_to_custom_id = dict(zip(df_dataset['Protein Sequence'], df_dataset['ID']))
            
    mapped_data = []
            
    for _, row in tqdm(df_links.iterrows()):
        p1, p2, score = row['protein1'], row['protein2'], row['combined_score']
                
        # protein1, protein2 to Sequence
        seq1 = protein_sequences.get(p1, None)
        seq2 = protein_sequences.get(p2, None)
        
        # Sequence to Protein ID
        id1 = sequence_to_custom_id.get(seq1, None)
        id2 = sequence_to_custom_id.get(seq2, None)
        
        # If both proteins in dataset Save
        if id1 is not None and id2 is not None:
            score = score / 999 # Min-Max Scaling
            mapped_data.append((id1, id2, score))
        
    df_mapped = pd.DataFrame(mapped_data, columns=['ID1', 'ID2', 'combined_score'])
    df_mapped.to_csv(output_file, index=False)
    print(f"Saved {len(df_mapped)} edges to {output_file}")
    
if __name__ == "__main__":
    main()