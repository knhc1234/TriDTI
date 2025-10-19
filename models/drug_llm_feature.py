from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, BertTokenizer, RobertaTokenizer
import torch
import numpy as np
from tqdm import tqdm
import sys

class DRUGLLM:
    def __init__(self, device):
        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        
        self.model = self.model.to(device)
        self.device = device
        self.max_length = 510
        #self.max_length = 512

    def get_representations(self, X_drug):
        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False

        for layer in self.model.roberta.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False

        batch_size = 1
        data = [X_drug[i * batch_size:(i + 1) * batch_size] for i in range((len(X_drug) + batch_size - 1) // batch_size )]

        self.model.eval()

        # drug_representations = []
        drug_embeddings = {}
        for temp_data in tqdm(data):
            inputs = self.tokenizer(temp_data, padding=True, max_length = self.max_length, truncation=True, return_tensors="pt").to(self.device)
            batch_lens = (inputs['attention_mask'] != 0).sum(1)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            token_representations = outputs.hidden_states[-1].to('cpu')

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, (smiles, tokens_len) in enumerate(zip(temp_data, batch_lens)):
                embedding = token_representations[i, 1 : tokens_len - 1].mean(0)
                drug_embeddings[smiles] = embedding.tolist()
                            
            del token_representations, inputs
            torch.cuda.empty_cache()

        return drug_embeddings