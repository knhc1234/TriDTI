import esm, torch, sys, os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel, BertTokenizer
from transformers import BertConfig, BertModel, T5Tokenizer, T5EncoderModel

# t33
class PROTEINLLM:
    def __init__(self,device):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        #self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.model = self.model.to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = device
        self.max_length = 1024

    def get_representations(self, X_target):
        data = []
        for i in range(len(X_target)):
            data.append(("protein"+str(i),X_target[i]))
        
        batch_size = 1
        data = [data[i * batch_size:(i + 1) * batch_size] for i in range((len(data) + batch_size - 1) // batch_size )]
        
        # Process batches (this supports multiple sequence inputs)
        self.model.eval()  # disables dropout for deterministic results
        protein_embeddings = {}  
        
        for temp_data in tqdm(data):
            batch_labels, batch_strs, batch_tokens = self.batch_converter(temp_data)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            
            # Truncate
            if batch_tokens.shape[1] > self.max_length:
                #batch_tokens[i, self.max_length:] = self.alphabet.padding_idx
                batch_tokens = batch_tokens[:, :self.max_length].clone()
                batch_lens = torch.clamp(batch_lens, max=self.max_length)

            batch_tokens = batch_tokens.to(self.device)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33]) # , return_contacts=True
            
            token_representations = results["representations"][33].to('cpu')
            
            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, (label, tokens_len) in enumerate(zip(batch_labels, batch_lens)):
                embedding = token_representations[i, 1 : tokens_len - 1].mean(0)
                protein_embeddings[batch_strs[i]] = embedding.tolist()
                # print sequence representation shape
                
            del results, batch_tokens
            torch.cuda.empty_cache() 

        #use torch stack to convert list of tensors to tensor
        return protein_embeddings