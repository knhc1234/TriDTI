import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

import random
from tqdm import tqdm
import numpy as np

import dgl
from dgl import DGLGraph

from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.gt import GraphormerLayer
from dgllife.model import GCN, GAT
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.conv import GATv2Conv
from torch.nn.utils.parametrizations import weight_norm
from dgl.nn.pytorch.glob import GlobalAttentionPooling

class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_A, z_B, index):
        """
        z_A: (B, D) - modality A (e.g., LLM)
        z_B: (B, D) - modality B (e.g., Graph)
        index: (B,) - entity ID (e.g., drug_id), or None during inference
        """
        if index is None:
            return torch.tensor(0.0, device=z_A.device)

        z_A = F.normalize(z_A, dim=1)
        z_B = F.normalize(z_B, dim=1)

        # Positive: cross-modality
        pos_sim = torch.sum(z_A * z_B, dim=1) / self.temperature  # (B,)
        pos_sim_exp = torch.exp(pos_sim)

        index = index.view(-1, 1)
        neg_mask = (index != index.T).float()
        
        # NegativeA: intra-modality (within anchor set only)
        sim_matrix_A = torch.matmul(z_A, z_A.T) / self.temperature
        neg_denom_A = (torch.exp(sim_matrix_A) * neg_mask).sum(dim=1) + 1e-8
        loss_A = -torch.log(pos_sim_exp / (pos_sim_exp + neg_denom_A))

        # NegativeB: intra-modality (within anchor set only)
        sim_matrix_B = torch.matmul(z_B, z_B.T) / self.temperature
        neg_denom_B = (torch.exp(sim_matrix_B) * neg_mask).sum(dim=1) + 1e-8
        loss_B = -torch.log(pos_sim_exp / (pos_sim_exp + neg_denom_B))

        return 0.5 * (loss_A.mean() + loss_B.mean())

class SoftAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        weights = self.attention(features)  # (B, N, 1)
        weights = torch.softmax(weights, dim=1)  # (B, N, 1)
        fused = torch.sum(features * weights, dim=1)  # (B, D)
        return fused, weights

class GIN(torch.nn.Module):
    def __init__(self, dim_h, num_node_features):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       ))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU(),))
 
    def forward(self, g, h):
        # Node embeddings
        h1 = self.conv1(g, h)
        h2 = self.conv2(g, h1)
 
        return h2

class GIN_Model(nn.Module):
    def __init__(self, in_feats, hidden_feats=128):
        super(GIN_Model, self).__init__()
    
        self.gnn = GIN(dim_h = hidden_feats, num_node_features = in_feats)

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata['h']
        node_embeds = self.gnn(batch_graph, node_feats)
        graph_embed = node_embeds.mean(dim=0)
        return graph_embed 

class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, conv_dim):
        super(ProteinCNN, self).__init__()

        self.embedding = nn.Embedding(26, embedding_dim)
        # 1-gram, 3-gram, 5-gram)
        self.conv1_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_dim, kernel_size=1)
        self.conv1_2 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=1)
        self.conv1_3 = nn.Conv1d(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=1)
        
        self.conv3_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_dim, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv1d(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=3, padding=1)
        
        self.conv5_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_dim, kernel_size=5, padding=2)
        self.conv5_2 = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=5, padding=2)
        self.conv5_3 = nn.Conv1d(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=5, padding=2)
        
        self.gelu = nn.GELU()
        self.pool = nn.AdaptiveMaxPool1d(1)  
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, embedding_dim) -> (batch, embedding_dim, seq_len)
        
        x1 = self.gelu(self.conv1_1(x))  # (batch, 128, seq_len) -> (batch, 512, 1)
        x1 = self.gelu(self.conv1_2(x1))
        x1 = self.pool(self.conv1_3(x1)).squeeze(-1)
        
        x3 = self.gelu(self.conv3_1(x))  # (batch, 128, seq_len) -> (batch, 512, 1)
        x3 = self.gelu(self.conv3_2(x3))
        x3 = self.pool(self.conv3_3(x3)).squeeze(-1)
        
        x5 = self.gelu(self.conv5_1(x))  # (batch, 128, seq_len) -> (batch, 512, 1)
        x5 = self.gelu(self.conv5_2(x5))
        x5 = self.pool(self.conv5_3(x5)).squeeze(-1)
        
        x = x1 + x3 + x5
        return x

class GATv2_Model(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, hidden_feats=128):
        super(GATv2_Model, self).__init__()
    
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        self.gnn1 = GATv2Conv(in_feats=dim_embedding, out_feats=hidden_feats, num_heads=1)
        self.gnn2 = GATv2Conv(in_feats=hidden_feats, out_feats=hidden_feats, num_heads=1)
        self.gelu = nn.GELU()
        self.output_feats = hidden_feats

    def forward(self, batch_graph, node_index=None):
        node_feats = batch_graph.ndata['h']
        node_feats = self.init_transform(node_feats)  

        node_feats1 = self.gnn1(batch_graph, node_feats)  
        node_feats1 = self.gelu(node_feats1)  

        node_feats2 = self.gnn2(batch_graph, node_feats1)
        node_feats = node_feats2[node_index].squeeze(0)
        return node_feats

class TriDTI(nn.Module): 
    def __init__(
        self,
        hidden_dim=512,
        projection_dim=128, 
        mol_dim=768,
        prot_dim=320,
        gcn_dim=128, 
        cnn_dim=128, 
        drug_atom_dim=79,
        drug_graph_dim=769,
        prot_graph_dim=1281,
        num_heads=8,
    ):
        super().__init__()

        self.contrastive_loss = CrossModalContrastiveLoss(temperature=0.1)
        
        self.gcn1 = GIN_Model(in_feats=drug_atom_dim, hidden_feats=gcn_dim)
        self.protein1 = ProteinCNN(embedding_dim = cnn_dim, conv_dim = cnn_dim)
        self.gcn2 = GATv2_Model(in_feats=drug_graph_dim, hidden_feats=gcn_dim)
        self.gcn3 = GATv2_Model(in_feats=prot_graph_dim, hidden_feats=gcn_dim)

        self.proj_mol_llm = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.proj_mol_graph = nn.Sequential(
            nn.Linear(gcn_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.proj_mol_ddi = nn.Sequential(
            nn.Linear(gcn_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.proj_prot_llm = nn.Sequential(
            nn.Linear(prot_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.proj_prot_cnn = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.proj_prot_ppi = nn.Sequential(
            nn.Linear(gcn_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.drug_fusion_module = SoftAttentionFusion(input_dim=projection_dim, hidden_dim=projection_dim)
        self.prot_fusion_module = SoftAttentionFusion(input_dim=projection_dim, hidden_dim=projection_dim)
        
        self.cross_attention_drug = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, batch_first=True) 
        self.cross_attention_prot = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, batch_first=True) 
        
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim * 2), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        """ Xavier Initialization """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  

    def forward(self, mg_list, dg_list, pg_list, drug_embedding, prot_embedding, d_id, p_id, dg_index, pg_index, protein_sequence):
        drug_llm = self.proj_mol_llm(drug_embedding)
        drug_graph = self.proj_mol_graph(torch.stack([self.gcn1(g) for g in mg_list]))
        ddi_graph = self.proj_mol_ddi(torch.stack([self.gcn2(g, idx) for g, idx in zip(dg_list, dg_index)]))
        
        prot_llm = self.proj_prot_llm(prot_embedding)
        prot_cnn = self.proj_prot_cnn(self.protein1(protein_sequence))
        ppi_graph = self.proj_prot_ppi(torch.stack([self.gcn3(g, idx) for g, idx in zip(pg_list, pg_index)]))
        
        kd_loss_drug_mol = self.contrastive_loss(drug_llm, drug_graph, d_id)
        kd_loss_drug_ddi = self.contrastive_loss(drug_llm, ddi_graph, d_id)
        
        kd_loss_prot_cnn = self.contrastive_loss(prot_llm, prot_cnn, p_id)
        kd_loss_prot_ppi = self.contrastive_loss(prot_llm, ppi_graph, p_id)
        
        total_kd_loss = kd_loss_drug_mol + kd_loss_drug_ddi + kd_loss_prot_cnn + kd_loss_prot_ppi

        drug_features_for_fusion = torch.stack([drug_llm, drug_graph, ddi_graph], dim=1) # (B, 3, projection_dim)
        prot_features_for_fusion = torch.stack([prot_llm, prot_cnn, ppi_graph], dim=1) # (B, 3, projection_dim)

        fused_drug_feature, drug_fusion_weights = self.drug_fusion_module(drug_features_for_fusion) # (B, projection_dim)
        fused_prot_feature, prot_fusion_weights = self.prot_fusion_module(prot_features_for_fusion) # (B, projection_dim)

        drug_cross_attn, _ = self.cross_attention_drug(fused_drug_feature.unsqueeze(1), fused_prot_feature.unsqueeze(1), fused_prot_feature.unsqueeze(1))
        prot_cross_attn, _ = self.cross_attention_prot(fused_prot_feature.unsqueeze(1), fused_drug_feature.unsqueeze(1), fused_drug_feature.unsqueeze(1))
        
        drug_final_feature = (drug_cross_attn.squeeze(1) + fused_drug_feature)
        prot_final_feature = (prot_cross_attn.squeeze(1) + fused_prot_feature)
        
        x = torch.cat([drug_final_feature, prot_final_feature], dim= -1)
        cls_out = self.mlp(x).squeeze(-1)

        return cls_out, total_kd_loss #drug_fusion_weights, prot_fusion_weights