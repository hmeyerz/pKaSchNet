from egnn_pytorch import EGNN_Network
import torch
import torch.nn as nn

import math

#EGNN
class EGNNBlock(nn.Module):
    """todo: try to take head out here
    egnn_net --> layer norm --> ffn head"""
    def __init__(self, dim, depth,hidden_dim,dropout,
                 num_positions, num_tokens,
                 num_nearest_neighbors,
                 norm_coors,
                 num_global_tokens, num_edge_tokens):
        super().__init__()
        self.egnn = EGNN_Network(
            dim=dim, depth=depth, dropout=dropout,
            num_positions=num_positions,
            num_tokens=num_tokens,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=norm_coors,
            num_edge_tokens=num_edge_tokens,
            num_global_tokens=num_global_tokens,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim*dim),
            nn.PReLU(),# LU(),
            nn.Linear(hidden_dim*dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, z, x):
        (h_list, coords) = self.egnn(z, x)
        h = h_list # [B,N,dim]
        h2 = h
        h  = self.norm1(h + h2)
        h2 = self.ffn(h)
        h  = self.norm2(h + h2)
        return [h], coords

class StackedEGNN(nn.Module):
    """TODO understand depth"""
    def __init__(self, dim, depth, hidden_dim, dropout,
                 num_positions, num_tokens,
                 num_nearest_neighbors,
                 norm_coors,
                 num_global_tokens, num_edge_tokens):
        super().__init__()
        self.blocks = nn.ModuleList([
            EGNNBlock(dim, depth, hidden_dim, dropout,
                      num_positions, num_tokens,
                      num_nearest_neighbors,
                      norm_coors,
                      num_global_tokens, num_edge_tokens)
            for _ in range(1)
        ])
    def forward(self, z, x):
        coords = x
        h_list = None
        for block in self.blocks:
            if h_list is None:
                h_list, coords = block(z, x)
            else:
                h_list, coords = block(z, coords)
        return h_list, coords

# --- RBF with learnable cutoff ---
class LearnableRBF(nn.Module):
    """TODO change cutout"""
    def __init__(self, num_basis=16, cutoff=10.0):
        super().__init__()
        #self.pairwise = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), dim=-1)
        self.cutoff = nn.Parameter(torch.tensor(cutoff))
        self.mu     = nn.Parameter(torch.linspace(0.0, 1.0, num_basis))
        self.gamma  = nn.Parameter(torch.tensor(12.0))
    
    def pairwise_distances(self, dist):
        return torch.norm(dist.unsqueeze(1) - dist.unsqueeze(0), dim=-1)

    
    def forward(self, dist):
        # dist: [B,N,N]
        dist = self.pairwise_distances(dist)
        mu = self.mu * self.cutoff     # [K]
        d  = dist.unsqueeze(-1)        # [B,N,N,1]
        return torch.exp(-self.gamma * (d - mu)**2)


#Attn. note encoding dropout of 0.03
#TODO: SPECIFy max len and dropout.
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000, dropout=0.03):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float)
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        cosp = torch.cos(pos * div)
        pe[:, 1::2] = cosp[:, : pe[:, 1::2].shape[1]]
        self.register_buffer('pe', pe.unsqueeze(1))
    def forward(self, x):
        return self.dropout(x + self.pe[: x.size(0)])

# --- Transformer‐style AttentionBlock ---
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.encoding = PositionalEncoding(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * hidden_dim),
            nn.PReLU(),
            nn.Linear(embed_dim * hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, key_padding_mask=None):
        # x: [seq_len, batch, embed_dim]
        x = self.encoding(x)
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x    = self.norm1(x + a)
        f    = self.ffn(x)
        x    = self.norm2(x + f)
        return x,_