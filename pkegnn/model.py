import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from egnn_pytorch import EGNN
from architecture import (StackedEGNN,
                          LearnableRBF,
                          AttentionBlock,
                          TunableBlock)

class Model(nn.Module):

    def __init__(self,c):
        super().__init__(); self.c=c
        inner_dim = c.dim + c.basis

        self.egnn = StackedEGNN(c.dim,c.depth,c.hidden_dim,c.dropout,
                                c.hood_k,98,c.num_neighbors,c.norm_coors).to(c.device)

        self.rbf  = TunableBlock(LearnableRBF(c.basis,20.).to(c.device), c.use_rbf)
        self.attn = TunableBlock(AttentionBlock(inner_dim,inner_dim,c.hidden_dim).to(c.device), c.use_attn)

        if c.n_aggregator=='linear':
            self.agg = nn.Linear(c.hood_k,1).to(c.device)
        elif c.n_aggregator=='nconv':
            self.agg = nn.Conv1d(c.hood_k,1,kernel_size=1,padding=0).to(c.device)
        elif c.n_aggregator=='pool':
            self.agg = None
        else: raise ValueError("aggregator must be 'linear' | 'nconv' | 'pool'")

        if c.f_aggregator=="rconv":
            self.ch_agg = nn.Conv1d(inner_dim,1,1).to(c.device)
        else:
            self.ch_agg = nn.Linear(inner_dim,1).to(c.device)
        #todo: put pooling option, too, it is easy.

        #optional
        self.boost = nn.Linear(inner_dim,inner_dim).to(c.device) if c.use_boost else nn.Identity()
        self.prot  = EGNN(dim=inner_dim, update_coors=True,norm_coors=True, norm_feats=True, 
                          valid_radius=c.pegnn_radius).to(c.device) \
                     if c.use_prot else nn.Identity()


    def forward(self,z,x):
        h,coord=self.egnn(z,x); h=h[0]                # (R,N,dim)
        cent=coord.mean(1,keepdim=True)               # (R,1,3)

        # --- build token ----------------------------------------------------------------
        r = self.rbf(cent,coord).transpose(1,2) if self.c.use_rbf else \
            h.new_zeros(h.size(0),self.c.basis,self.c.hood_k)
        tok = torch.cat((r,h.transpose(1,2)),dim=1)       # (R,C,N)

        att = self.attn(tok.permute(2,0,1))
        tok = att[0] if isinstance(att,(tuple,list)) else att
        tok = tok.permute(1,0,2)                      # (R,N,C)

        # --- aggregation ----------------------------------------------------------------
        #aggregate across nbrs
        if self.c.n_aggregator=='linear':
            preds = self.agg(tok.permute(2,0,1))
            preds = self.boost(preds.T)
            if self.c.use_prot:
                preds = self.prot(preds,
                                cent.permute(2,0,1).T)[0]
        elif self.c.n_aggregator=='nconv': #TODO CPU heavy. keep this option layout like dev_run, with lack of creativity
            preds = self.agg(tok).squeeze(-2).transpose(-2,1) 
            preds = self.boost(preds.T).transpose(-2,0).unsqueeze(0) 
            if self.c.use_prot:
                preds = self.prot(preds,
                                cent.permute(1,0,2))[0]
        else:   # pool
            preds = tok.max(1).values.unsqueeze(0)
            if self.c.use_prot:
                preds = self.prot(preds,
                                cent.permute(1,0,2))[0]
        
        #agg across channels
        print(preds.shape,preds[0].unsqueeze(2).shape)
        if self.c.f_aggregator == "rconv": #TODO CPU
            preds = self.ch_agg(preds[0].unsqueeze(2)).squeeze(0).T 
        else:
            preds = self.ch_agg(preds).squeeze(0).T 

        return preds
    
class HoodDS(Dataset):
    """Generate hoods around n_neighbors, a hyperparameter. This choice determines how many n_nearest neighbors
    the egnn encoder sees.
    
    This function takes as input the pdb.npz numpy files and returns an """
    def __init__(self, paths, cfg, keep_ids=False):
        self.data=[]; self.ids=[];self.keep_ids = keep_ids; 
        nbr=NearestNeighbors(n_neighbors=cfg.hood_k,algorithm='brute')
        for p in paths:
            try:
                d=np.load(p,allow_pickle=True)
                if len(d['sites'])==0: continue
                nbr.fit(d['pos']); idx=nbr.kneighbors(d['sites'],return_distance=False)
                self.data.append((torch.from_numpy(d['z'][idx]),
                                  torch.from_numpy(d['pos'][idx]),
                                  torch.from_numpy(d['pks'])))
                self.ids.append(os.path.splitext(os.path.basename(p))[0])
            except Exception as e: print("skip",p,e)
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        z,p,y=self.data[i]; 
        return z,p,y,self.ids[i] if self.keep_ids else z,p,y

    def pad(batch,k,device,keep_ids=False):
        ids=[b[3] for b in batch] if keep_ids else None
        B=len(batch); S=max(b[0].shape[0] for b in batch)
        zt=torch.zeros(B,S,k,dtype=torch.int32,device=device)
        pt=torch.zeros(B,S,k,3,dtype=torch.float32,device=device)
        yt=torch.full((B,S),float('nan'),device=device); mt=torch.zeros(B,S,dtype=torch.bool,device=device)
        for b,data in enumerate(batch):
            z,p,y=data[0],data[1],data[2]
            s=z.shape[0]; zt[b,:s]=z; pt[b,:s]=p; yt[b,:s]=y; mt[b,:s]=True
        return (zt,pt,yt,mt,ids) if keep_ids else (zt,pt,yt,mt)

    def split(paths,cfg):
        """deterministic and random #TODO tunable determinism"""
        if cfg.num_paths: paths=paths[:cfg.num_paths]
        rng=np.random.RandomState(cfg.split_seed)
        idx=rng.permutation(len(paths)); cut=int(len(paths)*cfg.split_ratio)
        return [paths[i] for i in idx[:cut]], [paths[i] for i in idx[cut:]]