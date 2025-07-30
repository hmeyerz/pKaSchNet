

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py – quick ablation runner (works on Python 3.6)

Examples
--------
# disable attention
$ python run.py --use_attn False

# shorter run, different LR
$ python run.py --epochs 15 --lr 1e-4
"""
# ---------------------------------------------------------------------
# 0) original imports and helpers  (UNCHANGED – trimmed here for brevity)
# ---------------------------------------------------------------------
import torch, math, itertools, random, os, glob, time, datetime
from collections import defaultdict
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.neighbors import NearestNeighbors
from egnn_pytorch import EGNN
from architecture import (StackedEGNN,
                          LearnableRBF,
                          AttentionBlock,
                          TunableBlock)
from invar import run_invariance_suite
# ================================================================
# 3) model
# ================================================================

class Model(nn.Module):

    def __init__(self,c):
        super().__init__(); self.c=c
        inner_dim = c.dim + c.basis

        self.egnn = StackedEGNN(c.dim,c.depth,c.hidden_dim,c.dropout,
                                c.hood_k,98,c.num_neighbors,c.norm_coors).to(c.device)

        self.rbf  = TunableBlock(LearnableRBF(c.basis,20.).to(c.device), c.use_rbf)
        self.attn = TunableBlock(AttentionBlock(inner_dim,inner_dim,c.hidden_dim).to(c.device), c.use_attn)

        if c.aggregator=='linear':
            #self.agg = nn.Linear(inner_dim,1).to(c.device)
            self.agg = nn.Linear(c.hood_k,1).to(c.device)
        elif c.aggregator=='nconv':
            self.agg = nn.Conv1d(c.hood_k,1,kernel_size=1,padding=0).to(c.device)
        elif c.aggregator=='pool':
            self.agg = None
        else: raise ValueError("aggregator must be 'linear' | 'nconv' | 'pool'")
        if c.use_conv=="True":
            self.ch_agg = nn.Conv1d(inner_dim,1,1).to(c.device)
        else:
            self.ch_agg = nn.Linear(inner_dim,1).to(c.device)

        #self.lin=nn.Linear(inner_dim if c.aggregator=='nconv' else 1,1).to(c.device)
        #self.nlin=nn.Linear(c.hood_k,1).to(c.device)
        self.boost = nn.Linear(inner_dim if c.aggregator=='nconv' else inner_dim,inner_dim if c.aggregator=='nconv' else inner_dim).to(c.device) if c.use_boost else nn.Identity()
        #self.boost = nn.Linear(inner_dim if c.aggregator=='nconv' else 1,inner_dim if c.aggregator=='nconv' else 1).to(c.device) if c.use_boost else nn.Identity()
        self.prot  = EGNN(dim=inner_dim if c.aggregator=='nconv' else inner_dim,update_coors=True,norm_coors=True, norm_feats=True, 
                          valid_radius=c.pegnn_radius).to(c.device) \
                     if c.use_prot else nn.Identity()
        self.rconv  = nn.Conv1d(inner_dim,1,1,padding=0).to(c.device) \
                     if c.use_conv else nn.Identity()

    def forward(self,z,x):
        #print(z.shape,x.shape)
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
        if self.c.aggregator=='linear':
            #preds=self.nlin(tok.permute(2,0,1).unsqueeze(0))
            #preds=self.agg(tok.max(dim=1).values) #across nodes
            
            #preds=self.boost(preds).squeeze(-1) #!
            #print(tok.shape) #([111, 100, 12])
            #print(tok.permute(2,0,1).shape) #torch.Size([12, 111, 100])
            preds = self.agg(tok.permute(2,0,1))#.max(1).values                # (R,1)
            #print(preds.T.shape)
            
            #print(preds.permute(2,0,1).shape) torch.Size([1, 12, 111])
            #preds=self.agg(preds.permute(2,1,0))
            #print(preds.shape,cent.shape) #torch.Size([1, 111, 1])
            preds = self.boost(preds.T)
            
            #print(preds.shape,cent.permute(2,0,1).T.shape)
            if self.c.use_prot:
                #torch.Size([1, 188, 1]) torch.Size([1, 188, 3])
                preds = self.prot(preds,
                                cent.permute(2,0,1).T)[0]
                #preds=self.lin(preds) 
                #print(z.shape,preds.shape)
        elif self.c.aggregator=='nconv':
            #print(tok.shape) #[126, 100, 12]) 126 nodes, batch_size=12 #this is mixing along channels (knnnodes), but length (feats) indep (batch, channels, nodes)
            preds = self.agg(tok) 
            preds=preds.squeeze(-2).transpose(-2,1) # #[1,n,feats_dim]
            #print(preds.squeeze(-1).shape)
            preds = self.boost(preds.T).transpose(-2,0).unsqueeze(0) #([1, 126, 12])
            #print(preds.shape)
            #print(preds.transpose(-2,0).shape,cent.transpose(-2,0).shape)
            if self.c.use_prot:
                preds = self.prot(preds,
                                cent.permute(1,0,2))[0]
            #preds=self.lin(preds) 
            #print(z.shape,preds.shape)
            #preds=self.lin(preds).squeeze(-1) 
                             # (R,1)
        else:   # pool
            #print(tok.shape)
            preds = tok.max(1).values#.mean(1,keepdim=True)      # (R,1)
            #print(preds.shape)
            if self.c.use_prot:
                preds = self.prot(preds.unsqueeze(0),
                                cent.permute(1,0,2))[0]
                #preds=self.lin(preds) 

        #print(preds.permute(0,1,2).shape)
        #print(preds[0].shape)
        #if self.c.use_conv:
        preds = self.ch_agg(preds).squeeze(0).T
        #else:


        return preds
    
class HoodDS(Dataset):
    """Generate hoods around n_neighbors, a hyperparameter. This choice determines how many n_nearest neighbors
    the egnn encoder sees.
    
    This function takes as input the pdb.npz numpy files and returns an """
    def __init__(self, paths, keep_ids=False):
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
        return (zt,pt,yt,mt,data[3]) if keep_ids else (zt,pt,yt,mt)

    def split(paths):
        """deterministic and random"""
        if cfg.num_paths: paths=paths[:cfg.num_paths]
        rng=np.random.RandomState(cfg.split_seed)
        idx=rng.permutation(len(paths)); cut=int(len(paths)*cfg.split_ratio)
        #cut=1
        #idx=[0,1,2]
        return [paths[i] for i in idx[:cut]], [paths[i] for i in idx[cut:]]
# ================================================================
# 4) loaders
# ================================================================


def run_hoodDS(cfg):
    coll = lambda b: HoodDS.pad(b,cfg.hood_k,cfg.device,cfg.analysis_mode)
    tr,val=HoodDS.split(sorted(glob.glob(cfg.INPUTS_DIR)))
    train_ds=HoodDS(tr,cfg.hood_k); val_ds=HoodDS(val,cfg.hood_k)
    tr_loader=DataLoader(train_ds,batch_size=cfg.batch_size,shuffle=True , generator=g,collate_fn=coll); val_loader=DataLoader(val_ds,batch_size=cfg.batch_size,shuffle=False,collate_fn=coll)
    return tr_loader, val_loader
def run(cfg, loader,train):
    model.train() if train else model.eval(); loss_sum=0;n=0;oloss_sum=0
    for z,x,y,m,*_ in loader:
        v=m.view(-1); z=z.view(-1,z.size(2))[v].to(cfg.device)
        x=x.view(-1,x.size(2),3)[v].to(cfg.device); y=y.view(-1)[v].to(cfg.device)
        with autocast(enabled=(cfg.device=='cuda')):
            pred=model(z,x).flatten(); loss=p_fn(pred,y)
        if train:
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        if cfg.study_metrics:
            other_loss = v_fn(pred,y)
            oloss_sum +=other_loss.item(); 
        loss_sum+=loss.item(); n+=1
        
    if not cfg.study_metrics:
        return loss_sum/n
    else:
        return (loss_sum/n, oloss_sum/n)

# ………………………………………………………………………………………………………………………
# (‑‑‑‑‑‑‑‑‑‑‑‑‑ ALL YOUR ORIGINAL CODE UNCHANGED ‑‑‑‑‑‑‑‑‑‑‑‑‑)
# keep everything exactly as in your post:  invariance suite,
# Cfg class, Model, HoodDS, run_hoodDS, train loop, etc.
# ………………………………………………………………………………………………………………………
class Cfg(dict):
    # ================================================================
    # 0) dashboard – flip anything here
    # ================================================================
    __getattr__ = dict.__getitem__; __setattr__ = dict.__setitem__


# ---------------------------------------------------------------------
#  CLI PATCH  – added for ablation convenience
# ---------------------------------------------------------------------
def _str2bool(v):
    "argparse doesn't have BooleanOptionalAction in py3.6"
    if v.lower() in ('yes', 'true', 't', '1'):  return True
    if v.lower() in ('no',  'false', 'f', '0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def _add_cfg_args(parser, cfg_dict):
    """Dynamically add an argparse flag for **every** key in cfg."""
    for k, v in cfg_dict.items():
        # skip private / complex objects
        if k.startswith('_') or callable(v): 
            continue
        if isinstance(v, bool):
            parser.add_argument('--' + k, type=_str2bool, default=None,
                                help='bool (default: {})'.format(v))
        else:
            parser.add_argument('--' + k, type=type(v), default=None,
                                help='(default: {})'.format(v))

if __name__ == '__main__':
    import argparse, sys

    # -----------------------------------------------------------------
    # (1) build the *default* cfg exactly the same way you already do
    # -----------------------------------------------------------------
    cfg = Cfg(
        INPUTS_DIR="../../../data/pkegnn_INS/inputs/*.npz",
        dim=6, basis=6, depth=2, hidden_dim=4, dropout=0.00,
        hood_k=100, num_neighbors=11, norm_coors=True,
        epochs=1, num_paths=20,
        aggregator='nconv',
        use_rbf=True,
        use_attn=True,
        use_boost=True,
        use_prot=True,
        use_conv=True,
        conv_kernel=7,
        pegnn_radius=8,
        loss_type='mse',
        study_metrics=True,
        lr=5e-3, batch_size=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=0, analysis_mode=False,
        split_ratio=0.5, split_seed=0,
        runid=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    )

    # -----------------------------------------------------------------
    # (2) parse CLI overrides
    # -----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_cfg_args(parser, cfg)        # one flag per cfg entry
    args = parser.parse_args()

    # apply only non‑None overrides
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    # -----------------------------------------------------------------
    # (3) now run exactly the same code as before
    # -----------------------------------------------------------------
    print("Run‑ID:", cfg.runid)
    if cfg.seed != "no seed":
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        g = torch.Generator(); g.manual_seed(cfg.seed)
    else:
        g = None

    model = Model(cfg)
    print("params:", sum(p.numel() for p in model.parameters()))

    tr_loader, va_loader = run_hoodDS(cfg)
    p_fn = nn.L1Loss() if cfg.loss_type == 'mae' else nn.MSELoss()
    v_fn = nn.MSELoss() if cfg.loss_type == 'mae' else nn.L1Loss()
    opt  = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.99, patience=0, #changed factor when went to full ds
                                                      cooldown=0, min_lr=1e-16)
    scaler = GradScaler(enabled=(cfg.device == 'cuda'))

    # -------------- training loop (unchanged) -----------------------
    #cfg.epochs = 10#int(cfg.epochs)
    losses=[]#,[]
    for e in range(cfg.epochs):
        tr = run(cfg, tr_loader, True)
        va = run(cfg, va_loader, False)
        if not cfg.study_metrics:
            sch.step(va)
            print("[{}/{}]  train {:.4f} | val {:.4f}"
                  .format(e+1, cfg.epochs, tr, va))
        else:
            sch.step(va[0])
            print("[{}/{}]  train {} {:.4f} | val {:.4f}"
                  .format(e+1, cfg.epochs, cfg.loss_type, tr[0], va[0]))
            
            print("     additional metrics: train {:.4f} | val {:.4f}"
                  .format(tr[1], va[1]))
            print()
        losses.append((tr, va))

    # -------------- invariance suite & checkpoint (unchanged) -------
    ckpt_name = "ckpt_{}.pt".format(cfg['runid'])
    #torch.save({...}, ckpt_name)
    print("Saved checkpoint:", ckpt_name)

    print("\n================  INVARIANCE SUITE  ================\n")
    stats = run_invariance_suite(model, tr_loader, cfg,
                                 max_batches=3, rot_trials=5,
                                 atol=5e-4, rtol=5e-4, verbose=True)
    print("\n--------------  summary (max abs error) -------------")
    for k, v in stats.items():
        print("{:6}: {:.3e}".format(k, v))
    print("-----------------------------------------------------")
    print("✔  thresholds used: atol=5e‑4, rtol=5e‑4")

    from matplotlib import pyplot as plt

    # split the recorded metrics --------------------------------------------------
    t = [a[0] for a in losses]      # train entries  -> [(mse, mae), …]
    v = [a[1] for a in losses]      # validation     -> [(mse, mae), …]

    tr_mse = [m[0] for m in t]
    tr_mae = [m[1] for m in t]
    val_mse = [m[0] for m in v]
    val_mae = [m[1] for m in v]
#todo
    # plot ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    # ── subplot 1 : MSE ───────────────────────────────────────────────────────────
    axes[0].plot(tr_mse, label='train MSE')
    axes[0].plot(val_mse, label='val MSE')
    axes[0].set_title('MSE')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()

    # ── subplot 2 : MAE ───────────────────────────────────────────────────────────
    axes[1].plot(tr_mae, label='train MAE')
    axes[1].plot(val_mae, label='val MAE')
    axes[1].set_title(f'MAE')
    axes[1].set_xlabel('epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.show()