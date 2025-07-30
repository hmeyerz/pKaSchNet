#winner
import torch, torch.nn as nn
import numpy as np
from egnn_pytorch import EGNN
from architecture import (StackedEGNN,
                          LearnableRBF,
                          AttentionBlock,
                          TunableBlock)
import time, datetime
import glob
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader


torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

# ================================================================
# 0) dashboard – flip anything here
# ================================================================
class Cfg(dict):
    __getattr__ = dict.__getitem__; __setattr__ = dict.__setitem__


# ================================================================
# 1) reproducibility
# ================================================================
import random, os, numpy as np, torch, glob, datetime


# ================================================================
# 2) dataset helpers
# ================================================================
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
class HoodDS(Dataset):
    def __init__(self, paths, k):
        self.data=[]; self.ids=[]
        nbr=NearestNeighbors(n_neighbors=k,algorithm='brute')
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
        z,p,y=self.data[i]; return z,p,y,self.ids[i]

def pad(batch,k,device,ret_ids):
    ids=[b[3] for b in batch] if ret_ids else None
    B=len(batch); S=max(b[0].shape[0] for b in batch)
    zt=torch.zeros(B,S,k,dtype=torch.int32,device=device)
    pt=torch.zeros(B,S,k,3,dtype=torch.float32,device=device)
    yt=torch.full((B,S),float('nan'),device=device); mt=torch.zeros(B,S,dtype=torch.bool,device=device)
    for b,(z,p,y,_) in enumerate(batch):
        s=z.shape[0]; zt[b,:s]=z; pt[b,:s]=p; yt[b,:s]=y; mt[b,:s]=True
    return (zt,pt,yt,mt,ids) if ret_ids else (zt,pt,yt,mt)

def split(paths):
    if cfg.num_paths: paths=paths[:cfg.num_paths]
    rng=np.random.RandomState(cfg.split_seed)
    idx=rng.permutation(len(paths)); cut=int(len(paths)*cfg.split_ratio)
    return [paths[i] for i in idx[:cut]], [paths[i] for i in idx[cut:]]

# ================================================================
# 3) model
# ================================================================

class Model(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        C = c.dim + c.basis

        self.egnn = StackedEGNN(c.dim,c.depth,c.hidden_dim,c.dropout,
                                c.hood_k,98,c.num_neighbors,c.norm_coors).to(c.device)

        self.rbf  = TunableBlock(LearnableRBF(c.basis,10.).to(c.device), c.use_rbf)
        self.attn = TunableBlock(AttentionBlock(C,C,c.hidden_dim).to(c.device), c.use_attn)

        if c.aggregator=='linear':
            self.agg = nn.Linear(C,1).to(c.device)
        elif c.aggregator=='nconv':
            self.agg = nn.Conv1d(c.hood_k,1,kernel_size=C,padding=0).to(c.device)
        elif c.aggregator=='pool':
            self.agg = None
        else: raise ValueError("aggregator must be 'linear' | 'nconv' | 'pool'")

        self.boost = nn.Linear(1,1).to(c.device) if c.use_boost else nn.Identity()
        self.prot  = EGNN(dim=1,update_coors=True,num_nearest_neighbors=2).to(c.device) \
                     if c.use_prot else nn.Identity()
        self.conv  = nn.Conv1d(1,1,c.conv_kernel,padding=c.conv_kernel//2).to(c.device) \
                     if c.use_conv else nn.Identity()

    def forward(self,z,x):
        h,coord=self.egnn(z,x); h=h[0]                # (R,N,dim)
        cent=coord.mean(1,keepdim=True)               # (R,1,3)

        # --- build token ----------------------------------------------------------------
        r = self.rbf(cent,coord).transpose(1,2) if self.c.use_rbf else \
            h.new_zeros(h.size(0),self.c.basis,self.c.hood_k)
        tok = torch.cat((r,h.transpose(1,2)),1)       # (R,C,N)

        att = self.attn(tok.permute(2,0,1))
        tok = att[0] if isinstance(att,(tuple,list)) else att
        tok = tok.permute(1,0,2)                      # (R,N,C)

        # --- aggregation ----------------------------------------------------------------
        if self.c.aggregator=='linear':
            preds = self.agg(tok) .max(1).values                # (R,1)
        elif self.c.aggregator=='nconv':
            preds = self.agg(tok).squeeze(-1)                   # (R,1)
        else:   # pool
            preds = tok.max(1).values.mean(1,keepdim=True)      # (R,1)

        preds = self.boost(preds)

        if self.c.use_prot:
            preds = self.prot(preds.unsqueeze(0),
                              cent.permute(1,0,2))[0].squeeze(0)

        if self.c.use_conv:
            preds = self.conv(preds.T.unsqueeze(0)).squeeze(0).T

        return preds
t0=time.time()
cfg = Cfg(
    # backbone
    dim=6, basis=6, depth=2, hidden_dim=4, dropout=0.04,
    hood_k=100, num_neighbors=11, norm_coors=True,

    epochs=200,

    # aggregation: 'linear' | 'nconv' | 'pool'
    aggregator   ='linear',

    # block switches
    use_rbf      =True,
    use_attn     =True,
    use_boost    =True,     # Linear(1→1) after aggregator
    use_prot     =True,      # protein‑level EGNN
    use_conv     =True,     # 1‑D conv after prot EGNN
    conv_kernel  =7,

    # training
    loss_type='mse',
    study_metrics=True,
    lr=5e-3, batch_size=1, #batchsize not safw to inc

    # misc
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=0, analysis_mode=False,
    num_paths=None, split_ratio=0.5, split_seed=0,
    runid=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
)
print("Run‑ID:", cfg.runid)
random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

model=Model(cfg); 
print("params:",sum(p.numel() for p in model.parameters()))

# ================================================================
# 4) loaders
# ================================================================
allp=glob.glob("1000inputs/*.npz")
tr,val=split(allp)
train_ds=HoodDS(tr,cfg.hood_k); val_ds=HoodDS(val,cfg.hood_k)
coll=lambda b: pad(b,cfg.hood_k,cfg.device,cfg.analysis_mode)
tr_loader=DataLoader(train_ds,batch_size=cfg.batch_size,shuffle=True ,collate_fn=coll)
va_loader=DataLoader(val_ds,batch_size=cfg.batch_size,shuffle=False,collate_fn=coll)

# ================================================================
# 5) training utils
# ================================================================
p_fn = nn.L1Loss() if cfg.loss_type=='mae' else nn.MSELoss()
v_fn = nn.MSELoss() if cfg.loss_type=='mae' else nn.L1Loss()
opt  = torch.optim.AdamW(model.parameters(),lr=cfg.lr)
sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',0.5,3)
from torch.cuda.amp import GradScaler, autocast
scaler=GradScaler(enabled=(cfg.device=='cuda'))

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

# ================================================================
# 6) train
# ================================================================
for e in range(cfg.epochs):
    tr=run(cfg,tr_loader,True)
    va=run(cfg,va_loader,False)
    if not cfg.study_metrics:
        sch.step(va)
        print(f"[{e+1}/{cfg.epochs}]  train {tr:.4f} | val {va:.4f}")
    else:
        sch.step(va[0])
        print(f"[{e+1}/{cfg.epochs}]  train {cfg.loss_type} {tr[0]:.4f} | {cfg.loss_type} val {va[0]:.4f}")
        print("     additional metrics: ",f"[{e+1}/{cfg.epochs}]  train {tr[1]:.4f} | val {va[1]:.4f}")
        print("")

print(time.time() - t0,"sec")

ckpt = {
    'model_state': model.state_dict(),
    'optim_state': opt.state_dict(),
    'sched_state': sch.state_dict(),
    'cfg'        : cfg,
}
ckpt_name = "ckpt_{}.pt".format(cfg['runid'])
#torch.save(ckpt, ckpt_name)
print("Saved checkpoint:", ckpt_name)
