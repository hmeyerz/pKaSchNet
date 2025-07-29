

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
from helpers import run_invariance_suite
from model import Model, HoodDS
# ================================================================
# 3) model
# ================================================================


def run_hoodDS(cfg):
    coll = lambda b: HoodDS.pad(b,cfg.hood_k,cfg.device,cfg.analysis_mode)
    tr,val=HoodDS.split(sorted(glob.glob(cfg.INPUTS_DIR)),cfg)
    train_ds=HoodDS(tr,cfg); val_ds=HoodDS(val,cfg)
    tr_loader=DataLoader(train_ds,batch_size=cfg.batch_size,shuffle=True , generator=g,collate_fn=coll); val_loader=DataLoader(val_ds,batch_size=cfg.batch_size,shuffle=False,collate_fn=coll)
    return tr_loader, val_loader

def run_model(cfg, loader,train):
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
        n_aggregator='nconv',
        f_aggregator='rconv',
        use_rbf=True, use_attn=True, use_boost=True, use_prot=True,
        pegnn_radius=8,
        loss_type='mse',
        study_metrics=True,
        lr=5e-3, batch_size=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=0, analysis_mode=False, #TODO
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
    if cfg.seed != "no seed": #TODO this wont work in fn split
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
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.99, patience=0, 
                                                      cooldown=0, min_lr=1e-16)
    scaler = GradScaler(enabled=(cfg.device == 'cuda'))

    # -------------- training loop (unchanged) -----------------------
    #cfg.epochs = 10#int(cfg.epochs)
    losses=[]#,[]
    for e in range(cfg.epochs):
        tr = run_model(cfg, tr_loader, True)
        va = run_model(cfg, va_loader, False)
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