# ──────────────────────────────────────────────────────────────
# train_utils.py   –  all the boiler‑plate in one place
# ──────────────────────────────────────────────────────────────
import os, random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from torch.cuda.amp import autocast, GradScaler

# ------------------------------------------------------------------
def set_seed(seed):
    """Full reproducibility for Python, NumPy, Torch & CUDA."""
    if seed is None: return
    random.seed(seed);          np.random.seed(seed)
    torch.manual_seed(seed);    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
# put this near the top of train_utils.py
# ---------------------------------------------------------------
def make_split(all_paths, cfg):
    """
    Returns train_paths, val_paths according to cfg.split_mode
      cfg.split_mode = 'random'  →  random % of data (seeded)
      cfg.split_mode = 'file'    →  read two text files with paths/IDs
    """
    if cfg.split_mode == 'random':
        rng = np.random.RandomState(cfg.split_seed)
        idx = rng.permutation(len(all_paths))
        k   = int(len(all_paths) * cfg.split_ratio)
        train_idx, val_idx = idx[:k], idx[k:]
        return [all_paths[i] for i in train_idx], [all_paths[i] for i in val_idx]

    elif cfg.split_mode == 'file':
        def read_list(txt):
            with open(txt) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # if the list contains naked IDs, map → full path
            if not lines or lines[0].endswith('.npz'):
                return lines
            id2path = {os.path.splitext(os.path.basename(p))[0]: p for p in all_paths}
            return [id2path[x] for x in lines if x in id2path]

        return read_list(cfg.split_files['train']), read_list(cfg.split_files['val'])
    else:
        raise ValueError("split_mode must be 'random' or 'file'")

# ------------------------------------------------------------------
class InMemoryHoodDataset(Dataset):
    """
    RAM‑resident neighbourhoods.
    __getitem__ returns (z,pos,y,id_str)
    """
    def __init__(self, paths, n_neighbors, pin_memory=False):
        super().__init__()
        self.data, self.ids = [], []
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")

        for p in paths:
            try:
                dat      = np.load(p, allow_pickle=True)
                z_all    = dat["z"].astype(np.int32)
                pos_all  = dat["pos"].astype(np.float32)
                sites    = dat["sites"].astype(np.float32)
                y        = dat["pks"].astype(np.float32)

                if len(sites) == 0: continue        # skip empty entries

                nbrs.fit(pos_all)
                idx = nbrs.kneighbors(sites, return_distance=False)

                z_hood   = torch.from_numpy(z_all[idx])
                pos_hood = torch.from_numpy(pos_all[idx])
                y        = torch.from_numpy(y)

                if pin_memory:
                    z_hood = z_hood.pin_memory(); pos_hood = pos_hood.pin_memory(); y = y.pin_memory()

                self.data.append((z_hood, pos_hood, y))
                self.ids .append(os.path.splitext(os.path.basename(p))[0])
            except Exception as e:
                print("skipping", p, ":", e)

    def __len__(self):  return len(self.data)
    def __getitem__(self, i):
        z,pos,y = self.data[i]
        return z,pos,y,self.ids[i]          # tuple‑len == 4

# ------------------------------------------------------------------
def pad_collate(batch, N_NEIGHBORS, device):
    """
    Accepts list[(z,pos,y,id), …].
    Returns  (zs,pos,ys,mask,ids_list)
    """
    ids = [b[3] for b in batch]        # keep IDs untouched

    B      = len(batch)
    S_max  = max(b[0].shape[0] for b in batch)

    zs   = torch.zeros (B, S_max, N_NEIGHBORS,   dtype=torch.int32 , device=device)
    pos  = torch.zeros (B, S_max, N_NEIGHBORS,3, dtype=torch.float32, device=device)
    ys   = torch.full  ((B, S_max), float("nan"), dtype=torch.float32, device=device)
    mask = torch.zeros (B, S_max,               dtype=torch.bool,     device=device)

    for b,(z_b,pos_b,y_b,_) in enumerate(batch):
        S = z_b.shape[0]
        zs  [b,:S] = z_b.to(device)
        pos [b,:S] = pos_b.to(device)
        ys  [b,:S] = y_b.to(device)
        mask[b,:S] = True
    return zs,pos,ys,mask,ids

# ------------------------------------------------------------------
def one_epoch(model, loader, device, primary_fn, secondary_fn,
              optimizer=None, scaler=None):
    train = optimizer is not None
    if train: model.train()
    else:     model.eval()

    p_sum=s_sum=0.0; n=0
    for z,pos,y,mask,ids in loader:
        valid = mask.view(-1)
        z_r   = z.view(-1, z.size(2))[valid].to(device)
        x_r   = pos.view(-1, pos.size(2), 3)[valid].to(device)
        y_r   = y.view(-1)[valid].to(device)

        with autocast(enabled=(device=='cuda')):
            pred   = model(z_r, x_r).flatten()
            loss   = primary_fn(pred, y_r)
            p_val  = loss.item()
            s_val  = secondary_fn(pred, y_r).item() if secondary_fn else 0.0

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

        p_sum += p_val;  s_sum += s_val;  n += 1
    return p_sum/n, (s_sum/n if secondary_fn else None)


def epoch_loop(cfg, model, loader, train, primary_fn, secondary_fn, 
              optimizer=None, scaler=None):
    if train: model.train()
    else:     model.eval()

    p_sum=s_sum=0.0; n=0
    for z,pos,y,mask,ids in loader:      # ← 5‑tuple from train_utils.collate
        valid = mask.view(-1)
        z_r   = z.view(-1, z.size(2))[valid].to(cfg.device)
        x_r   = pos.view(-1, pos.size(2), 3)[valid].to(cfg.device)
        y_r   = y.view(-1)[valid].to(cfg.device)

        with autocast(enabled=(cfg.device=='cuda')):
            pred = model(z_r, x_r).flatten()
            loss = primary_fn(pred, y_r)
            p_val = loss.item()
            s_val = secondary_fn(pred, y_r).item() if secondary_fn else 0.0

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

        p_sum += p_val;  s_sum += s_val;  n += 1

        # ---- OPTIONAL: keep ids & attention for analysis ----------
        # if cfg.save_attn:
        #     save_your_stuff(ids, model.attn.module.attn_weights)

    return p_sum/n, (s_sum/n if secondary_fn else None)