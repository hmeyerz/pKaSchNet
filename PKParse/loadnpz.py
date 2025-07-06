import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np

np.random.seed(0)


class InMemoryNpzDataset(Dataset):
    """load entire dataset into RAM as torch.Tensors"""
    def __init__(self, paths, pin_memory=False):
        self.data = []
        for p in paths:
            a = np.load(p, allow_pickle=True)
            zs = [torch.tensor(z_i, dtype=torch.int32)   for z_i in a["z"]]
            xs = [torch.tensor(x_i, dtype=torch.float32) for x_i in a["pos"]]
            ys = [torch.tensor(y_i, dtype=torch.float32) for y_i in a["pks"]]
            if pin_memory:
                zs = [z.pin_memory() for z in zs]
                xs = [x.pin_memory() for x in xs]
                ys = [y.pin_memory() for y in ys]
            self.data.append((zs, xs, ys))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def loader(pin_memory,val_split,num_files):
    paths  = np.char.array(glob.glob("./inputs/*.npz"))
    np.shuffle(paths)
    train,val=1-num_files*(.001 * val_split), num_files*(.001 * val_split)
    train_ds = InMemoryNpzDataset(glob.glob("./inputs/*.npz")[train:])
    val_ds   = InMemoryNpzDataset(glob.glob("./inputs/*.npz")[train:val])

    def collate_graphs(batch):
        return batch
    
    train_loader = DataLoader(
        train_ds, batch_size=50, shuffle=True,
        num_workers=0, pin_memory=pin_memory, collate_fn=collate_graphs
    )
    val_loader = DataLoader(
        val_ds, batch_size=50, shuffle=False,
        num_workers=0, pin_memory=pin_memory, collate_fn=collate_graphs
    )

    return train_loader, val_loader
