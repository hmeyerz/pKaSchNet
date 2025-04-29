
from egnn_pytorch import EGNN_Network
import torch
import torch.nn as nn
import numpy as np
import glob
import time
import gzip
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) *
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            # Handle odd dimensions by filling the remaining column with cos()
            pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor with shape (seq_length, batch_size, embed_dim)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class SimpleMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SimpleMultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        """
        x: Tensor of shape (seq_length, batch_size, embed_dim)
        """
        attn_outputs, attn_weights = self.multihead_attn(x, x, x)
        return attn_outputs


net=EGNN_Network(dim=6,
    depth=2,
    num_positions=500,
    num_tokens=200,
    num_nearest_neighbors=2, 
    num_edge_tokens=2,
    global_linear_attn_every=1,
    global_linear_attn_dim_head=6,
    num_global_tokens=2,
    adj_dim=3,
    fourier_features=5,
    m_dim=10,
    dropout=0.3)

lin=nn.Linear(6,1)
A = PositionalEncoding(22)
mha = SimpleMultiheadAttention(22,1)
losses=[]
paths=np.char.array(glob.glob("/home/jrhoernschemeyer/Desktop/data_prep/inputs/*.npz"))
val=paths[np.random.random_integers(low=0,high=len(paths)-2,size=np.int(0.4*len(paths)))]
train=np.char.array(list(set(paths).difference(set(val))))
np.savez_compressed("/home/jrhoernschemeyer/Desktop/data_prep/split.npz",val=val, train=train)

optimizer= torch.optim.Adam(list(net.parameters()) + list(mha.parameters()) + list(lin.parameters()), lr=.01, weight_decay=0.01)
criterion = nn.HuberLoss()
to=time.time()

for i in range(25):
    
    train = list(np.array(train)[np.random.permutation(len(train))])
    print("epoch",i)
    for path in train:
        losses=[]
        pdb=np.char.encode(path[-8:-4])
        mha.train()
        net.train()
        lin.train()
        optimizer.zero_grad()
        a=np.load(path,allow_pickle=True)
        zs,xs,targets=a["z"],a["pos"],a["pks"]
        n = zs.shape[0]
        #shuffle
        idx = np.random.permutation(n)
        zs,xs,targets=zs[idx],xs[idx],targets[idx]
        
        for z,x,y in zip(zs,xs,targets):
            x=torch.tensor(list(x)).unsqueeze(0)
            z=torch.tensor(list(z),dtype=torch.int32)
            #remove H
            #mask = (z != 1)
            #z = z[mask]
            #x = x[mask]

            out=net(z,x)
            pk=torch.sum(out[0],dim=1)/(torch.max(out[0]))
            out=lin(pk)

            
            loss = criterion(torch.sum(out),torch.tensor(y))
            losses.append(np.round(loss.item(),3))
            loss.backward()
            optimizer.step()
        with gzip.open("/home/jrhoernschemeyer/Desktop/data_prep/trainresults_final.gz","a") as f:
            f.write(np.char.encode(str(np.round(np.mean(losses),3))))
            f.write(b" ")
            f.close()
        
    for path in val:
        mha.eval()
        net.eval()
        lin.eval()

        a=np.load(path,allow_pickle=True)
        zs,xs,targets=a["z"],a["pos"],a["pks"]
        #n = zs.shape[0]
        #shuffle
        #idx = np.random.permutation(n)
        #zs,xs,targets=zs[idx],xs[idx],targets[idx]
        
        for z,x,y in zip(zs,xs,targets):
            x=torch.from_numpy(np.array(x)).unsqueeze(0)
            z=torch.from_numpy(np.array(z)).int()#dtype=torch.int32)
            out=net(z,x)
            pk=torch.sum(out[0],dim=1)/(torch.max(out[0]))
            out=lin(pk)
            loss = criterion(torch.sum(out),torch.tensor(y))
            losses.append(loss.item())
                
            
        with gzip.open("/home/jrhoernschemeyer/Desktop/data_prep/valresultsfinal.gz","a") as f:
            f.write(np.char.encode(str(np.round(np.mean(losses).item(),3))))
            f.write(b" ")
            f.close()
        print((time.time()-to)/60,"minutes")
        
            
    torch.save(net.state_dict(),"/home/jrhoernschemeyer/Desktop/data_prep/egnnfinal")
    torch.save(mha.state_dict(),"/home/jrhoernschemeyer/Desktop/data_prep/mhafinal")
    torch.save(mha.state_dict(),"/home/jrhoernschemeyer/Desktop/data_prep/mhafinal")#nettest")    


