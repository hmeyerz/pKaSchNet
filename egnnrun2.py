from egnn_pytorch import EGNN_Network
import torch.nn as nn
import numpy as np
#import random
import torch
#from matplotlib import pyplot as plt
import glob
#from collections import defaultdict, OrderedDict
#import dask.dataframe as dd
#import math
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


def EGNN1():
    """This layer creates an embedding as well as hopefully navigates dimerrors produced by putting molecules with
    varying node size.
    in
    feats = 1 x n_atoms

    input can be minimum one atom, this accomdates for rogue solo cofactors and unhydrogonated water."""
    net = EGNN_Network(
        num_positions = 500, # unless what you are passing in is an unordered set, set this to the maximum sequence length
        dim = 1,
        depth=2,
        num_tokens=200,
        num_nearest_neighbors=2,
        dropout=0.1)
    return net



net = EGNN1()
A = PositionalEncoding(22)
mha = SimpleMultiheadAttention(22,1)
optimizer2 = torch.optim.Adam(list(net.parameters())+list(mha.parameters()), lr=.001, weight_decay=0.01)
criterion = nn.HuberLoss()


to=time.time()
loss=None
paths=glob.glob("/home/jrhoernschemeyer/Desktop/data_prep/nometals/inputs/*")
paths=np.array(paths)
val=list(paths[np.random.random_integers(low=0,high=len(paths)-2,size=np.int(0.3*len(paths)))])
train=list(set(paths).difference(set(val)))
paths=list(paths)
#val.append()

for i in range(200):
    #shuffle pdbs
    n = len(train)#zs.shape[0]
    idx = np.random.permutation(n)
    train=np.array(train)
    train = list(train[idx])

    try: #train
        print("epoch",i)
        for j,path in enumerate(train):
            pdb=np.char.encode(path[-8:-4])
            #break
            mha.train()
            net.train()
            optimizer2.zero_grad()

            #pdb=path[-7:-3]
            data=np.load(path,allow_pickle=True)
            zs,xs,targets,ids=data["z"],data["R"],data["y"],data["ids"]
            n = zs.shape[0]
            #shuffle
            idx = np.random.permutation(n)
            
            if len(zs) != len(targets):
                #print("!",pdb,"MIA")
                continue
            zs,xs,targets,ids=zs[idx],xs[idx],targets[idx],ids[idx]

            for z,x,y,id in zip(zs,xs,targets,ids):
                # z: numpy array of shape (n,)
                #z = torch.from_numpy(z).float()      # -> (n,)
                #z = torch.from_numpy(z).int().unsqueeze(0) # -> (1, n, 1)
                #x= torch.from_numpy(x)     # -> (n, 3) or (n, n)
                #x = x.unsqueeze(0) 
                #               # -> (1, n, 3) or (1, n, n)
                
                #out = net(z,x)#torch.tensor(x,dtype=torch.int32).unsqueeze(-3))
                try:
                    out = net(torch.from_numpy(z).int().unsqueeze(0),torch.from_numpy(x).unsqueeze(0))
                    
                    #out = net(torch.from_numpy(z).unsqueeze(0),torch.from_numpy(x).unsqueeze(0))
                    pk=torch.mean(torch.mean(torch.mean(mha(A(out[0][0].unsqueeze(2))), dim=0),dim=0))
                    loss = criterion(pk,torch.tensor(y))
                    loss.backward()
                    optimizer2.step()
                    with gzip.open("/home/jrhoernschemeyer/Desktop/data_prep/trainresults7.gz","a") as f:
                        f.write(np.char.encode(str(np.round(loss.item(),3))))# + b" " + id + pdb + b"\n")
                        #f.write(np.char.encode(str(y)))
                        #f.write(b" ")
                        #f.write(pdb)
                        #f.write(np.char.encode(str(pk.item())))
                        #f.write(b" ")
                        #f.write(np.char.encode(str(loss.item())))
                        #f.write(b" ")
                        #f.write(id)
                        #f.write(b" ")
                        #f.write(np.char.encode(pdb))
                        f.write(b"\n")
                    #f.write(np.char.encode(path[-7:-3]))
                    f.close()
                except:# Exception as e:
                        del train[j]
                        #print("except",pdb,id,e)
                        #print("lass train loss",loss)
                        continue
            #print((time.time()-to)/60)
            

                
            
            #print(out[])
            #zz,cc,tt,ii=b["z"], b["R"],b["y"],b["ids"]


        #A_shuffled = zs[idx]
        #B_shuffled = B[idx]
        #C_shuffled = C[idx]
        #D_shuffled = D[idx]
        #A_shuffled
        print("validating. last loss",loss.item(),"time",(time.time()-to)/60,"minutes")
        
        for k,path in enumerate(val):
            mha.eval()
            net.eval()
            pdb=np.char.encode(path[-8:-4])
            
            #print((loss,"lastloss",time.time()-to)/60,"minutes")
            data=np.load(path,allow_pickle=True)
            zs,xs,targets,ids=data["z"],data["R"],data["y"],data["ids"]
            if len(zs) != len(targets):
                
                #print("!",pdb,"MIA")
                continue
            for z,x,y,id in zip(zs,xs,targets,ids):
                    try:
                        with torch.no_grad():
                            out = net(torch.tensor(z,dtype=torch.int32).unsqueeze(0),torch.from_numpy(x).unsqueeze(0))
                            pk=torch.mean(torch.mean(torch.mean(mha(A(out[0][0].unsqueeze(2))), dim=0),dim=0))
                            loss = criterion(pk,torch.tensor(y))
                            #loss.backward()
                            #optimizer2.step()
                        with gzip.open("/home/jrhoernschemeyer/Desktop/data_prep/valresults7.gz","a") as f:
                            #f.write(np.char.encode(str(y)))
                            #f.write(b" ")
                            #f.write(np.round(pk.item()
                            #f.write(b" ")
                            f.write(np.char.encode(str(np.round(loss.item(),3))))# + b" " + id + pdb + b"\n")
                            #f.write(pdb)
                            #f.write(b" ")
                            #f.write(id + b" " )
                            #f.write(b" ")
                            #f.write(np.char.encode(pdb))
                            f.write(b"\n")
                        #f.write(np.char.encode(path[-7:-3]))
                    except:# Exception as e:
                        del val[k]
                        #print("except",pdb,id,e)
                        #print("last val loss",loss)
                        continue
                        #except Exception as e:

                #f.close()
        #print((loss,"lastloss",time.time()-to)/60,"minutes")

    except Exception as e:
        print(e)
        print(pdb)
        torch.save(net.state_dict(),"/home/jrhoernschemeyer/Desktop/data_prep/nometals/egnn6_>200epoch_669structs")
        torch.save(mha.state_dict(),"/home/jrhoernschemeyer/Desktop/data_prep/nometals/mha6_>200epoch_669structs")#nettest")    
        continue
        #print("200 epochs, 659 pdb structures (60/40 split). minimal model.", (time.time()-to)/60,"minutes")


                

torch.save(net.state_dict(),"/home/jrhoernschemeyer/Desktop/data_prep/nometals/egnn6_200epoch_669structs")
torch.save(mha.state_dict(),"/home/jrhoernschemeyer/Desktop/data_prep/nometals/mha6_200epoch_669structs")#nettest")
#print(loss, "last loss")    
print("200 epochs, 659 pdb structures (60/40 split). less minimal model/reduced LR.", (time.time()-to)/60,"minutes")


