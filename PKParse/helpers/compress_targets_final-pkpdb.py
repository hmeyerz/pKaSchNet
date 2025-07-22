import polars as pl
from collections import OrderedDict
import gzip
import time
import subprocess

to = time.time()

# Read the full CSV, preserving all columns
#df = pl.read_csv("/home/pbuser/Desktop/Jessi/data_prep/pkas.csv", separator=";")
f="/home/jrhoernschemeyer/Desktop/data_prep/pkas.csv"
df = pl.read_csv(f, separator=";",skip_rows=0)
d=df.to_dict(as_series=False)
import numpy as np
a=np.array(list(d.values()))
sorted_indices = np.argsort(a)
# Squeeze to remove the extra dimension
a_flat = a.squeeze()               # Now shape becomes (12628148,)
indices_flat = sorted_indices.squeeze()

# Use the flattened indices to retrieve sorted data
sorted_data = a_flat[indices_flat]



parts = np.char.partition(sorted_data, ";")

pdbs=parts[:,0]
resi_data=parts[:,2]
arr = pdbs#np.array([...])  # your sorted data

unique_vals, counts = np.unique(arr, return_counts=True)

badtargets=[]

print("unique pdbs gotten, starting parsing targets", (to - time.time())/60)
code = {"H":"0",
        "A":"1",
        "L":"2",
        "T":"3",
        "G":"4",
        "C":"5"}
refpk = {
    "A": np.float32(3.79),
    "C": np.float32(8.67),
    "G": np.float32(4.20),
    "H": np.float32(6.74),
    "L": np.float32(10.46),
    "T": np.float32(9.59),
}

#
split_indices = np.cumsum(counts)[:-1]
subs=np.split(resi_data,split_indices)




for pdbdata,pdb in zip(subs,unique_vals):
    try:
        par = np.char.rpartition(pdbdata, ";")
        info = [np.char.split(np.atleast_1d(line[0]),";")[0] for line in par if line[0]]
        
        ids= np.char.encode([np.char.add(np.char.add(i[2],i[0]),code[i[1][0]]) for i in info if not np.char.startswith(i[1],"CT") and not np.char.startswith(i[1],"N") ])
        #ids = [np.char.add(np.char.add(i[2],i[0]),code[i[1][0]]) for i in info if not np.char.startswith(i[1],"CT") and not np.char.startswith(i[1],"N") ]
        
        pks=[np.round(np.float32(p[2]) - refpk[i[1][0][0]],6) for i,p in zip(info,par) if not np.char.startswith(i[1],"CT") and not np.char.startswith(i[1],"N")]

        np.savez_compressed(f"/home/jrhoernschemeyer/Desktop/data_prep/targets/{pdb}.npz",pks=pks,ids=ids)
        #np.load(f"/home/jrhoernschemeyer/Desktop/data_prep/targets/{pdb}.npz")
    except Exception as e:
        badtargets.append(pdb)
        print(pdb)
        print(e)

with gzip.open("/home/jrhoernschemeyer/Desktop/data_prep/badtargets.gz", "wb") as f:
    for pdb in badtargets:
        f.write(pdb.encode())
        f.write(b" ")



print("finished", (time.time()-to)/60)