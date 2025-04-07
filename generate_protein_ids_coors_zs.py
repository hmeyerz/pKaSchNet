import glob
import time


anions = {b"HIS":("ND1", "ND2"),
        b"ASP":("OD1","OD2"),
        b"LYS":("NZ"),
        b"TYR":("OH"),
        b"GLU":("OE1", "OE2"),
        b"CYS":("SG")}
code = {b"HIS":b"0",
        b"ASP":b"1",
        b"LYS":b"2",
        b"TYR":b"3",
        b"GLU":b"4",
        b"CYS":b"5"}

elements = {
    b"H": 1, b"C": 6, b"N": 7, b"O": 8, 
    b"NA": 11, b"MG": 12,  b"P": 15, b"S": 16, b"CL": 17, b"K": 19,
    b"CA": 20, b"MN": 25, b"FE": 26, 
    b"CU": 29, b"ZN": 30}

import numpy as np
import gzip
import torch

to = time.time()
paths=glob.glob("/home/jrhoernschemeyer/Desktop/data_prep/structures/pdbs/*.gz")
#print(paths)
badpaths=[]

for path in paths:
    try:
            
        with gzip.open(path, "r") as f:#"/home/jrhoernschemeyer/Desktop/data_prep/structures/" + pdb + ".g", "r") as f:
            #resis, counter = OrderedDict(), 0
            lines=np.array(f.readlines())

        lines=np.array([line for line in lines if np.char.startswith(line,b"ATOM")])#for line in lines if np.char.startswith(line,"ATOM")])
        infos = [np.char.split(np.atleast_1d(L))[0] for L in lines]
        infos = [[info[n] for n in [3,4,5,6,7,8,-1] if info[3] in list(anions.keys())] for info in infos]#[[info[n] for n in [4,5,6,7,8,11] if info[3] in list(anions.keys())] for info in infos])
        ids,counts=np.unique([np.char.add(np.char.add(np.char.add(i[2],i[1]),b" "),code[i[0]]) for i in infos if i],return_counts=True)
        coors=[[np.float32(info[i]) for i in [3,4,5]] for info in infos if info]
        species=[elements[((info[-1].strip(b"-+1234567890")))] for info in infos if info ] 

    #for i,path in enumerate(paths):

        with gzip.open(f"/home/jrhoernschemeyer/Desktop/data_prep/structures/ids/{path[-11:-7]}.gz", "wb") as f:
            for id, count in zip(ids,counts):
                #print(id,count)
                f.write(f"{np.char.decode(id).item()} {count}\n".encode())

        torch.save(torch.tensor(coors),f"/home/jrhoernschemeyer/Desktop/data_prep/structures/coors/{path[-11:-7]}")
        torch.save(torch.tensor(species),f"/home/jrhoernschemeyer/Desktop/data_prep/structures/species/{path[-11:-7]}")


    except:
        badpaths.append(path)
        print("F-", path)
        continue
     
print("done.time", (time.time() - to)/60)


with open("badpaths","w") as f:
    for path in badpaths:
        f.write(path)

print("fails", badpaths)