#make species and coord tensors for metals noH and reduce
import glob
import gzip
import numpy as np
import torch
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

to=time.time()
badpaths=[]

#metals_H, metals_noH = glob.glob(path + ""),glob.glob(path + "")
#nometals_noH = glob.glob(path + "zipped-fixed-noH/*.gz")

def generate_data(metals=False, hydrogens=False):
    """full pdbwise, into targets.txt file, coords and species tensors.
    
    re: saving IDs which are used to align labels, for this specific simulataneous database production (4 databases, 2 structures per pdb), since "noHs" includes all the resis
    that reduced (hs) does, I only need to generat the structure IDs (resinames), once.
    
    TODO: integrate using the fixed structure, if there is no reduced."""
    path = '/home/jrhoernschemeyer/Desktop/data_prep/'

    paths = glob.glob(path + "/nometals/zipped-reduced/*.gz") + glob.glob(path + "/nometals/zipped-reduced-oddname/*.gz")
            
        
    for path in paths:
        
        pdb = path[-7:-3]
        #print(pdb)

        #generate coords
        try:
            with gzip.open(path, "r") as f:#"/home/jrhoernschemeyer/Desktop/data_prep/structures/" + pdb + ".g", "r") as f:
                #resis, counter = OrderedDict(), 0
                lines=np.array(f.readlines())
                f.close()

            lines=np.array([line for line in lines if np.char.startswith(line,b"ATOM")])#for line in lines if np.char.startswith(line,"ATOM")])
            infos = [np.char.split(np.atleast_1d(L))[0] for L in lines]
            infos = [[info[n] for n in [3,4,5,6,7,8,11] if info[3] in list(anions.keys()) and info[8] != b"1.00" and info[7] != b"1.00"] for info in infos]#[[info[n] for n in [4,5,6,7,8,11] if info[3] in list(anions.keys())] for info in infos])
            infos = [i for i in infos if i]
            ids,counts=np.unique([np.char.add(np.char.add(np.char.add(i[2],i[1]),b" "),code[i[0]]) for i in infos if i],return_counts=True) #need counts for hoods
            

            coors=[[np.float32(info[i]) for i in [3,4,5]] for info in infos if info]
            species=[elements[(info[-1].strip(b"+-1234"))] for info in infos if info ] 
            #if not metals:
            with gzip.open(f"/home/jrhoernschemeyer/Desktop/data_prep/nometals/ids/{pdb}.gz", "wb") as f:
                    for id, count in zip(ids,counts):
                        f.write(f"{np.char.decode(id).item()} {count}\n".encode())
                    #f.close()#

                
            with gzip.open("/home/jrhoernschemeyer/Desktop/data_prep/nometals/failedprep-reduced.gz","wb") as f:
                for path in badpaths:
                    f.write(f"{path}\n".encode())
                    #f.close()
            torch.save(torch.tensor(coors),f"/home/jrhoernschemeyer/Desktop/data_prep/nometals/structures_data/reduced/coors/{pdb}")
            torch.save(torch.tensor(species),f"/home/jrhoernschemeyer/Desktop/data_prep/nometals/structures_data/reduced/species/{pdb}")



        except:

        
            badpaths.append(pdb)
            print("Fail!", pdb)
            continue

        
            
#generate_data(metals=False, hydrogens=False)
#print("nometalsnoH done", (time.time() - to)/60)
generate_data(metals=False, hydrogens=True)
print("nometals H done", (time.time() - to)/60)
#generate_data(metals=True, hydrogens=True)
print("metals H done", (time.time() - to)/60)
#generate_data(metals=True, hydrogens=False)
print("metals noH done. also, all done", (time.time() - to)/60)