import gzip
import glob
import re
import numpy as np
import torch
import time
to=time.time()

#idfiles=glob.glob("/home/jrhoernschemeyer/Desktop/data_prep/structures/ids/*")

#ids=glob.glob("./targets/ids/*")
def align():
    idfiles=glob.glob("/home/jrhoernschemeyer/Desktop/data_prep/structures/ids/*")
    for file in idfiles[1:1500]:
        try:
            pdb=file[-7:-3]
            #print(pdb)
            #print(file)
            with gzip.open(file,"r") as f:
                lines=f.readlines()
                #print(lines)
            mmyresis = sorted(lines, key=lambda s: (re.match(r'(\d+)([A-Za-z])', s.decode()).group(2).lower(), int(re.match(r'(\d+)([A-Za-z])', s.decode()).group(1))))
                
            with gzip.open(f"/home/jrhoernschemeyer/Desktop/data_prep/targets/ids/{pdb}.gz","r") as f:
                lines=f.readlines()
                #print(lines)
                f.close()

            ppypka = sorted(lines, key=lambda s: (re.match(r'(\d+)([A-Za-z])', s.decode()).group(2).lower(), int(re.match(r'(\d+)([A-Za-z])', s.decode()).group(1))))


            myresis = np.array([inf.split()[0] for inf in mmyresis])
            pypka = np.array([inf.split()[0] for inf in ppypka])



            # Pre-compile regex for efficiency.
            pattern = re.compile(r'(\d+)([A-Za-z])')

            # Build a list with (original_index, letter, number, line)
            indexed_lines = []
            for idx, line in enumerate(lines):
                decoded = line.decode().strip()  # Decode once per line.
                m = pattern.match(decoded)
                if m:
                    num = int(m.group(1))
                    letter = m.group(2).lower()
                    indexed_lines.append((idx, letter, num, line))
                else:
                    indexed_lines.append((idx, '', 0, line))

            # Sort by letter then by number.
            indexed_lines.sort(key=lambda x: (x[1], x[2]))

            # Build the mapping and extract sorted lines.
            old_to_new = {orig_idx: new_idx for new_idx, (orig_idx, letter, num, line) in enumerate(indexed_lines)}
            #sorted_lines = [line for (orig_idx, letter, num, line) in indexed_lines]







            
            with gzip.open(f"/home/jrhoernschemeyer/Desktop/data_prep/targets/{pdb}.gz","r") as f:
                tlines=f.readlines()
                #print(tlines)
            

            pkas={id:tlines[old_to_new[i]].strip() for i,id in enumerate(pypka)}#range(len(pypka))]



        #def align(pks,myresis): #pks is pypka
        #pkids = sorted(pks, key=lambda s: (s[-1], int(re.match(r'\d+', s.decode()).group())))

            pypkaresis =[int(res[:-1]) for res in pypka]
            myresis=[int(res[:-1]) for res in myresis]
            labels=[]
            j_0=0
            ids=[]
            counter,counter2=0,1
            for pk in pypkaresis:
                my=myresis[0]
                py=pk
                #print("origin!",my,py)
                if py > j_0:
                    j_0 = py

                    if my == py:
                        #print("success easy")
                        id=np.char.add(str(py), chr(64+counter2))
                        #print(id)
                        id=np.char.encode(np.char.add(str(py),chr(64+counter2)))

                        labels.append(np.float32(pkas[id.item()]))
                        ids.append(id)
                        del myresis[0]
                    
                    elif my - counter > py:
                        #print("pypka has one we dont have, or the resname was bad in the pdb file (wont happen for fixed pdbs)", my, py)
                        #added resi'
                        #print(my,py)
                        #'means that pypka has one that we dont and should never happen'

                        counter+=1

                    elif my - counter < py:
                        
                        #print("processing a resi i have that they maybe didnt do")
                        
                        #a resi they didnt do e.g. sulfur cys
                        del myresis[0]
                        my = myresis[0]
                        if my == py:
                            
                            id=np.char.encode(np.char.add(str(py),chr(64+counter2)))
                            #print(id)
                            #print("they did do-success")
                            labels.append(np.float32(pkas[id.item()]))
                            
                            ids.append(id)
                            #ids.append(my)
                            #labels[n] = pks[str(py) + chr(64+counter2)]
                            del myresis[0]
                        else:
                            #print(my,py)
                            del myresis[0]


                        continue

                else: #chain break in my data
                    counter2+=1
                    #print("chain")
                    j_0 = py

                
            with gzip.open(f"./labels/ids/{pdb}.gz", "wb") as f:
                for id in mmyresis:
                    #print(id)
                    f.write(id)

            torch.save(torch.tensor(labels), f"./labels/{pdb}")
            #print(pypka)
        except:
            print(pdb, "fail")
            

align()
print("1500", (to - time.time())/60)