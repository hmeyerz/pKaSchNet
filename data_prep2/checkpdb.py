
import sys
import os
pdb_path = sys.argv[1]

def check_pdb(pdb_path):
    salts = []
    with open(pdb_path, 'r+') as file: 
            
            for line in file.readlines():

                if not line.startswith("ATOM"):

                    if line.startswith("HETATM"):

                        if line.split()[1:][-1] in ["ZN", "S", "FE", "MG", "MN", "CO", "NI", "CU"]: #exclude hetero sulfur
                            os.remove(pdb_path)
                            break
                        file.write(line)
                    
                    else:
                        file.write(line)
                                                    
                else: #ATOM

                    if line.split()[1:][-1] in ["ZN", "FE", "MG", "MN", "CO", "NI", "CU"]:
                        os.remove(pdb_path)
                        break

                    else:
                        file.write(line)

check_pdb(pdb_path)