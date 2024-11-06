#TODO delete print statements
import sys
import os

pdb_path = sys.argv[1]

def check_pdb(pdb_path):
    with open(pdb_path, 'r') as file: 
            
        for line in file:
            if line.startswith("ATOM"):
                if line.split()[1:][-1] in ["ZN", "FE", "MG", "MN", "CO", "NI", "CU"]:
                    os.remove(pdb_path)
                    break

            elif line.startswith("HETATM"):
                    if line.split()[1:][-1] in ["ZN", "S", "FE", "MG", "MN", "CO", "NI", "CU"]: #exclude hetero sulfur
                        os.remove(pdb_path)
                        break


check_pdb(pdb_path)