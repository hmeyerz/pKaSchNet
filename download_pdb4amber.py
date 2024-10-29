import dask.dataframe as dd
from ordered_set import OrderedSet
from Bio.PDB import *
import os
import subprocess
local_folder="/Users/jessihoernschemeyer/pKaSchNet"

def read_database(path):
    """csv --> dask df"""
    #make the dask data frame from the PYPKA csv
    dk=dd.read_csv(path, delimiter=';', na_filter=False, dtype={'idcode':'category', 
                                                                    'residue_number':'uint8',
                                                                    'pk': 'float32',
                                                                    'residue_name':'category',
                                                                    'chain': 'category'
                                                                    })
                                                            
    dk=dk.rename(columns={'idcode': 'PDB ID', 'residue_number': 'Res ID', 'residue_name': 'Res Name', 'residue_number': 'Res ID', 'pk': 'pKa', 'chain' : 'Chain'}) #rename columns to match df from pkad 
    dk=dk.sort_values(['PDB ID', 'Res ID'], ascending=[True, True]) 

    return dk.compute().reset_index()

def download_pdbs(dask_df):
    for i in range(len(dask_df)): 
        pdbname=pdbs[i]
        PDBList().retrieve_pdb_file(str.lower(pdbname),obsolete=False, pdir='PDB',file_format = 'pdb')
        inf, outf = f"{local_folder}/PDB/pdb{pdbname}.ent", f"{local_folder}/pdb4amber/{pdbname}.pdb"
        subprocess.run(["pdb4amber", "-i", inf, "-o", outf, "--reduce"])
        #pdb4amber -i {inf} -o {outf} --reduce
        os.remove(inf) #delete first pdb
        os.remove(f"{local_folder}/pdb4amber/{pdbname}_sslink")
        os.remove(f"{local_folder}/pdb4amber/{pdbname}_nonprot.pdb")


dask_df = read_database(f"{local_folder}/pkas.csv")
pdbs = list(OrderedSet(list(dask_df["PDB ID"])))
download_pdbs(dask_df)
