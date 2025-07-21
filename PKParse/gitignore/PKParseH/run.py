##########NO HYDROGENS, FULL PERIODIC* (or deteriums) PREP##########


########### code by Jessi Hoernschemeyer made together with PhD candidate Jesse Jones and Dr. Maria Andrea Mroginski, 
########### supported by Dr. Cecilia Clementi TU Biomodeling, and ChatGPT
########## © Summer 2023  - Summer 2025, Berlin ####################

###*Besides those which are confirmed not in any PDBS in the RCSB, and hydrogens
#   and momentarily, besides ionic nitrogen/oxygen

from utils import constants, load_train_dir
from pkparse import parser

import glob, os
from zipfile import ZipFile, ZIP_DEFLATED
import gzip
import numpy as np
"""TODO: explain the maps 

array([[b'1A', b'1A'],
       [b'2A2', b'2A2'],
       [b'3A', b'3A'],
       [b'4A', b'4A'],
       [b'5A1', b'5A1'],
       [b'6A5', b'6A5'],
       
       How only titratable have a val after chain."""

import os, time

#data directories
rcsb_paths     =np.char.array(glob.glob("../../data/pdbs/rcsb/*.gz"))
modeled_paths   =np.char.array(glob.glob("../../data/pdbs/fixed/*.gz"))
oldpdbs, newpdbs = load_train_dir(modeled_paths,rcsb_paths)

OUT_DIR = "../Hinputs"          
OUT_ZIP = "../pkegnn_dataset.zip"
MAP_DIR = "../res_maps" #TODO dont make maps after its been done before and e.g. wanna develop cys bridges or add hydrogens or whatever
FAIL_GZ = "../badparse.gz" #log



# --- train dict once on 200 random PDBs before the big loop ----------
#subset = random.sample(list(glob.glob("../pkegnn_inputs_test/*.npz")), k=200)
#train_zstd_dict(subset, DICT)

constants = constants() 

#print(oldpdbs,newpdbs)
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
with gzip.open(FAIL_GZ, "wb") as f:
    f.close()
if not os.path.exists(MAP_DIR):
    os.mkdir(MAP_DIR)


print("accessing source directories working if files printed:",newpdbs[0],oldpdbs[0])

def run(selection="all"):
    """This takes the intersection of the fixed and original gzipped pdb files which are on disk and wraps them in a try-except clause."""
    t0 = time.time()

    newpdbs, oldpdbs = load_train_dir(modeled_paths,rcsb_paths)
    print(len(newpdbs))

    if selection == "all":
        indices = len(oldpdbs)
    elif type(selection) == int:
        indices = selection
    else: 
        newpdbs=np.char.array([f"../../data/pdbs/rcsb/{selection}.gz"])
        oldpdbs=np.char.array([f"../../data/pdbs/rcsb/{selection}.pdb.gz"])

    for i in range(0,indices):
        try:
            parser(newpdbs[i], oldpdbs[i]).run()

        except Exception as e:
            print(newpdbs[i])
            pdb=newpdbs[i].split("gz")[0][-5:-1].encode()
            print("error in", pdb, ":", e)
            with gzip.open("../badparse.gz", "ab") as f:
                f.write(pdb)
                f.write(b"\n")
                continue

    print("zipping entire db",(time.time() - t0)/60, "mins")

    with ZipFile(OUT_ZIP, "w", compression=ZIP_DEFLATED) as zf:
        for path in glob.glob(os.path.join(OUT_DIR, "*.npz")):
            zf.write(path, arcname=os.path.basename(path)) 
    print("Wrote", OUT_ZIP)
    print("data prep time:", (time.time() - t0)/60, "mins", indices, "pdbs")