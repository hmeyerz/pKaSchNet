from utils import constants, load_train_dir
from pkparse import parser
#from utils import pack_npz_single#pack_npz_to_zlib
#import random
import glob

import glob, os
from zipfile import ZipFile, ZIP_DEFLATED

SRC_DIR = "../inputs"            # where *.bundle.npz already live
OUT_ZIP = "../pkegnn_dataset.zip"


# --- train dict once on 200 random PDBs before the big loop ----------
#subset = random.sample(list(glob.glob("../pkegnn_inputs_test/*.npz")), k=200)
#train_zstd_dict(subset, DICT)





import os, time
to=time.time()

DATA_DIR = os.getcwd()
nbrs, constants = constants(data_dir=DATA_DIR,n_neighbors=500) #obsolete neighbors
oldpdbs, newpdbs = load_train_dir()
#print(oldpdbs,newpdbs)
if not os.path.exists("../inputs/"):
    os.mkdir("../inputs/")



def run(num_pdbs="all"):
    t0 = time.time()
    if num_pdbs == "all":
        indices = range(len(oldpdbs))
    else:
        indices = num_pdbs

    for i in range(indices):
        try:
            parser(newpdbs[i], oldpdbs[i]).run()

                        # new: convert the freshly-written raw npz → bundle.npz
            #raw = f"../inputs/{newpdbs[i][-7:-3]}.npz"
            #if os.path.exists(raw):
             #   pack_npz_single(raw)         # writes 1abc.bundle.npz next to raw
                #os.remove(raw)               # optional: delete raw to save space



        except Exception as e:
            print(e)
            continue
    
    print("zipping entire db",(time.time() - t0)/60, "mins")

    with ZipFile(OUT_ZIP, "w", compression=ZIP_DEFLATED) as zf:
        for path in glob.glob(os.path.join(SRC_DIR, "*.npz")):
            zf.write(path, arcname=os.path.basename(path))   # 1bjm.bundle.npz → ZIP
    print("Wrote", OUT_ZIP)
    print("data prep time:", (time.time() - t0)/60, "mins", indices, "pdbs")