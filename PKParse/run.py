from utils import constants, load_train_dir
from pkparse import parser

import glob, os
from zipfile import ZipFile, ZIP_DEFLATED

SRC_DIR = "../inputs"            # where *.bundle.npz already live
OUT_ZIP = "../pkegnn_dataset.zip"


# --- train dict once on 200 random PDBs before the big loop ----------
#subset = random.sample(list(glob.glob("../pkegnn_inputs_test/*.npz")), k=200)
#train_zstd_dict(subset, DICT)





import os, time
to=time.time()

#DATA_DIR = os.getcwd()
constants = constants() #obsolete neighbors
oldpdbs, newpdbs = load_train_dir()
#print(oldpdbs,newpdbs)
if not os.path.exists("../inputs/"):
    os.mkdir("../inputs/")


print(newpdbs[0],oldpdbs[0])
def run(num_pdbs="all"):
    """This takes the intersection of the fixed and original gzipped pdb files which are on disk and wraps them in a try-except clause."""
    t0 = time.time()
    if num_pdbs == "all":
        indices = range(len(oldpdbs))
    else:
        indices = num_pdbs

    for i in range(indices):
        try:
            parser(newpdbs[i], oldpdbs[i]).run()

        except Exception as e:
            print(e)
            continue
    
    print("zipping entire db",(time.time() - t0)/60, "mins")

    with ZipFile(OUT_ZIP, "w", compression=ZIP_DEFLATED) as zf:
        for path in glob.glob(os.path.join(SRC_DIR, "*.npz")):
            zf.write(path, arcname=os.path.basename(path))   # 1bjm.bundle.npz → ZIP
    print("Wrote", OUT_ZIP)
    print("data prep time:", (time.time() - t0)/60, "mins", indices, "pdbs")