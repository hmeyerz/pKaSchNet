from utils import constants, load_train_dir
from pkparse2 import parser
import os
import time
to=time.time()

DATA_DIR = os.getcwd()
nbrs, constants = constants(data_dir=DATA_DIR,n_neighbors=50)
oldpdbs, newpdbs = load_train_dir(DATA_DIR)
#print(oldpdbs,newpdbs)
if not os.path.exists("../pkegnn_inputs_mini/"):
    os.mkdir("../pkegnn_inputs_mini/")

def run(num_pdbs):
    if num_pdbs=="all":
        n=range(len(oldpdbs))
    else:
        n=num_pdbs
    for i in range(20,n):
        try:
            parser(newpdbs[i],oldpdbs[i]).run()
        

        #break
        except Exception as e:
            print(e)
            #break
            continue
    print("data prep time:", (time.time() - to)/60, "mins", num_pdbs, "pdbs")

