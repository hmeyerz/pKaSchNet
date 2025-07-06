from utils import constants, load_train_dir
from pkparse import parser
import os

DATA_DIR = os.getcwd()
nbrs, constants = constants(data_dir=DATA_DIR,n_neighbors=50)
oldpdbs, newpdbs = load_train_dir(DATA_DIR)

def run():
    for i in len(oldpdbs):
        try:
            parser(newpdbs[i],oldpdbs[i], constants).run()
        except:
            print("exception #todoraise")
            continue

