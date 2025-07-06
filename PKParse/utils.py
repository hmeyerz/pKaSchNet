from sklearn.neighbors import NearestNeighbors
import numpy as np
import glob


fullcode = {
    b"HIS":b"0", b"ASP":b"1", b"LYS":b"2",
    b"TYR":b"3", b"GLU":b"4", b"CYS":b"5"
}
code = {
    b"H":b"0", b"A":b"1", b"L":b"2",
    b"T":b"3", b"G":b"4", b"C":b"5"
}
amber = {
    0:b"OE2", 1:b"OD2", 2:b"SG",
    3:b"OH",  4:b"NZ",  5:b"ND1"
}

three2one = {
    b"ALA":"A", b"ARG":"R", b"ASN":"N", b"ASP":"D", b"CYS":"C",
    b"GLU":"E", b"GLN":"Q", b"GLY":"G", b"HIS":"H", b"ILE":"I",
    b"LEU":"L", b"LYS":"K", b"MET":"M", b"PHE":"F", b"PRO":"P",
    b"SER":"S", b"THR":"T", b"TRP":"W", b"TYR":"Y", b"VAL":"V"
}
elements = {
    b"D":1, b"H":1, b"LI":3, b"C":6, b"N":7, b"O":8,
    b"F":9, b"NA":11, b"MG":12, b"P":15, b"S":16, b"CL":17,
    b"K":19, b"CA":20, b"V":23, b"CR": 24, b"MN":25, b"FE":26, b"CO":27, b"NI":28,
    b"CU":29, b"ZN":30, b"SE":34, b"MO":42, b"RU": 44, b"SN":50,
    b"I":53, b"CS":55, b"W":74, b"PT":78, b"BI": 83
}

cofactors={b"F":9,
    b"NA": 11, b"MG": 12,  b"P": 15, b"S": 16, b"CL": 17, b"K": 19,
    b"CA": 20,  b"V":23, b"CR": 24, b"MN": 25, b"FE": 26, b"CO":27, b"NI":28,
    b"CU": 29, b"ZN": 30,
    b"SE":34,
    b"MO":42,
    b"I":53, 
    b"W":74,
    b"BI": 83}


def constants(data_dir, n_neighbors):
    """use because pkparser is per pdb."""
    ligands = np.load(data_dir + "/ligands.npz")["data"]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
    cnbrs = NearestNeighbors(algorithm="brute") #TODO
    byte_constants = {"titratables_full": fullcode,
                 "titratables_short": code,
                 "aa_code": three2one,
                 "elements": elements,
                 "cofactors": cofactors,
                 "ligands": ligands,
                 "disulfide_nn": cnbrs,
                 "amber_sites": amber
    }
    return nbrs, byte_constants

def load_train_dir(data_dir):
    """returns the strings for the intersection of the old pdbs (rcsb) and the new pdbs (modeled)"""
    pdbs=np.char.array(glob.glob(data_dir + "/rcsb/*.gz"))
    fixed_pdbs=np.char.array(glob.glob(data_dir + "/fixed/*.gz"))
    pdbs, og, mdled = np.intersect1d(np.char.array([f[-11:-7] for f in pdbs]),pdbs2=np.char.array([f[-7:-3] for f in fixed_pdbs]), return_indices=True)
    return pdbs[og], fixed_pdbs[mdled]

