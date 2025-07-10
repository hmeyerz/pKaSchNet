from sklearn.neighbors import NearestNeighbors
import numpy as np
import glob

# ---------------------------------------------------------------------------
#  Fast converter:  <pdb>.npz  ➜  .coords_elms.zlib + 3 side-car .npy files
# ---------------------------------------------------------------------------
import zlib, pathlib
from typing import Union


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
# Complete periodic table (IUPAC symbols) + deuterium, as bytes → atomic number
cofactors = {
    b"D":   1,   b"H":   1,
    b"He":  2,   b"Li":  3,   b"Be":  4,   b"B":   5,   b"C":   6,   b"N":   7,
    b"O":   8,   b"F":   9,   b"Ne": 10,   b"Na": 11,   b"Mg": 12,   b"Al": 13,
    b"Si": 14,   b"P":  15,   b"S":  16,   b"Cl": 17,   b"Ar": 18,   b"K":  19,
    b"Ca": 20,   b"Sc": 21,   b"Ti": 22,   b"V":  23,   b"Cr": 24,   b"Mn": 25,
    b"Fe": 26,   b"Co": 27,   b"Ni": 28,   b"Cu": 29,   b"Zn": 30,   b"Ga": 31,
    b"Ge": 32,   b"As": 33,   b"Se": 34,   b"Br": 35,   b"Kr": 36,   b"Rb": 37,
    b"Sr": 38,   b"Y":  39,   b"Zr": 40,   b"Nb": 41,   b"Mo": 42,   b"Tc": 43,
    b"Ru": 44,   b"Rh": 45,   b"Pd": 46,   b"Ag": 47,   b"Cd": 48,   b"In": 49,
    b"Sn": 50,   b"Sb": 51,   b"Te": 52,   b"I":  53,   b"Xe": 54,   b"Cs": 55,
    b"Ba": 56,   b"La": 57,   b"Ce": 58,   b"Pr": 59,   b"Nd": 60,   b"Pm": 61,
    b"Sm": 62,   b"Eu": 63,   b"Gd": 64,   b"Tb": 65,   b"Dy": 66,   b"Ho": 67,
    b"Er": 68,   b"Tm": 69,   b"Yb": 70,   b"Lu": 71,   b"Hf": 72,   b"Ta": 73,
    b"W":  74,   b"Re": 75,   b"Os": 76,   b"Ir": 77,   b"Pt": 78,   b"Au": 79,
    b"Hg": 80,   b"Tl": 81,   b"Pb": 82,   b"Bi": 83,   b"Po": 84,   b"At": 85,
    b"Rn": 86,   b"Fr": 87,   b"Ra": 88,   b"Ac": 89,   b"Th": 90,   b"Pa": 91,
    b"U":  92,   b"Np": 93,   b"Pu": 94,   b"Am": 95,   b"Cm": 96,   b"Bk": 97,
    b"Cf": 98,   b"Es": 99,   b"Fm":100,   b"Md":101,   b"No":102,   b"Lr":103,
    b"Rf":104,   b"Db":105,   b"Sg":106,   b"Bh":107,   b"Hs":108,   b"Mt":109,
    b"Ds":110,   b"Rg":111,   b"Cn":112,   b"Nh":113,   b"Fl":114,   b"Mc":115,
    b"Lv":116,   b"Ts":117,   b"Og":118
}

elements = {b"C":   6,   b"N":   7,
    b"O":   8,  b"S":  16}

# ---------------------------------------------------------------------------
#  utils.pack_npz_to_zlib  –  now supports *flat* or *object* layouts
# ---------------------------------------------------------------------------
#import numpy as np, zlib, pathlib
#from typing import Union

# ---------------------------------------------------------------------------
#  (A)  Train a Zstd dictionary on a handful of raw parser *.npz files
#       Run _once_ before building the DB.
# ---------------------------------------------------------------------------
import random, zstandard as zstd, numpy as np, pathlib, io, os, json

# ---------------------------------------------------------------------------
#  Train a 32 KiB Zstandard dictionary from sample payloads  (deterministic)
# ---------------------------------------------------------------------------
import zstandard as zstd, io, pathlib, numpy as np

def train_zstd_dict(sample_npzs, dict_path="pkegnn.dict", precision=0.001):
    """
    sample_npzs : list[str]  raw <pdb>.npz files produced by parser.run()
    dict_path   : output filename (32 KiB)
    """
    # build uncompressed payloads in-memory
    samples = [ _build_uncompressed_bundle(p, precision) for p in sample_npzs ]
    # official trainer (since zstandard-0.15)
    zstd_dict = zstd.train_dictionary(32_768, samples)
    pathlib.Path(dict_path).write_bytes(zstd_dict.as_bytes())
    print("[train_zstd_dict]  wrote", dict_path,
          "from", len(samples), "samples")

# ---------------------------------------------------------------------------
#  (B)  Append one raw <pdb>.npz to the single *.zst file + update index
# ---------------------------------------------------------------------------
def append_to_zstd_db(db_path, idx_path, dict_path, raw_npz, precision=0.001):
    """
    db_path  : pkegnn.zst          (created if missing)
    idx_path : pkegnn.idx.json     (created/updated)
    dict_path: pkegnn.dict         (created by train_zstd_dict)
    raw_npz  : <pdb>.npz from parser.run()
    """
    pdb_id   = pathlib.Path(raw_npz).stem[:4]
    payload  = _build_uncompressed_bundle(raw_npz, precision)   # bytes

    cdict    = zstd.ZstdCompressionDict(pathlib.Path(dict_path).read_bytes())
    zcomp    = zstd.ZstdCompressor(level=18, dict=cdict)        # max ratio
    frame    = zcomp.compress(payload)

    # --- append to data file ---
    with open(db_path, "ab") as fh:
        offset = fh.tell()
        fh.write(frame)

    # --- update / create index ---
    idx = {}
    if pathlib.Path(idx_path).exists():
        idx = json.load(open(idx_path))
    idx[pdb_id] = (offset, len(frame))
    json.dump(idx, open(idx_path, "w"))
    print(f"[append_to_zstd_db]  {pdb_id}  frame={len(frame)//1024} KB")

# ---------------------------------------------------------------------------
#  (C)  Internal helper: build the *un-compressed* binary bundle
#       (coords Δ-encoded int16  |  uint8 elements  |  seglen, anchor, ids, pks, prec)
# ---------------------------------------------------------------------------
def _build_uncompressed_bundle(npz_path, precision):
    dat    = np.load(str(npz_path), allow_pickle=True)
    z_flat = dat["z"].astype(np.uint8)
    xyz    = dat["pos"].astype(np.float32)
    anchor = dat["anchor"];  ids = dat["ids"];  pks = dat["pks"]
    seglen = (dat["segment_lengths"] if "segment_lengths" in dat.files
              else np.diff(np.append(anchor, len(z_flat))).astype(np.uint16))

    # Δ-encode
    q   = np.round(xyz/precision).astype(np.int32)
    d16 = np.empty_like(q, dtype=np.int16)
    d16[0]  = q[0].astype(np.int16)
    d16[1:] = (q[1:] - q[:-1]).astype(np.int16)

    buf = io.BytesIO()
    buf.write(d16.tobytes());  buf.write(z_flat.tobytes())
    np.save(buf, seglen, allow_pickle=False)
    np.save(buf, anchor, allow_pickle=False)
    np.save(buf, ids,    allow_pickle=False)
    np.save(buf, pks,    allow_pickle=False)
    np.save(buf, np.float32(precision), allow_pickle=False)
    return buf.getvalue()


def _flatten_legacy(zs_obj, coords_obj):
    """old object-array layout → flat arrays + segment lengths list"""
    seg_lens    = [len(z) for z in zs_obj]
    coords_flat = np.vstack(coords_obj)                # (ΣN,3)
    elms_flat   = np.concatenate(zs_obj).astype(np.uint8, copy=False)
    return elms_flat, coords_flat, seg_lens

def _flatten_compact(elms_flat, coords_flat, anchor):
    """new compact layout already flat – just derive seg_lens"""
    anchor = anchor.astype(np.int64)
    seg_lens = np.diff(np.append(anchor, len(elms_flat))).tolist()
    return elms_flat, coords_flat, seg_lens

def pack_npz_to_zlib(npz_path: Union[str, pathlib.Path],
                     out_prefix: str = None,
                     precision: float = 0.001,
                     compression_lvl: int = 9) -> None:
    """
    Convert parser output (.npz) to the 4-file compressed bundle.
    Handles both the *legacy* object arrays and the new *compact* layout.
    """
    npz_path = pathlib.Path(npz_path)
    if not npz_path.exists():
        print("[pack_npz_to_zlib]  SKIP (missing)", npz_path.name)
        return
    if out_prefix is None:
        out_prefix = str(npz_path.with_suffix(""))

    dat = np.load(npz_path, allow_pickle=True)

    # -------- detect layout --------
    if 'anchor' in dat.files:        # new compact form
        elms_flat, coords_flat, seg_lens = _flatten_compact(
            dat['z'], dat['pos'], dat['anchor'])
        seg_ids = dat['ids']
        pks     = dat['pks']
    else:                            # legacy object arrays
        elms_flat, coords_flat, seg_lens = _flatten_legacy(
            dat['z'], dat['pos'])
        seg_ids = dat['ids']
        pks     = dat['pks']

    # ---------- quantise + Δ-encode ----------
    q = np.round(coords_flat/precision).astype(np.int32)
    deltas = np.empty_like(q, dtype=np.int16)
    deltas[0]  = q[0].astype(np.int16)
    deltas[1:] = (q[1:] - q[:-1]).astype(np.int16)

    blob = deltas.tobytes() + elms_flat.tobytes()
    compressed = zlib.compress(blob, level=compression_lvl)

    with open(f"{out_prefix}.coords_elms.zlib","wb") as fh:
        fh.write(compressed)
    np.save(f"{out_prefix}.segment_ids.npy",      seg_ids)
    np.save(f"{out_prefix}.segment_lengths.npy",  np.array(seg_lens, dtype=np.uint16))
    np.save(f"{out_prefix}.per_segment_scalars.npy", pks.astype(np.float32, copy=False))
    np.save(f"{out_prefix}.anchor.npy",           dat['anchor'])        # NEW

    print(f"[pack_npz_to_zlib]  {npz_path.name}: {len(seg_lens)} segments, "
          f"{coords_flat.shape[0]} atoms → {len(compressed)} bytes")

def constants(data_dir, n_neighbors):
    """use because pkparser is per pdb."""
    ligands = np.load(data_dir + "/data/ligands.npz")["data"]
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

def load_train_dir():
    """returns the strings for the intersection of the old pdbs (rcsb) and the new pdbs (modeled)"""
    pdbs=np.char.array(glob.glob("../../rcsb/*.gz"))
    fixed_pdbs=np.char.array(glob.glob("../../fixed/*.gz"))
    _, og, mdled = np.intersect1d(np.char.array([f[-11:-7] for f in pdbs]),np.char.array([f[-7:-3] for f in fixed_pdbs]), return_indices=True)
    return pdbs[og], fixed_pdbs[mdled]


def pack_npz_to_zlib_old(npz_path: Union[str, pathlib.Path],
                     out_prefix: str = None,
                     precision: float = 0.001,
                     compression_lvl: int = 9) -> None:
    """
    Turn a parser-generated *.npz into the compact four-file bundle expected by
    CompressedCoordsDataset.  Safe on Python 3.6.

    Parameters
    ----------
    npz_path : str or Path
        Path to the .npz produced by parser.run().
    out_prefix : str, optional
        Prefix for the output files (defaults to npz_path without '.npz').
    precision : float, default 0.001
        Å grid size for integer quantisation (smaller = larger files).
    compression_lvl : int, default 9
        zlib compression level (0–9).
    """
    npz_path = pathlib.Path(npz_path)
    if out_prefix is None:
        out_prefix = str(npz_path.with_suffix(""))

    # ---------- load ----------
    dat     = np.load(npz_path, allow_pickle=True)
    zs      = dat["z"]      # object array of 1-D uint8 arrays
    coords  = dat["pos"]    # object array of (Ni,3) float32
    seg_ids = dat["ids"]    # ('S*',) byte array
    pks     = dat["pks"]    # per-segment float32

    n_seg = len(zs)
    if not (n_seg == len(coords) == len(seg_ids) == len(pks)):
        raise ValueError("segment count mismatch inside " + str(npz_path))

    # ---------- flatten ----------
    seg_lens    = [len(z) for z in zs]
    tot_atoms  = sum(seg_lens)
    coords_flat = np.empty((tot_atoms, 3),  dtype=np.float32)
    elms_flat   = np.empty(tot_atoms,       dtype=np.uint8)

    cursor = 0
    for z_arr, xyz in zip(zs, coords):
        n = len(z_arr)
        coords_flat[cursor:cursor+n] = xyz
        elms_flat[cursor:cursor+n]   = z_arr
        cursor += n
    # ---------- quantise & delta-encode ----------
    q_int32 = np.round(coords_flat/precision).astype(np.int32)
    deltas  = np.empty_like(q_int32, dtype=np.int16)
    deltas[0]  = q_int32[0].astype(np.int16)
    deltas[1:] = (q_int32[1:] - q_int32[:-1]).astype(np.int16)

    # ---------- serialise ----------
    blob       = deltas.tobytes() + elms_flat.tobytes()
    compressed = zlib.compress(blob, level=compression_lvl)

    # ---------- write ----------
    with open(f"{out_prefix}.coords_elms.zlib", "wb") as fh:
        fh.write(compressed)

    np.save(f"{out_prefix}.segment_ids.npy",      seg_ids)
    np.save(f"{out_prefix}.segment_lengths.npy",  np.array(seg_lens, dtype=np.uint16))
    np.save(f"{out_prefix}.per_segment_scalars.npy",
            pks.astype(np.float32, copy=False))

    print(f"[pack_npz_to_zlib]  {npz_path.name}: "
          f"{n_seg} segments, {coords_flat.shape[0]} atoms → "
          f"{len(compressed)} bytes")

# ---------------------------------------------------------------------------
#  utils.pack_npz_single  –  turn the raw parser .npz into ONE tiny bundle.npz
# ---------------------------------------------------------------------------
import numpy as np, zlib, pathlib

def pack_npz_single(npz_in, precision=0.001, lvl=9):
    """
    Convert  <pdb>.npz  created by parser.run()  into  <pdb>.bundle.npz.
    Keeps delta-compressed coords+elements AND the critical 'anchor' array.
    """
    npz_in   = pathlib.Path(npz_in)
    out_path = str(npz_in.with_suffix(".npz"))
    dat      = np.load(npz_in, allow_pickle=True)

    z_flat = dat["z"].astype(np.uint8)
    xyz    = dat["pos"].astype(np.float32)
    anchor = dat["anchor"]
    ids    = dat["ids"];  pks = dat["pks"]

    # segment_lengths: reuse if present, else derive from anchor
    if "segment_lengths" in dat.files:
        seglen = dat["segment_lengths"]
    else:
        seglen = np.diff(np.append(anchor, len(z_flat))).astype(np.uint16)

    # ---- delta-encode & compress coords + elements ----
    q   = np.round(xyz/precision).astype(np.int32)
    d16 = np.empty_like(q, dtype=np.int16)
    d16[0] = q[0].astype(np.int16)
    d16[1:] = (q[1:] - q[:-1]).astype(np.int16)
    blob = d16.tobytes() + z_flat.tobytes()
    blob_c = zlib.compress(blob, level=lvl)

    # ---- write single bundled npz ----
    np.savez_compressed(
        out_path,
        blob   = np.frombuffer(blob_c, dtype=np.uint8),
        seglen = seglen,
        anchor = anchor,
        ids    = ids,
        pks    = pks,
        prec   = np.float32(precision)
    )
    return out_path
