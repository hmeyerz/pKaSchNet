import polars as pl
from collections import OrderedDict
import gzip

code={"L":0,
      "C":1,
      "A":2,
      "G":3,
      "T":4,
      "H":5}


# Read the full CSV, preserving all columns
df = pl.read_csv("/home/pbuser/Desktop/Jessi/data_prep/pkas.csv", separator=";")

# Get unique idcode keys from the DataFrame and sort them
unique_keys = df["idcode"].unique().to_list()
unique_keys.sort()

all_targets = {}


# Filter the full DataFrame for each pdb
for key in unique_keys:
    targets = OrderedDict()
    for row in df.filter(pl.col("idcode") == key).drop("idcode").iter_rows():
        resname=row[1]
        if resname[:2] not in ("NT", "CT"):
            targets[f"{row[2]}{row[0]}{code[resname[0]]}"] = row[3]
    all_targets[key] = targets

#save
with gzip.open("/home/pbuser/Desktop/Jessi/data_prep/targets.gz', 'wb") as f:
    for k, v in all_targets.items():
        f.write(f"{k}\n".encode())
        for k2,v2 in v.items():
            f.write(f"{k2} {v2}\n".encode())
        f.write(f"  \n".encode())
