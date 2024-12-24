newlines, Z=[],[]
with open("/Users/jessihoernschemeyer/pKaSchNet/PDB/107l.ent.pdb.prot", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("A"):
            L=line.split()
            e = L[2]
            
            newlines.append("  ".join([e[0], L[5],L[6],L[7],'\n']))

with open("/Users/jessihoernschemeyer/pKaSchNet/PDB/1107l.xyz", "w") as f:
    f.write(f"{len(newlines)}\n")
    f.write(f"\n")
    f.writelines(newlines)