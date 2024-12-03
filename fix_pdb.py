
import pdbfixer
from openmm.app import PDBFile
import sys

def fix_pdb(infile):
    """gets og residue mapping. then pdbfixes, which adds missing residues in accordance with the sequence, adds hydrogens at pH 7.
    and then removes alt conforms (conformB)"""

    mapy, resis = [],[]
    infile = f"/Users/jessihoernschemeyer/pKaSchNet/PDB/{pdb}.ent"

    def get_xmapping(infile):
        """original experimentally obtained residue and termini 
        resis = ['121BLYS44.106', ...]
        ters = [164]"""
        ters=[]
        with open(infile, "r") as file:
            for line in file:
                if line.startswith("ATOM" or "HETATM"):
                        if line[12:17].strip() == 'CA':
                            L = line.split()[2:-2] 
                            resi = f"{L[3]}{L[2]}{L[1]}{L[4]}" #name, chain, resnum, xcoord


                elif line.startswith('TER'):
                    L = line.split()
                    ters.append(f"{L[4]}{L[3]}{L[2]}")
    
        with open(f"{infile}_resis.txt", "w") as f:
            f.write(pdb)
            f.write(' '.join(ters))
            f.write(' '.join(resis))


    get_xmapping(infile)

    fixer =pdbfixer.PDBFixer(infile)
    fixer.findNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.replaceNonstandardResidues()

    missing_resis = fixer.missingResidues

    if missing_resis: #save unfixed file
        PDBFile.writeFile(fixer.topology, fixer.positions, f"{infile}-unfix")
        with open(f"{infile}_resis.txt", "a") as f:
                f.write(str(missing_resis))

    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, f"{infile}-fix")

    def get_ymapping1_removealts(infile):
        """fixes pdbfixer. gets second side of map"""
        ters=[]
        infile = f"{infile}-fix"
        print(infile, type(infile))
        with open(infile, "r") as file:
            lines = file.readlines()

        with open(infile, "w") as file:
            for line in lines:
                if line.startswith(('ATOM', 'HETATM')): 
                    res_num = line[22:26].strip()
                    if not res_num.isdigit() and res_num[-1] != 'A':
                        return
                    else:
                        file.write(line)
                        mapy.append(f"{res_num}{line[21]}{line[17:20]}")
                    
                    #resname, chain = line[17:20], line[21] #make ymap1
                   

                elif line.startswith(("TER")):
                    file.write(line)
                    L=line.split()
                    ters.append(f"{L[4]}{L[3]}{L[2]}")

                else:
                    file.write(line)

                    
            with open(f"{infile}_resis.txt", "a") as f:
                f.write(' '.join(set(mapy)))

        
    get_ymapping1_removealts(infile)

pdb=sys.argv[1]
fix_pdb(f"/Users/jessihoernschemeyer/pKaSchNet/PDB/{pdb}.ent")