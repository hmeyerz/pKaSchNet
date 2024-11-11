
import pdbfixer
from openmm.app import PDBFile
import sys

infile = sys.argv[1]

def fix_pdb(infile):
    "pdbfixes, adds hydrogens and then removes alt conforms"
    fixer=pdbfixer.PDBFixer(infile)
    fixer.findNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()

    fixer.replaceNonstandardResidues()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    file_path = f"{infile}-fix.pdb"

    with open(file_path, "r") as infile:
        lines = infile.readlines()

    # Write the fixed content to a new file
    with open(file_path, "w") as file:
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):  # Process only atom or heteroatom lines
                res_num = line[22:27].strip()
                print("HI",res_num)
                if not res_num.isdigit() and res_num[-1] != 'A':
                    print("hello")
                    continue
                else:
                    file.write(line)
            else:
                file.write(line)
    

fix_pdb(infile)