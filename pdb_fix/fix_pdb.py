
import pdbfixer
from openmm.app import PDBFile
import sys

infile = sys.argv[1]

def fix_pdb(infile):
    fixer=pdbfixer.PDBFixer(infile)
    fixer.findNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()

    fixer.replaceNonstandardResidues()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    with open(f"{infile}-fix.pdb", "w") as output_file:
        PDBFile.writeFile(fixer.topology, fixer.positions, output_file)

fix_pdb(infile)