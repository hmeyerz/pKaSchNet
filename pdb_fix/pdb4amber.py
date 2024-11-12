import os
from itertools import chain
import argparse
import parmed


leap_resis = (
    "ALA", "ARG", "ASN", "CYS",
    "GLN", "GLY", "HID", "HIE", "HIP", "HYP",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "HOH"
)


string_types = str

__version__ = '1.6.dev'

class AmberPDBFixer(object):
    ''' Base class (?) for handling pdb4amber (try to mimic
    original code)

    Parameters
    ----------
    parm : str or parmed.Structure or None, default None
    '''

    def __init__(self, parm=None):
        self.parm = parm


    

    def rename_residues_forleap(self): #HERE

        for residue in self.parm.residues:
            name = residue.name
            if name == 'ASP':
                residue.name = 'ASH'
            elif name == 'GLU':
                residue.name = 'GLH'
            elif name == 'HIS':
                residue.name = 'HIP'
            elif name not in leap_resis:
                residue.name = 'DUM'
            else:
                pass

        return self

    
    def write_pdb(self, filename):
        '''

        Parameters
        ----------
        filename : str or file object
        '''
        self.parm.write_pdb(filename)

    def _write_renum(self, basename):
        ''' write original and renumbered residue index
        '''

        with open(basename + '_renum.txt', 'w') as fh:
            for residue in self.parm.residues:
                fh.write("%3s %1s %5s    %3s %5s\n" %
                   (residue.name, residue.chain, residue.number, residue.name,
                    residue.idx + 1))



def run(
        arg_pdbout,
        arg_pdbin):

    # always reset handlers to avoid duplication if run method is called more
    # than once

    base_filename, _ = os.path.splitext(arg_pdbout)

    pdbin = arg_pdbin
    print(pdbin)

    parm = parmed.read_PDB(pdbin)

    pdbfixer = AmberPDBFixer(parm)

    pdbfixer._write_renum(base_filename)

    pdbfixer.rename_residues_forleap()

    # =====================================================================
    # make final output to new PDB file
    # =====================================================================
    final_coordinates = pdbfixer.parm.get_coordinates()[0] #model 0, watever that means
    write_kwargs = dict(coordinates=final_coordinates)
    write_kwargs['increase_tercount'] = False # so CONECT record can work properly
    

    pdb_out_filename = arg_pdbout
    pdbfixer.parm.save(pdb_out_filename, overwrite=True)


    return #ns_names, gaplist, sslist


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in",
        metavar="FILE",
        dest="pdbin",
        help="PDB input file (default: stdin)",
        default='stdin')
    parser.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        dest="pdbout",
        help="PDB output file (default: stdout)",
        default='stdout')
    parser.add_argument(
        "--constantph",
        action="store_true",
        dest="constantph",
        help="rename GLU,ASP,HIS for constant pH simulation")


    opt = parser.parse_args(argv)

    # pdbin : {str, file object, parmed.Structure}
    
    pdbin = opt.pdbin

    

    run(
        arg_pdbout=opt.pdbout,
        arg_pdbin=pdbin)





