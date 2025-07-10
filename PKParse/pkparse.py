import numpy as np
import gzip
import glob
import edlib
import os
from utils import constants


data_dir = os.getcwd()
nbrs, constants = constants(data_dir=data_dir,n_neighbors=5)
fullcode = constants["titratables_full"]
code = constants["titratables_short"]
three_to_one = constants["aa_code"]
elements = constants["elements"]
cof = constants["cofactors"]
ligands = constants["ligands"]
cnbrs = constants["disulfide_nn"]
amber=constants["amber_sites"]



class parser():
    def __init__(self, gzipped_pdb, ref_pdb):
        self.path    = gzipped_pdb
        self.pdb     = gzipped_pdb[-7:-3]
        self.targets = np.load(f"../../targets/{self.pdb}.npz")
        self.save_dir = "../inputs/"
        self.ligands = ligands
        self.ref_pdb = ref_pdb
        #self.nbrs    = nbrs
        #print(self.pdb)
        #self.num_nbrs
        
    def residue_map(self):
            """This function uses the RCSB reference pdb as the reference sequence
            for sequence alignment through edlib. This allows for the incoming residue 
            numbers to be arbitrary and allows for custom PDDs and to be fixed by modeling
            software (does not support files with non-standardized naming conventions, where
            standard is considered to be Amber, even if that isn't correct.)
            
            The output is matched pairwise such that insertions and deletions trigger 
            the increasse of their respective counter, which gets their keys made by
            residue number, chain, and residue name. 

            These make the mapping which are the only IDs which get through to be matched with 
            the pkPDB targets.
            """
            # helper: extract per-chain sequences & idx-lists into parallel lists
            def extract(path):
                seqs,seq,idxs,keys = [],[],[],[]   # list of lists of idx‐bytes
                lastkey=None
                with gzip.open(path, "rb") as fh:
                    for raw in fh:
                        if not raw.startswith(b"ATOM"): continue
                        elif raw.startswith(b"TER"):
                            seqs.append("".join(seq))
                            idxs.append(keys)
                            keys,seq=[],[]
                            continue

                        ch,resi = raw[21:22], raw[17:20]
                        key = raw[22:26].strip() + ch + fullcode.get(resi,b"")
                        
                        if lastkey==key:
                            lastkey=key
                            continue

                        keys.append(key)
                        aa1 = three_to_one.get(resi, "X")
                        seq.append(aa1)
                        lastkey=key
                        
                seqs.append("".join(seq))
                idxs.append(keys)

                # finalize: join seqs, convert idxs to np.char.array
                seq_strs = seqs#["".join(s) for s in seqs]
                idx_arrs  = [np.char.array(i) for i in idxs]
                return seq_strs, idx_arrs

            # read ref & current
            ref_seqs, ref_idxs = extract(self.ref_pdb)
            cur_seqs, cur_idxs = extract(self.path)

            mapping = {}
            # align each chain by index
            for c, oseq in enumerate(ref_seqs):
                if c >= len(cur_seqs):
                    break
                nseq  = cur_seqs[c]
                nidx = cur_idxs[c]
                oidx = ref_idxs[c]

                # 1) compute the alignment path
                res = edlib.align(oseq, nseq, mode="NW", task="path")
                # skip if no CIGAR produced
                if not res.get("cigar"):
                    # you could log a warning here if you like:
                    print(f"Warning: no alignment path for chain {c}, skipping")
                    self.mapping=None
                    return

                nice = edlib.getNiceAlignment(res, oseq, nseq)
                ref_aln, qry_aln = nice["target_aligned"], nice["query_aligned"]
                i = j = 0
                for r_c, q_c in zip(ref_aln, qry_aln):
                    if r_c != "-":
                        if q_c != "-":
                            if r_c == q_c:
                                mapping[nidx[j]] = oidx[i]
                                i += 1; j += 1
                            else: 
                                i += 1; j += 1
                        else: j += 1 #insertion
                    else: i += 1 #deletion

            self.mapping = mapping

    def parse_titratable_lines(self, lines):
        """get info from asp,glu,his,cys,tyr. removes hydrogens. keeps insertions now cuz have aligner"""
            
        amber_set = amber.values()      # byte-strings of titratable names
        self.species,self.coors, self.sites  = [],[],[]
        last_resnum                                   = None
        cur_species, cur_coords     =[],[]
        flag=False
        cur_species, cur_coords, others_c,others_s      =[],[],[],[]

        # 1) Single-pass parse & group by residue
        for line in lines:
            
            if line.lstrip().startswith(b"H"): continue
            # skip insertions 
            resnum = line[10:15].strip(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        

            # new residue?  flush the previous group
            if resnum != last_resnum and cur_species:
                if flag: #long term memory
                    self.species.append(cur_species)
                    self.coors.append(cur_coords)
                else: #send to others
                    others_s.append(cur_species)
                    others_c.append((cur_coords))
                cur_species, cur_coords = [], []
                flag=False
            
            cur_species.append(elements[line[-9:].split()[0][0:]])
            cur_coords.append((float(line[18:26]), float(line[26:34]), float(line[34:42])))

            if line[:5].strip() in amber_set:
                flag=True
                self.sites.append(line)

            last_resnum = resnum
        if flag: #long term memory
            self.species.append(cur_species)
            self.coors.append(cur_coords)
        else:
            others_s.append(cur_species)
            others_c.append((cur_coords))

        if others_s: self.others.append((np.concatenate(others_s),np.vstack(others_c)))

        self.species,self.coors=np.array(self.species,dtype=object),np.array(self.coors,dtype=object)
        
    
    def aggregate_others(self):
        """#TODO"""
        #def stack_columns(data):
        """
        data: list of (int_arr, coord_arr) tuples, where
        - int_arr is either a 1D np.ndarray of ints or an object array of int sub-arrays
        - coord_arr is either a 2D np.ndarray of shape (M,3) or an object array of 2D sub-arrays
        Returns:
        - all_ints: 1D np.ndarray of all ints concatenated
        - all_coords: 2D np.ndarray of shape (total_rows, 3)
        """
        int_chunks, coord_chunks = [], []
        for int_arr, coord_arr in self.others:
            int_chunks.append(int_arr)
            coord_chunks.append(coord_arr)
        all_ints   = np.concatenate(int_chunks, axis=0)
        all_coords = np.vstack(coord_chunks)
        return all_ints, all_coords

    
    def parse_others(self, lines):
        """parse all other non-ligand ATOM lines"""
        species, coors = [], []
        for line in lines:
            # skip hydrogens
            if not line[0:2].strip().startswith(b"H"):
                #try:
                species.append(cof[line[-9:].split()[0][0:]])
                coors.append((float(line[18:26]), float(line[26:34]), float(line[34:42])))
                #except Exception as e:
                #    print("atom not in dictionary",e,self.pdb)
        self.others.append((species, coors))

    def parse_pdb(self):
        """to do try except re: encoding/gzipped
        #hi b'HG23 ILE A 492      65.222 102.163  26.506  1.00  0.00           H   std\n'
        # #encode everything if user didnt gzip their filess? #TODO"""

        with gzip.open(self.path, "rb") as f: #TODO?
            lines=f.readlines()

        titratables, others  = [], []

        for line in lines:
            if line.startswith(b"HETATM"):
                if line[16:20].strip() in self.ligands or cof: others.append(line[12:])
            elif line.startswith(b"ATOM"):
                if line[16:20].strip() in fullcode: titratables.append(line[12:])
                else: others.append(line[12:])

        self.parse_others(others)
        self.parse_titratable_lines(titratables)
        
    

    def hoods(self,coors,species):#,sitecoors):
        others=self.aggregate_others()
        coors = np.concatenate([*coors, np.vstack(others[1])], axis=0)
        self.all_coors=np.vstack(coors)
        self.all_species=np.concatenate([*[*species, others[0]]]) 

    def get_disulfides(self,lines):
        """in: titratable lines"""
        cys_lines = [(i,line) for i,line in enumerate(lines) if line[5:6] == b"C"]
        if cys_lines:   
            cys_coors = [(float(line[1][18:26]),float(line[1][26:34]),float(line[1][34:42])) for line in cys_lines] #TODO confirm its S?
            cnbrs.fit(cys_coors)
            nbrs = cnbrs.radius_neighbors(cys_coors,radius=2.1, return_distance=False) #TODO
            bridges=[]
            for a,cys in zip(nbrs,cys_lines):
                if len(a) == 2:
                    bridges.append(cys[0])
            self.disulfides=bridges
        else: self.disulfides=None
        

    
    def run(self):
        self.others=[]
        sitecoors=[]
        ids,sites=[],[]
        
        
        #get structures of titratable residues
        self.residue_map()
        if not self.mapping:
            return
        self.parse_pdb()
        
        
        #hereeeee is the craziness. #jesse
        for line in self.sites:
            id=line[10:18].strip() + line[9:10] + code[line[5:6]] #get ids frrom sites
            id=self.mapping.get(id) #from map
            if id:
                ids.append(id)
                sitecoors.append((float(line[18:26]),float(line[26:34]),float(line[34:42])))
                sites.append(line)
        sitecoors=np.array(sitecoors).astype(np.float32)
       #that fixes when there is not a titratable site in the titratable residue. now onto targets
        
        self.get_disulfides(sites) 
        if self.disulfides: 
            sulf=np.array(self.disulfides)
            self.others.append((np.concatenate(self.species[sulf]), np.vstack(self.coors[sulf])))

        pkpdb=self.targets
        common, pidx, midx = np.intersect1d(pkpdb["ids"], ids,
                                            return_indices=True)
        site_mask = np.ones(len(self.species), dtype=bool)
        site_mask[midx] = False                            # False → keep

        if site_mask.any():                                # spill invalid sites
            self.others.append((np.concatenate(self.species[site_mask]),
                                np.vstack(self.coors[site_mask])))
            
        self.coors,self.species = self.coors[midx], self.species[midx]
        others=self.aggregate_others()
        #self.hoods(self.coors[midx], self.species[midx])#, self.sitecoors[midx])
        atom_coords = np.concatenate([*self.coors, np.vstack(others[1])], axis=0).astype(np.float32)
        atom_elems  = np.concatenate([*self.species,
                                      others[0]]).astype(np.uint8)


        np.savez_compressed(
    self.save_dir + f"{self.pdb}.npz",
    z=atom_elems,
    pos=atom_coords,
    pks=pkpdb["pks"][pidx],
    ids=common,
    sites=sitecoors[midx]

)
        return self

