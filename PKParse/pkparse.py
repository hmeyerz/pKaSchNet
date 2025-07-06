import numpy as np
import gzip
import glob
import edlib
import os
from init import constants


data_dir = os.getcwd()
nbrs, constants = constants(data_dir=data_dir,n_neighbors=50)
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
        self.targets = np.load(data_dir + "/targets/{self.pdb}.npz")
        self.save_dir = data_dir + "/ßinputs/"
        self.ligands = ligands
        self.ref_pdb = ref_pdb
        self.nbrs    = nbrs
        
    def residue_map(self):
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
                    continue

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
        """get info from asp,glu,his,cys,tyr. removes hydrogens."""
            
        amber_set = amber.values()      # byte-strings of titratable names
        self.species,self.coors, self.sites, self.ids  = [],[],[],[]
        last_resnum, flag                                    = None, True
        cur_species, cur_coords, others_c,others_s      =[],[],[],[]

        # 1) Single-pass parse & group by residue
        for line in lines:
            if line.lstrip().startswith(b"H"): continue
            # skip insertions 
            resnum = line[10:15].strip()
            if not resnum.isdigit():
                self.others.append(line)
                continue

            # new residue?  flush the previous group
            if resnum != last_resnum and cur_species:
                self.species.append(cur_species)
                self.coors.append(cur_coords)
                cur_species, cur_coords = [], []
                
            cur_species.append(elements[line[-8:].strip()[0:1]])
            cur_coords.append((float(line[18:26]), float(line[26:34]), float(line[34:42])))

            if line[:5].strip() in amber_set:
                flag=False
                self.sites.append(line)
            
            last_resnum = resnum

        if not flag: #titratable flag
            self.species.append(cur_species)
            self.coors.append((cur_coords))
            flag=True
        else:
            others_s.append(cur_species)
            others_c.append((cur_coords))
            flag=True
        if others_s: self.others.append((np.concatenate(others_s),np.vstack(others_c)))

        self.species,self.coors=np.array(self.species),np.array(self.coors)
        
    
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
                species.append(elements[line[-8:].strip()[0:1]])
                coors.append((float(line[18:26]), float(line[26:34]), float(line[34:42])))
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
        
    

    def hoods(self,coors,species,sitecoors):
        others=self.aggregate_others()
        coors = np.concatenate([*coors, np.vstack(others[1])], axis=0)
        nbrs = self.nbrs.fit(np.vstack(coors)).kneighbors(sitecoors,500, return_distance=False)
        species=np.concatenate([*[*species, others[0]]]) 
        self.species,self.coors=[species[n_ixs] for n_ixs in nbrs], [coors[n_ixs] for n_ixs in nbrs]
    
    
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
        
        #get structures of titratable residues
        self.residue_map()
        self.parse_pdb()

        
        ids,idxs,sites=[],[],[]
        for i,line in enumerate([t for t in self.sites]):
            id=line[10:18].strip() + line[9:10] + code[line[5:6]]
            id=self.mapping.get(id)
            if id:
                ids.append(id)
                sitecoors.append((float(line[18:26]),float(line[26:34]),float(line[34:42])))
                sites.append(line)
            else: idxs.append(i)
        self.sitecoors=np.array(sitecoors)
        if idxs:
            mask1 = np.ones(len(self.species), dtype=bool)
            idxs=np.array(idxs)
            mask1[idxs] = False
            self.others.append((np.concatenate(self.species[idxs]), np.vstack(self.coors[idxs])))
            self.species,self.coors=self.species[mask1],self.coors[mask1]

        self.get_disulfides(sites) 
        if self.disulfides: 
            sulf=np.array(self.disulfides)
            self.others.append((np.concatenate(self.species[sulf]), np.vstack(self.coors[sulf])))
            
        pkpdb=self.targets
        self.ids=ids
        mask = np.ones(len(self.ids), dtype=bool)
        self.ids, pidx, midx = np.intersect1d(pkpdb["ids"], self.ids,return_indices=True)
        mask[midx] = False
        if mask.any():
            self.others.append((np.concatenate(self.species[mask]), np.vstack(self.coors[mask])))

        self.hoods(self.coors[midx],self.species[midx],self.sitecoors[midx])

        np.savez_compressed(
    self.save_dir + f"{self.pdb}.npz",
    z=np.array(self.species),
    pos=np.array(self.coors),
    pks=np.array(pkpdb["pks"][pidx]),
    ids=np.array(self.ids)
)
        #except:
            #return
            #print(self.pdb)
    

        return self