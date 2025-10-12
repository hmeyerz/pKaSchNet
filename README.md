This work is for my project: GNNs on pkPDB (ablated with attention). It aims to improve over both a minimal model (pk-EGNN) and PKAI. This repo shows the development from the idea of using SchNet-based GNN implementations (SchNetPack, ASE), to the vanilla EGNN "pK-EGNN" using my NPZ dataset with metals/ligands (TUB Property) on modeled RCSB data (TUB Property) and compressed pkPDB shifts with PKAD as val set. This work was to develop my master's thesis using CFCNNs to predict protein pKa values. It has been developed for 2.75 years with the help of the TUBiomodeling and FUB (CC+team) and undergone many iterations and potential final implementations, until the best ones were found for production.

Original Work of Jessi Rose Hoernschemeyer, myself and chatGPT authored the codes here except for the SchNet architecture or TorchMD_Net architectures, if they are still in the repo or when they were. 

Do not reproduce, distribute, or publish final models that have been tuned on outputs from the software in this repo or specific architecture implementations here e.g. schedulers, or codes directly reproduced from methods designed here (ML pipelines: SchNetPack/ASE --> NPZ/PyTorch) without explicit usage rights from jrhoernschemeyer@gmail.com. 

June 2023
For each titratable residue (and its neighboring residues) in PKAD+pkPDB
-Atom positions & names obtained from PDBs + MDAnalysis → ASE
-Atom types from rtf topology file → ?
-Partial charges obtained from rtf topology file → ASE
My Data
Experimental: PKAD
● 1,241 PKA values
● 154 PDBs
Theoretical: pkPDB (PypKa)
● 12,628,149 PKA values (as
of 18/5/2023)

Input Data (ASE Database)
1. Atomic Positions and Charges Z
list of ASE Atoms Objects. A single array of every atom in my entire data set in order of pandas data frame
which is the only way i can suggest to preserve molecular information
● 1 ‘Atoms Object’ = a collection of atom objects
○ 1 Atom Object = atom name, positions, partial charge
2. Dictionary of molecular properties (pKa, Residue Name, Residue Number, Chain)
{property_name1: property1_molecule1}
Unuseable/Unable to be incorporated information: Atom Type
-Choose Data Split (50/25/25, 60/20/20..), Batch Size, Number of Workers
Build Model
1. Define input module
2. Representation
● Choice of representation: SchNet
● Choice of algorithm size (number of interactions, )
○ Build multiple algorithms of different sizes, train them, and choose best
● Choice of Loss Function: MSE
● Choice of Cutoff Function (+ choice of cutoff)
● Choice of Radial Basis Function (+# of Gaussians)
3.DefineOutput:MolecularProperty(pKavalue)

April 2024

Model 1 (schnet)
Each cutout size depending on cutoff radius.
cutout = every atom of the titratable residue +& neighboring
residues (excluding H20 & those with digits)
Data = Sql3lite database with an Atoms objects (= a cutout):
Data Size : 40cutouts is 1800kb
Cutout : Atoms(symbols=‘Every Atom of the Entire Cutout’,
 positions=.....)
Output: atomic energies
Model 2
Input: atomwise or residue-wise energies
Anything else? %SASA (biopython?) Residue_depth
(biopython)? Change coordinate system? B-factor?
Output: PKA
How did I get my data?
● PDB name, Residue Number, Chain → generates cutout (MDA) from PDBs
● It would take 38 days to generate the neighborhoods on my CPU
?’s: Data reduction? How to efficiently write the data


Dec 2024

RCSB (PDBs without metals)
⇣
PDBFixer (fill missing atoms)
⇣
Reduce (optimize Hs) 
⇣
PDB2PQR (Partial Charges*)
⇣
Biopython (make cuts*)
⇣
MACE 

Feb 2025

Aggregation Methods:

SCHNET: Linear →Sum Aggregate

EGNN: Linear →Optional Aggregate

MACE: Linear →Sum Aggregate

PKA = F(species, position, charge)

============================================================

**+++++++++++++++++++++++++++++++++++++++++++++++++++++++**

Project history: 

**Plan Phase: 2023**

Make 2 pka-regressors --> 
make 1 deep learning regressor on pkPDB labels (no AlphaFold) and PKAD -->
make 1 CNN on pkPDB+PKAD without metals -->

*Planned Database: pkPDB+PKAD unmodeled proteins which don't have metals (~60K proteins), protonated and unprotonated*, every atom who is a member of a labeled site

**Prebuild Phase: 2023-2024**

Adapt SchNet via TorchMD-Net to pKa-prediction   Inputs = Z, R, rule-based partial charges and atom types.      Independent Rcuts with Hydrogens, all-atom    

Adapt SchNet w/ pytorch geometric -> pK          Inputs = Z, R       pkPDB+PKAD                                 Independent Rcuts without Hydrogens, all-atom   

Adapt SchNet energy outputs w/ SchNetPack -> pK  Inputs = Z, R       pkPDB+PKAD                                 Independent Rcuts without Hydrogens, all-atom    

Transfer learning: pretrained SchNet E-> pK      Inputs = Z, R       pkPDB+PKAD                                 Independent Rcuts without Hydrogens, all-atom   

*Database: pkPDB+PKAD protonated PDBs which don't have metals (~60K proteins)*, every atom who is a member of a labeled site

**Plan to Product (2024-2025)**

Adapt/Employ MACE, NequIP, or PAINN to pKa-prediction Inputs = Z, R, maybe q   pkPDB+PKAD                            Full protein without Hydrogens, all-atom        

Wrap EGNN-Network for macro pKa-prediction       Inputs = Z, R            pkPDB                                 Independent Rcuts without Hydroges, all-atom     

*Database: all pkPDB protonated PDBs, stripped (Z=6,7,8, or 16), dehydrated (full 121K)*, every atom who is a member of a labeled site

1EGNN_Network, 1mha, + Linears --> pK-shift      Inputs = Z, R            pkPDB                                 Independent Rcuts, batch = protein, all-atom   

*Database: all pkPDB protonated PDB solutes, dehydrated (121K pkPDB, no AlphaFold)*, every atom of a biological molecule, all-H

GAT (EGNN/MHA/readout layer) --> pK-shift        Inputs = Z, R            pkPDB 121K                            Independent Kcuts, batch = protein, all-atom     



Today:
--> Hybrid architecture, multiresolution  -->    Inputs = Z, R                                                  Coupled Kcuts, batch = X proteins, all-atom      


For my project we remove partial charges and assume charges can be gotten from Z, thus pKa = F(charges(species,pos), pos) of local env



