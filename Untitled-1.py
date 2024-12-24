###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.data import AtomicData
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_sum

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
)
from .utils import (
    get_edge_vectors_and_lengths,
)

# pylint: disable=C0302

from torch_geometric.nn import radius_graph

@compile_mode("script")



class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        correlation: Union[int, List[int]],
        gate: Optional[Callable], #activation
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        cueq_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int32)
        )
        heads = ["default"]
        self.heads = heads

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions



        #edge_index = radius_graph(positions, r=cutoff_radius)

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        
        #pairrepulsion
        self.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=num_polynomial_cutoff)
        self.pair_repulsion = True

        #spjericalharmonics
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        
        
        # Interactions and readout
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
        )

        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps

        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
        )
        self.products = torch.nn.ModuleList([prod])


        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearReadoutBlock(
                hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config
            )
        )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2: #last layer
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer

            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
            )
            self.interactions.append(inter)

            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
            )
            self.products.append(prod)


            if i == num_interactions - 2: #last layer
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                        cueq_config,
                    )
                )

            else:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config
                    )
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        
        N = data["positions"].shape[0]
        
        num_atoms_arange = torch.arange(N)
        num_graphs = data["ptr"].numel() - 1
        node_heads = (torch.zeros_like(data["batch"]))
        
       # Create a Mask for Titratable Sites
        #Identify the nodes (atoms or residues) corresponding to titratable sites where pKa values are known.
        #Create a mask (a binary array) indicating whether a node should contribute to the loss:
        
        mask = torch.zeros(N, dtype=uint8)
        mask[titratable_node_indices] = 1  # Only titratable sites are "active" #HERE

        # Create a mask for titratable sites (node-specific labels)
        mask[titratable_node_indices] = True  # Mark titratable sites

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads 
        ] #return torch.matmul(x, torch.atleast_2d(self.atomic_energies).T)

        
        #e0 = scatter_sum(
            #src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        #)  # [n_graphs, n_heads]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        positions=data["positions"]

        edge_index = radius_graph(positions, r=cutoff_radius) #dtype? #connectivity

        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=edge_index,
            shifts = torch.zeros((edge_index.size(1), 3)),
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], edge_index, self.atomic_numbers
        )


        #pair repulsion
        pair_node_energy = self.pair_repulsion_fn(
            lengths, data["node_attrs"], edge_index, self.atomic_numbers
        )
        #pair_energy = scatter_sum(
            #src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
        #)  # [n_graphs,]



        # Interactions
        reps = [node_e0, pair_node_energy]
        node_feats_list = []

        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            #interacrion block
            node_feats, sc = interaction( #can be based off small distance
                node_attrs=data["node_attrs"],
                node_feats=node_feats, #self.node_embedding(data["node_attrs"])
                edge_attrs=edge_attrs, #connectivity. "vectors"
                edge_feats=edge_feats, #Zs, connectivity, features
                edge_index=edge_index,
            )

            #update features on output of interaction nodefeats, sc
            node_feats = product( 
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

            #here jessi
            node_feats_list.append(node_feats)


            node_energies = readout(node_feats, node_heads)[
                num_atoms_arange, node_heads
            ]  # [n_nodes, len(heads)]

            x = node_energies +  pair_node_energy


            codes =[0,1,2,0] #GLY30, THR26, TYR18, ILE27 
            nodes={}
            #edge_index[0,:] is the indicies for energy
            c1ei = edge_index[0,:] #len should equal nodeEs and set gives node is
            for idx, count in torch.unique_consecutive(c1ei, return_inverse=False, return_counts=True, dim=None):
                if codes[idx]==0: #make dict
                    continue
                else:
                    #ints = edge_index[edge_index[:, 0] == idx]
                    ints = edge_index[idx:idx*count]
                    contributing_sites = ints[:,1] #pals
                    for i in contributing_sites:
                        indices = torch.where(ints)[0]
                        new_node = x[idx] + torch.sum(x[i]/(lengths[j]) for i, j in zip(contributing_sites, indices))
                        nodes.update({idx: new_node})
                        #new_node_idxs.append(idx)

        c = 1
        
        

        rep = torch.stack(nodes.values()).T

        cg_rep = torch.matmul(c*rep, node_e0)
        
        positions = positions[torch.stack(nodes.keys())] #apply mask to positions

        import torch
        import torch.nn.functional as F
        # Linear layers to map features to query, key, and value
        query_layer = torch.nn.Linear(5, 3)  # Map features to a 3D query vector
        key_layer = torch.nn.Linear(5, 3)    # Map features to a 3D key vector
        value_layer = torch.nn.Linear(5, 3)  # Map features to a 3D value vector

        # Compute query, key, and value for each atom
        query = query_layer(torch.cat((cg_rep, positions))) # Shape: (3 atoms, 3)
        key = key_layer(atoms)      # Shape: (3 atoms, 3)
        value = value_layer(atoms)  # Shape: (3 atoms, 3)

        # Compute pairwise attention scores (dot product between query and key)
        scores = torch.matmul(query, key.T) / (3 ** 0.5)  # Shape: (3 atoms, 3 atoms)

        # Normalize scores using softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)  # Shape: (3, 3)

        # Compute weighted sum of values for each atom
        output = torch.matmul(attention_weights, value)  # Shape: (3 atoms, 3)

        # Single representation for each node
        #node_features = torch.randn(10, 64)  # 10 nodes, 64-dimensional representation

        # Compute attention scores directly
        n_attention_feats = 2 #proportional to number of states?
        attention_scores = torch.matmul(cg_rep, positions.T) / (n_attention_feats ** 0.5)  # Self-attention
        attention_weights = F.softmax(attention_scores, dim=-1)



        # Weighted aggregation (output still has the same shape as input)
        output = torch.matmul(attention_weights, node_features)
        #print("Output shape:", output.shape)

        edge_index = radius_graph(positions)

        import torch
        import torch.nn.functional as F

        #attention mechanism for this one with e.g. phe still

        #attention_weights = F.softmax(attention_scores, dim=-1)


            #coarse grain here

            #energy = scatter_sum(
                #src=node_energies,
                #index=data["batch"],
                #dim=0,
                #dim_size=num_graphs,
            #)  # [n_graphs,]
            #energies.append(energy)
            #node_energies_list.append(node_energies)
        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1) #here can concatenate w pboltzmann


        #go for round 2?


        # Sum over energy contributions
        #contributions = torch.stack(energies, dim=-1)
        #total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]



        return {
            #"energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "node_feats": node_feats_out, #can get "partial chathes?"
        }







@compile_mode("script") ##HERE











class BOTNet(torch.nn.Module): #experiment w modular in botnet?
    """
    edge_index: determines edges (connections) between nodes (atoms) [2, num_connections]
    """
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        avg_num_neighbors: float,
        atomic_numbers: List[int],
    ):
        super().__init__()
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readouts

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(inter.irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(inter.irreps_out))

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:

        data.positions.requires_grad = True #this means positions can be updated later during training
        num_atoms_arange = torch.arange(data.positions.shape[0])


        ##Get e0: initial 
        # each atom has a base energy contribution Eo based on its Z
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, data["head"][data["batch"]]
        ]
        
        #adds up all the Eo contributions to get a starting point for total energy
        #e0 = scatter_sum( #SUMS ALL
            #src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        #)  # [n_graphs, n_heads]


        # Embeddings
        node_feats = self.node_embedding(data.node_attrs) #create rep based on q and z 
        
        #calculate direction from 1 connected to another, and distance between
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )

        #calculates angular information using spherical harmonics
        edge_attrs = self.spherical_harmonics(vectors)
        
        # calculates distance-based features for the atom pairs dp on Z also:
        #distances converted into bessels or polynomials
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions #updates atomic features based on neighbors
        #energies = [e0] #SUMMED ALL
        atom_reps = [node_e0]
        
        for interaction, readout in zip(self.interactions, self.readouts):
            
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs, #output [num_connections, sphericalharmonicsD (dt by max_ell)]
                edge_feats=edge_feats, #output [num_conn, radialD]
                edge_index=data.edge_index,
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]  #predict atom-wise energy
            #D= N atoms!
    
        

            #energy = scatter_sum( #SUMS ALL
                #src=node_energies, index=data.batch, dim=-1, dim_size=data.num_graphs
            #)  
            atom_reps.append(node_energies) #SUMMED ALL

        return
        # Sum over energy contributions
        ##contributions = torch.stack(energies, dim=-1) [e0, interaction_energy]

        ##jessi here

        ##site_energy = torch.sum(contributions[i]
        #total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        

        output = {
            #"energy": total_energy,
            "contributions": contributions
        }

        return output


