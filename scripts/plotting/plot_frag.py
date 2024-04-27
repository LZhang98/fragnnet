import argparse
import pickle
import torch as th
from tqdm import tqdm

from frag_gnn.utils.frag_utils import load_frag_d, th_long_to_mask
from frag_gnn.utils.plot_utils import plot_spectra_sparse, get_mol_im
from frag_gnn.utils.spec_utils import *

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem as Chem

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from rdkit.Chem import Draw, AllChem

from io import BytesIO
from math import ceil
import re
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from rdkit.Chem import Draw, AllChem
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mc
from adjustText import adjust_text
from matplotlib.text import Text
from frag_gnn.utils.plot_utils import get_mol_im, get_highlighted_bonds
from PIL import Image
from torch_geometric.utils import to_networkx
from frag_gnn.utils.frag_utils import timed_get_dags
import networkx as nx   
import pygraphviz

def plot_mol_frag_axes( mol, 
                        atom_highlight_idx:list,
                        bond_highlight_idx:list,image_width=300, image_height=300):

    #print(image_width, image_height)
    d2d = Draw.MolDraw2DCairo(image_width, image_height)
    options = d2d.drawOptions()
    rgba_color = (0.529, 0.808, 0.98)# transparent blue
    options.fillHighlights=True
    options.setHighlightColour((rgba_color))
    options.highlightBondWidthMultiplier=20
    options.clearBackground = False
    d2d.DrawMolecule(mol,highlightAtoms=atom_highlight_idx, highlightBonds =  bond_highlight_idx)
    d2d.FinishDrawing()
    bio = BytesIO(d2d.GetDrawingText())
    return Image.open(bio)

if __name__ == "__main__":
    mol = Chem.MolFromSmiles('C(N(C))O')
    depth = 3
    frag_d = timed_get_dags(mol, 0, depth, True, 4, 150, False,False,1,None, False,False,False)
    #print(frag_d)
    data = frag_d['dag']
    #networkX_graph = to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"])
    #nx.draw(networkX_graph)
    #plt.show()
    frag_cc_depth = frag_d['dag'].x[:,0:depth+1]
    frag_cc_longs = frag_d['dag'].x[:,depth+1:depth+2]

    num_nodes = frag_d['dag_num_nodes']
    idx_to_formula = frag_d['idx_to_formula']
    cc_bit_mask = th_long_to_mask(frag_cc_longs)
    #print(frag_cc_longs)
    #print(f">num_nodes {num_nodes}")
    
    #mol = Chem.MolFromSmiles(smiles)
    node_min_depths = []
    for i in range(num_nodes):
        node_cc_bit_mask = cc_bit_mask[i]
        #print(node_cc_bit_mask)
        #print(frag_cc_depth[i])
        node_min_depth = (frag_cc_depth[i] == 1).nonzero(as_tuple=True)[0][0]
        print(i, frag_cc_depth[i], node_min_depth)
        node_min_depths.append(node_min_depth)
        node_atom_indices = node_cc_bit_mask.nonzero(as_tuple=True)[0].tolist()
        node_bond_indices = get_highlighted_bonds(mol, node_atom_indices)
        #print(node_atom_indices)
        #print(node_bond_indices)
        node_image = plot_mol_frag_axes(mol,node_atom_indices,node_bond_indices)
        node_image.save(f"figs/icml/frag_example/frag_{i}.png")

    #print(data.edge_index.tolist())
    edge_list = data.edge_index.T.tolist()
    #print(edge_list) 
    with open("figs/icml/frag_example/frag_md.md", "w+") as f:
        f.write(f"```mermaid\n")
        f.write(f"graph TD\n")
        for i in range(num_nodes):
             f.write(f""" f{i}("<b>Node {i} depth{node_min_depths[i]}</b><img src='frag_{i}.png'; width='300' />"):::green\n""")
        for edge in edge_list:
            link_len = node_min_depths[edge[1]]-node_min_depths[edge[0]]
            #print(link_len)
            print(link_len)
            if link_len < 2:
                link_len = 2

            link = "-" * link_len + '>'
            f.write(f"    f{edge[0]} {link} f{edge[1]}\n")
        f.write(f"```\n")