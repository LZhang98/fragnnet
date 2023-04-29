import dgl
import torch as th
import numpy as np
from dgl.data.utils import load_graphs

class FragDAG(object):
    """_summary_
    Args:
        object (_type_): _description_
    """
    def __init__(self) -> None:
        """_summary_
        """
        self._nx_graph = None
        self._meta = None
        
    def read_from_save(self, dgl_file:str, th_file:str, verbose = False):
        """_summary_

        Args:
            dgl_file (str): _description_
            th_file (str): _description_
        """

        if verbose:
            print(">load dgl", dgl_file)
        
        glist, label_dict = load_graphs(dgl_file)
		
        if verbose:
             print(">load meta", th_file)
        self._meta = th.load(th_file)
        self._nx_graph = dgl.to_networkx(glist[0],
						node_attrs=['depth','h_formulae_idx','base_formula','mask'])
        

    def populate_data(self):
        """_summary_
        """
        dag_d = self._dag_d
        idx_to_formula = dag_d["idx_to_formula"]
        G = dag_d["nx_dag"]
        pos_dict = {}

        node_mass_l = []
        for node in G.nodes(data = True):
            #print(node)
            node_depth_mask = node[1]['depth']

            # compute depth and add pos
            depth = 0
            for depth_value in node_depth_mask:
                if depth_value == 1:
                    break
                depth += 1
            if depth not in pos_dict:
                pos_dict[depth] = 0
            else:
                pos_dict[depth] += 1
            node[1]['pos'] = (depth,pos_dict[depth])
            
            # add image
            node_atom_mask = node[1]['mask']
            highlight_atom = []
            atom_idx = 0
            for value in node_atom_mask.tolist():
                if value == 1:
                    highlight_atom.append(atom_idx)
                atom_idx += 1

    
    def _get_matched_fragment():
        #abc
    
    #def get_dag_fig(self):
    