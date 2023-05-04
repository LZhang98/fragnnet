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
        glist, _ = load_graphs(dgl_file)
		
        if verbose:
            print(">load meta", th_file)
        self._meta = th.load(th_file)

        if verbose:
            print(">covert to nx graph", th_file)
        self._nx_graph = dgl.to_networkx(glist[0],
						node_attrs=['depth','h_formulae_idx','base_formula','mask'])
        
        # parse saved data
        idx_to_formula = self._meta["idx_to_formula"]
        pos_dict = {}

        node_mass_l = []
        for node in self._nx_graph.nodes(data = True):
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
        pass
        #abc

    def save_to_file(self, dgl_file:str, th_file:str, verbose = False):
        """_summary_

        Args:
            dgl_file (str): _description_
            th_file (str): _description_
        """

        if verbose:
            print(f">save dgl to f{dgl_file}")
        assert self._dag_dgl_graph is not None
        dgl.save_graphs(dgl_file,self._dag_dgl_graph)

        if verbose:
            print(f">save meta to {th_file}")
        
        th_d = dict(
        mol_id=self._mol_id,
        formula_peak_probs=self._formula_peak_probs,
        formula_peak_mzs=self._formula_peak_mzs,
        idx_to_formula=self._idx_to_formula)
        
        th.save(th_d,th_file)

    # land of static methods or class methods
    @staticmethod
    def mask_from_int(num_nodes:int,cc_int:int,bitmask=True)->list[int]:
        """_summary_

        Args:
            num_nodes (int): _description_
            cc_int (int): _description_
            bitmask (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        cc = []
        quot = cc_int
        for i in range(num_nodes):
            quot, rem = divmod(quot,2)
            if bitmask:
                cc.append(int(rem))
            else:
                if rem == 1:
                    cc.append(i)
        cc = tuple(cc)
        return cc

    @staticmethod
    def mask_to_int(cc:list[int])->int:
        """_summary_

        Args:
            cc (list[int]): _description_

        Returns:
            _type_: _description_
        """
        return sum([2**i for i in cc])

    @staticmethod
    def get_arrays_from_mol_d(mol_d:dict):
        """_summary_

        Args:
            mol_d (dict): _description_

        Returns:
            _type_: _description_
        """
        num_nodes = mol_d["atom_mask_arr"].shape[0]
        num_edges = mol_d["bond_mask_arr"].shape[0]
        assert num_nodes <= MAX_NUM_NODES, num_nodes
        assert num_edges <= MAX_NUM_EDGES, num_edges
        edges = np.zeros((MAX_NUM_EDGES,2),dtype=DTYPE)
        for bond_idx, bond in enumerate(mol_d["bonds"]):
            edges[bond_idx,0] = bond[0]
            edges[bond_idx,1] = bond[1]
        edge_to_node_idx = compute_frags.py_compute_node_to_edge_idx(num_nodes,num_edges,edges)
        # print(edge_to_node_idx)
        node_mask = np.zeros((MAX_NUM_NODES,),dtype=DTYPE)
        node_mask[:num_nodes] = 1
        edge_mask = np.zeros((MAX_NUM_EDGES,),dtype=DTYPE)
        edge_mask[:num_edges] = 1
        return num_nodes, num_edges, node_mask, edges, edge_mask, edge_to_node_idx
    
    @classmethod
    def formula_arr_to_str(cls, formula:list|np.ndarray):
        """ use canonical order """
        formula_str = ""
        elem_d = {}
        for idx, count in enumerate(formula):
            elem = cls.IDX_TO_ELEMENT[idx]
            elem_d[elem] = count

        for elem in cls.CANONICAL_ELEMENT_ORDER:
            if elem in elem_d:
                count = elem_d[elem]
                if count > 0:
                    formula_str += elem
                if count > 1:
                    formula_str += str(count)
        return formula_str

    @classmethod
    def extract_mol_info(cls,smiles_or_mol:str|Chem.rdchem.Mol, use_default_valence=False) -> dict:
        if isinstance(smiles_or_mol,str):
            # rdkit mol stuff
            mol = Chem.MolFromSmiles(smiles_or_mol)
            # TODO: think about adding Hs
            rdMolStandardize.Cleanup(mol)
            te = rdMolStandardize.TautomerEnumerator()
            mol = te.Canonicalize(mol)
        else:
            assert isinstance(smiles_or_mol,Chem.rdchem.Mol), type(smiles_or_mol)
            mol = smiles_or_mol


        #GetValenceList
        # some checks
        charge = rdmolops.GetFormalCharge(mol)
        assert charge == 0, charge
        # enumerate atoms
        sbond_arr = []
        ve_arr = []
        hs_arr = []
        elems = []
        num_hs = 0
        num_atoms = 0
        num_radicals = 0
        element_counts = {elem:0 for elem in cls.ELEMENT_TO_VE.keys()}

        for atom in mol.GetAtoms():
            cur_idx = atom.GetIdx()
            cur_num_hs = atom.GetTotalNumHs()
            cur_deg = atom.GetTotalDegree()
            # cur_num_bonds = atom.GetNumBonds()
            cur_element = atom.GetSymbol()

            if cur_element not in element_counts:
                raise ValueError(f"Molecules with {cur_element} atom(s) currently not supported")
            
            element_counts[cur_element] += 1
            elems.append(cur_element)
            sbond_arr.append(cur_deg - cur_num_hs)
            #ve_arr.append(ELEMENT_TO_VE[cur_element])

            # set valence value for each atom
            # assumption we will use current valance unless
            # use default valence flag is set to true
            # not defualt valence can be -1 for transition metals
            # on paper we should not encounter them at all
            ve_value = atom.GetTotalValence()
            if use_default_valence:
                default_valence = PT_Table.GetDefaultValence(cur_element)
                if default_valence != -1:
                    ve_value = min(ve_value, default_valence)

            ve_arr.append(ve_value)

            hs_arr.append(cur_num_hs)
            num_hs += cur_num_hs
            num_atoms += 1
            num_radicals += atom.GetNumRadicalElectrons()

        element_counts["H"] = num_hs
        assert num_radicals == 0, num_radicals
        sbond_arr = np.array(sbond_arr)
        ve_arr = np.array(ve_arr)
        # enumerate bonds
        atoms_to_bonds = {}
        bonds = []
        num_bonds = 0
        adj = np.zeros((num_atoms, num_atoms), dtype=np.int32)
        #adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        for bond in mol.GetBonds():
            cur_idx = bond.GetIdx()
            from_idx = bond.GetBeginAtomIdx()
            to_idx = bond.GetEndAtomIdx()
            assert from_idx != to_idx
            adj[from_idx, to_idx] = 1

            bonds.append((from_idx, to_idx))

            if from_idx not in atoms_to_bonds:
                atoms_to_bonds[from_idx] = [cur_idx]
            else:
                atoms_to_bonds[from_idx].append(cur_idx)

            if to_idx not in atoms_to_bonds:
                atoms_to_bonds[to_idx] = [cur_idx]
            else:
                atoms_to_bonds[to_idx].append(cur_idx)
            num_bonds += 1


        mol_d = {}
        mol_d["mol"] = mol
        mol_d["num_hs"] = num_hs
        mol_d["sbond_arr"] = sbond_arr
        mol_d["ve_arr"] = ve_arr
        mol_d["hs_arr"] = hs_arr
        mol_d["elems"] = elems
        mol_d["bonds"] = bonds
        mol_d["bond_mask_arr"] = np.ones((num_bonds,), dtype=bool)
        mol_d["atom_mask_arr"] = np.ones((num_atoms,), dtype=bool) # can be computed
        mol_d["atoms_to_bonds"] = atoms_to_bonds # can be computed
        mol_d["element_counts"] = element_counts

        return mol_d