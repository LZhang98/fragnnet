import frag_gnn.utils.data_utils as data_utils
import frag_gnn.utils.frag_utils as frag_utils
import frag_gnn.frag.compute_frags as compute_frags

from pprint import pprint
import rdkit.Chem as Chem

def graph_hash_helper(mol, include_bond_type=False):

	mol_d = frag_utils.extract_mol_info(mol)
	elems = mol_d["elems"]
	bond_type_idxs = mol_d["bond_type_idxs"]
	num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = frag_utils.get_fraggen_input_arrays(mol_d)
	assert num_nodes == len(elems), (num_nodes, len(elems))
	assert num_edges == len(bond_type_idxs), (num_edges, len(bond_type_idxs))
	graph_hash = frag_utils.wl_hash(
		elems, 
		bond_type_idxs, 
		node_mask, 
		edges[:num_edges], 
		node_to_edge_idx[:num_nodes],
		include_bond_type=include_bond_type)
	return graph_hash

def subgraph_hash_helper(mol, max_depth=3, include_bond_type=False):

	mol_d = frag_utils.extract_mol_info(mol)
	elems = mol_d["elems"]
	bond_type_idxs = mol_d["bond_type_idxs"]
	num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = frag_utils.get_fraggen_input_arrays(mol_d)
	nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = compute_frags.compute_ccs(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		node_to_edge_idx,
		max_depth,
		int(1e6)
	)
	subgraph_hashes = []
	num_subgraphs = nodes_mask_matrix.shape[0]
	for i in range(num_subgraphs):
		subgraph_mask = nodes_mask_matrix[i]
		subgraph_hash = frag_utils.wl_hash(
			elems, 
			bond_type_idxs, 
			subgraph_mask, 
			edges[:num_edges], 
			node_to_edge_idx[:num_nodes],
			include_bond_type=include_bond_type)
		subgraph_hashes.append(subgraph_hash)
	return subgraph_hashes

def graph_test_1():
	""" Aspirin randomizations """

	base_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
	num_rseeds = 5
	graph_hashes = []
	for i in range(num_rseeds):
		random_smiles = data_utils.randomize_smiles(base_smiles,rseed=i-1)
		mol = Chem.MolFromSmiles(random_smiles)
		graph_hash = graph_hash_helper(mol)
		graph_hashes.append(graph_hash)
	assert len(set(graph_hashes)) == 1

def graph_test_2():
	""" Butane and butenes """

	smiles_1 = "CCCC" # butane
	smiles_2 = "C=CCC" # 1-butene
	smiles_3 = "CC=CC" # 2-butene
	smileses = [smiles_1, smiles_2, smiles_3]
	graph_hashes = []
	for include_bond_type in [False, True]:
		for smiles in smileses:
			mol = Chem.MolFromSmiles(smiles)
			graph_hash = graph_hash_helper(mol,include_bond_type=include_bond_type)
			graph_hashes.append(graph_hash)
	assert graph_hashes[0] == graph_hashes[1] == graph_hashes[2]
	assert graph_hashes[0] != graph_hashes[3]
	assert graph_hashes[0] != graph_hashes[4]
	assert graph_hashes[0] != graph_hashes[5]
	assert graph_hashes[3] != graph_hashes[4]
	assert graph_hashes[3] != graph_hashes[5]
	assert graph_hashes[4] != graph_hashes[5]

def subgraph_test_1():
	""" Propane """

	smiles = "CCC"
	mol = Chem.MolFromSmiles(smiles)
	subgraph_hashes_nb = subgraph_hash_helper(mol, include_bond_type=False)
	subgraph_hashes_b = subgraph_hash_helper(mol, include_bond_type=True)
	assert len(subgraph_hashes_nb) == len(subgraph_hashes_b)
	assert len(subgraph_hashes_nb) == 6
	assert len(set(subgraph_hashes_nb)) == 3
	assert len(subgraph_hashes_b) == 6
	assert len(set(subgraph_hashes_b)) == 3

def subgraph_test_2():
	""" Propene """

	smiles = "C=CC"
	mol = Chem.MolFromSmiles(smiles)
	subgraph_hashes_nb = subgraph_hash_helper(mol, include_bond_type=False)
	subgraph_hashes_b = subgraph_hash_helper(mol, include_bond_type=True)
	assert len(subgraph_hashes_nb) == len(subgraph_hashes_b)
	assert len(subgraph_hashes_nb) == 6
	assert len(set(subgraph_hashes_nb)) == 3
	assert len(subgraph_hashes_b) == 6
	assert len(set(subgraph_hashes_b)) == 4

def subgraph_test_3():
	""" 1-Methylcyclohexene """

	smiles = "CC1=CCCCC1"
	mol = Chem.MolFromSmiles(smiles)
	subgraph_hashes_nb = subgraph_hash_helper(mol, include_bond_type=False)
	subgraph_hashes_b = subgraph_hash_helper(mol, include_bond_type=True)
	assert len(subgraph_hashes_nb) == len(subgraph_hashes_b)
	assert len(subgraph_hashes_nb) == 48
	# assert len(set(subgraph_hashes_nb)) == 11, len(set(subgraph_hashes_nb))
	assert len(set(subgraph_hashes_nb)) == 12, len(set(subgraph_hashes_nb))
	assert len(subgraph_hashes_b) == 48
	# assert len(set(subgraph_hashes_b)) == 20, len(set(subgraph_hashes_b))
	assert len(set(subgraph_hashes_b)) == 21, len(set(subgraph_hashes_b))

if __name__ == "__main__":

	# TODO: test with variety of molecules
	# TODO: time it (integrate with run_compute_frags.py?)

	graph_test_1()
	graph_test_2()
	subgraph_test_1()
	subgraph_test_2()
	subgraph_test_3()
