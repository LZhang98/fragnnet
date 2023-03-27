import numpy as np
import networkx as nx

import frag_utils
import compute_frags


def cython_dag(mol_d,max_depth,dag_type,h_prior,max_h_transfer):

	assert dag_type in ["formula","cch"], dag_type
	assert h_prior in [True,False], h_prior
	num_nodes, num_edges, node_mask, edges, edge_mask, edge_to_node_idx = frag_utils.get_arrays_from_mol_d(mol_d)
	# time this part
	ccs, ccs_parents, ccs_depths = compute_frags.py_compute_ccs_recursive(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		edge_to_node_idx,
		0,
		max_depth,
		True,
		True,
		True
	)
	assert len(ccs) == len(ccs_parents) == len(ccs_depths)
	ccs_to_parents = frag_utils.collect_ccs_parents(ccs,ccs_parents)
	ccs_to_depths = frag_utils.collect_ccs_depths(ccs,ccs_depths)

	# construct graph
	cc_g = nx.DiGraph()
	ccs_un = sorted(list(set(ccs)))
	assert len(ccs_un) == len(ccs_to_parents) == len(ccs_to_depths)
	cc_g.add_nodes_from(ccs_un)
	# add node depth information
	nx.set_node_attributes(cc_g,ccs_to_depths,"depth")
	# add node mask information
	ccs_to_masks = {cc : np.array(frag_utils.convert_from_int(num_nodes,cc,bitmask=True)) for cc in ccs_un}
	nx.set_node_attributes(cc_g,ccs_to_masks,"mask")
	# add edges
	edges_to_diff_masks = {}
	for cc, parents in ccs_to_parents.items():
		cc_mask = ccs_to_masks[cc]
		for parent in parents:
			cc_g.add_edge(parent,cc)
			parent_mask = ccs_to_masks[parent]
			diff_mask = parent_mask - cc_mask
			edges_to_diff_masks[(parent,cc)] = diff_mask
	# add edge mask information
	nx.set_edge_attributes(cc_g,edges_to_diff_masks,"diff_mask")

	if dag_type == "formula":
		ccs_to_formulae = {}
		all_formulae = set()
		for cc_int in ccs_un:
			cc_mask = frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False)
			formulae = frag_utils.compute_approximate_formulae([cc_mask],mol_d,h_prior=h_prior,max_h_transfer=max_h_transfer)
			formulae = [frag_utils.get_formula_string(f) for f in formulae]
			ccs_to_formulae[cc_int] = formulae
			all_formulae.update(formulae)
		# print(np.mean([len(v) for v in ccs_to_formulae.values()]))
		formulae_to_idx = {formula:idx for idx,formula in enumerate(sorted(list(set(all_formulae))))}
		for k,v in ccs_to_formulae.items():
			ccs_to_formulae[k] = [formulae_to_idx[f] for f in v]
		formula_g = nx.DiGraph()
		formula_g.add_nodes_from(list(range(len(formulae_to_idx))))
		for cc,formulae in ccs_to_formulae.items():
			cc_out_edges = cc_g.out_edges(cc)
			for out_formula in formulae:
				for cc_out_edge in cc_out_edges:
					assert cc_out_edge[0] == cc
					for in_formula in ccs_to_formulae[cc_out_edge[1]]:
						formula_g.add_edge(out_formula,in_formula)
		return cc_g, formula_g
	
	else:
		# identify nodes
		all_cchs = set()
		dropped_ccs = []
		for cc_int in ccs_un:
			cc_mask = frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False)
			cchs = frag_utils.compute_approximate_cchs([cc_mask],mol_d,h_prior=h_prior,max_h_transfer=max_h_transfer)
			if len(cchs) == 0:
				dropped_ccs.append(cc_int)
				continue
			cchs = [(cc_int,cch[1]) for cch in cchs]
			all_cchs.update(cchs)
		# all_ccs = sorted(list(set([cch[0] for cch in all_cchs])))
		print(f"> dropped {len(dropped_ccs)} ccs: {dropped_ccs}")
		cch_to_idx = {}
		cc_to_cch_idx = {}
		for idx, cch in enumerate(all_cchs):
			cc, h = cch[0], cch[1]
			if cc not in cc_to_cch_idx:
				cc_to_cch_idx[cc] = [idx]
			else:
				cc_to_cch_idx[cc].append(idx)
			cch_to_idx[cch] = idx
		cch_g = nx.DiGraph()
		cch_g.add_nodes_from(list(cch_to_idx.values()))
		# set up edges
		for cch, src_idx in cch_to_idx.items():
			cc, h = cch[0], cch[1]
			cc_out_edges = cc_g.out_edges(cc)
			for cc_out_edge in cc_out_edges:
				src_cc = cc_out_edge[0]
				assert src_cc == cc, (src_cc,cc)
				dst_cc = cc_out_edge[1]
				for dst_idx in cc_to_cch_idx.get(dst_cc,tuple()):
					cch_g.add_edge(src_idx,dst_idx)
		return cc_g, cch_g
