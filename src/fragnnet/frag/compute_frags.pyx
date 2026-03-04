# cython: profile=False
# cython: linetrace=False
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
import signal
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

cimport numpy as cnp

# Global interrupt flag
interrupted = False

def signal_handler(signum, frame):
	global interrupted
	interrupted = True

# Register the handler
signal.signal(signal.SIGINT, signal_handler)

MAX_NUM_NODES = 128  # hard cap on 80!!
MAX_NUM_EDGES = 4 * MAX_NUM_NODES
MASK_DTYPE = np.uint8
MAX_EDGE_PER_NODE = 6  # Sulfur

class TimeoutError(RuntimeError):
	pass

cdef void np_fill(int [::1] array, int val) nogil:
	cdef Py_ssize_t i, num_elems = array.shape[0]
	for i in range(num_elems):
		array[i] = val

cdef void char_fill(char [::1] array, char val) nogil:
	cdef Py_ssize_t i, num_elems = array.shape[0]
	for i in range(num_elems):
		array[i] = val

cdef void char2d_fill(char [:,::1] array, char val) nogil:
	cdef Py_ssize_t i, j
	cdef Py_ssize_t rows = array.shape[0]
	cdef Py_ssize_t cols = array.shape[1]
	for i in range(rows):
		for j in range(cols):
			array[i, j] = val

cdef int np_sum(char [::1] array) nogil:
	cdef Py_ssize_t i, num_elems = array.shape[0]
	cdef int total = 0
	for i in range(num_elems):
		total += array[i]
	return total

cdef long get_time():
	cdef timespec ts
	cdef long current
	clock_gettime(CLOCK_REALTIME, &ts)
	current = ts.tv_sec
	return current

def mask_to_binary_hash(char [::1] mask):
	return np.asarray(mask).tobytes()

cdef int [:,::1] compute_node_to_edge_idx(int num_nodes, int num_edges, int [:,::1] edges):
	# use np fill instead of np.ones to speed up
	cdef int [:,::1] node_to_edge_idx = np.full((num_nodes, MAX_EDGE_PER_NODE), -1, dtype=np.intc)
	cdef int src_node, dst_node, j, k
	for j in range(num_edges):
		src_node = edges[j,0]
		dst_node = edges[j,1]
		for k in range(MAX_EDGE_PER_NODE):
			if node_to_edge_idx[src_node,k] == -1:
				node_to_edge_idx[src_node,k] = j
				break
		for k in range(MAX_EDGE_PER_NODE):
			if node_to_edge_idx[dst_node,k] == -1:
				node_to_edge_idx[dst_node,k] = j
				break
	return node_to_edge_idx

def py_compute_node_to_edge_idx(int num_nodes, int num_edges, int [:,::1] edges):
	return np.asarray(compute_node_to_edge_idx(num_nodes,num_edges,edges),dtype=np.intc)

cdef int connected_components(
		int num_nodes,
		int num_edges,
		char [::1] node_mask,
		int [:,::1] edges,
		char [::1] edge_mask,
		int [:,::1] node_to_edge_idx,
		int [::1] ws_visited,
		int [::1] ws_enqueued,
		int [::1] ws_queue,
		char [:,::1] ws_cc_nodes,
		char [:,::1] ws_cc_edges,
	) except -1:
	# Reset workspace buffers — replaces 5x np.zeros/np.full allocations per call.
	# ws_queue is a circular buffer: writes always precede reads, so no init needed.
	np_fill(ws_visited, 0)
	np_fill(ws_enqueued, 0)
	char2d_fill(ws_cc_nodes, 0)
	char2d_fill(ws_cc_edges, 0)

	cdef Py_ssize_t num_visited = 0
	cdef Py_ssize_t num_nonmask = np_sum(node_mask)
	cdef Py_ssize_t cur_node
	cdef Py_ssize_t num_ccs = 0
	cdef Py_ssize_t start_queue = 0
	cdef Py_ssize_t end_queue = 0
	cdef Py_ssize_t cur_edge
	cdef Py_ssize_t i
	cdef Py_ssize_t j
	cdef Py_ssize_t neighbour

	# run BFS
	while num_visited < num_nonmask:
		if start_queue != end_queue:
			# dequeue
			cur_node = ws_queue[start_queue]
			start_queue = (start_queue + 1) % num_nodes
			ws_visited[cur_node] = 1
			num_visited += 1
			ws_cc_nodes[num_ccs-1,cur_node] = 1
			# add neighbours
			for j in range(MAX_EDGE_PER_NODE):
				cur_edge = node_to_edge_idx[cur_node,j]
				# check if null
				if cur_edge == -1:
					break
				# check if edge is masked
				if edge_mask[cur_edge] == 0:
					continue
				# add unvisited neighbours
				if edges[cur_edge,0] == cur_node:
					neighbour = edges[cur_edge,1]
				else:
					neighbour = edges[cur_edge,0]

				# if we have not seen node before
				if ws_enqueued[neighbour] == 0:
					ws_queue[end_queue] = neighbour
					end_queue = (end_queue + 1) % num_nodes
					ws_enqueued[neighbour] = 1

				# add edge to cc
				ws_cc_edges[num_ccs-1,cur_edge] = 1

		else:
			# check for new nodes to seed BFS
			for i in range(num_nodes):
				if node_mask[i] == 1 and ws_visited[i] == 0:
					# enqueue
					ws_queue[end_queue] = i
					end_queue = (end_queue + 1) % num_nodes
					ws_enqueued[i] = 1
					num_ccs += 1
					break

	if num_visited != num_nonmask:
		raise ValueError(f'num_visited ({num_visited}) != num_nonmask ({num_nonmask})')

	if start_queue != end_queue:
		raise ValueError(f'start_queue ({start_queue}) != end_queue ({end_queue}), num_ccs {num_ccs}')

	if num_ccs < 0 or num_ccs > 2:
		raise ValueError(f'{num_ccs} num_ccs <0 or num_ccs > 2')

	return num_ccs

def compute_ccs(
		int num_nodes,
		int num_edges,
		char [::1] node_mask,
		int [:,::1] edges,
		char [::1] edge_mask,
		int [:,::1] node_to_edge_idx,
		int max_depth,
		long time_limit
	):

	# and a time check
	assert num_nodes <= MAX_NUM_NODES, num_nodes
	assert num_edges <= MAX_NUM_EDGES, num_edges
	cdef long start_time = get_time()

	# this need happen within loop
	cdef long cur_time = 0

	# Pre-allocate workspace buffers once — shared across all connected_components calls.
	# Eliminates 5x numpy allocations per edge per fragment (O(E * N_frags) total).
	cdef int [::1] ws_visited  = np.empty(num_nodes, dtype=np.intc)
	cdef int [::1] ws_enqueued = np.empty(num_nodes, dtype=np.intc)
	cdef int [::1] ws_queue    = np.empty(num_nodes, dtype=np.intc)
	cdef char [:,::1] ws_cc_nodes = np.empty((2, num_nodes), dtype=MASK_DTYPE)
	cdef char [:,::1] ws_cc_edges = np.empty((2, num_edges), dtype=MASK_DTYPE)

	# define a load of things
	cdef list child_node_mask_list = []
	cdef list child_edge_mask_list = []
	cdef int node_idx
	cdef int num_edges_nomask = 0

	cdef int ccs_idx = 0
	cdef int ccs_end = 0
	cdef int c = 0
	cdef int i = 0
	cdef int current_depth = 0

	# start with return_parents
	node_mask_key = mask_to_binary_hash(node_mask)
	edge_mask_key = mask_to_binary_hash(edge_mask)

	cdef list ccs = [node_mask_key]
	cdef list cc_edges = [edge_mask_key]
	cdef list ccs_depths = [0]
	cdef dict cc_node_mask_dict = {node_mask_key:node_mask}
	cdef dict cc_edge_mask_dict = {edge_mask_key:edge_mask}
	# Pre-built list of active edge indices per fragment: avoids scanning all num_edges
	# each iteration — inner loop runs O(active_edges) instead of O(total_edges).
	cdef dict cc_edge_active_dict = {edge_mask_key:[i for i in range(num_edges) if edge_mask[i] == 1]}
	cdef dict has_seen_dict = {}
	cdef dict ccs_depth_dict = {node_mask_key:set({0})}
	cdef int num_ccs = 0

	# data for graphs
	cdef dict css_to_id_dict = {node_mask_key:0}
	# needed by pyg
	# dag_edge_dict, used to keep track dag edges and min depth
	dag_edge_dict = {} #set({})

	force_stop = False
	reached_depth = 0
	# for each depth
	for current_depth in range(max_depth):
		# keep all the ccs mask for next round
		child_edge_mask_list = []
		ccs_end = len(ccs)

		for node_idx in range(ccs_idx, ccs_end):
			# let us check time
			cur_time = get_time()
			if interrupted:
				raise KeyboardInterrupt("Execution interrupted by user (Ctrl+C)")

			if cur_time - start_time > time_limit:
				force_stop = True
				break

			# get mask hash and mask
			curent_node_mask_key = ccs[node_idx]
			current_edge_mask_key = cc_edges[node_idx]
			current_node_mask = cc_node_mask_dict[curent_node_mask_key]
			current_edge_mask = cc_edge_mask_dict[current_edge_mask_key]

			# Use tuple key instead of bytes concatenation to avoid ~150-byte copy per iteration
			node_full_mask_key = (curent_node_mask_key, current_edge_mask_key)

			# check if we have seen this configuration before
			need_skip = False
			if node_full_mask_key not in has_seen_dict:
				has_seen_dict[node_full_mask_key] = current_depth
			elif has_seen_dict[node_full_mask_key] > current_depth:
				has_seen_dict[node_full_mask_key] = current_depth
			else:
				need_skip = True

			# active_edges is pre-built: iterate only live bonds, skip np_sum scan
			active_edges = cc_edge_active_dict[current_edge_mask_key]
			if need_skip or len(active_edges) == 0:
				continue

			for i in active_edges:
				# mask edge, kill bond
				current_edge_mask[i] = 0
				# compute ccs — returns plain int; results written into ws_cc_nodes/ws_cc_edges
				new_num_ccs = connected_components(
					num_nodes,
					num_edges,
					current_node_mask,
					edges,
					current_edge_mask,
					node_to_edge_idx,
					ws_visited,
					ws_enqueued,
					ws_queue,
					ws_cc_nodes,
					ws_cc_edges,
				)

				# restore edge
				current_edge_mask[i] = 1
				# save data and save edge and node configure
				for c in range(new_num_ccs):
					# save cc_nodes
					child_node_mask_key = np.asarray(ws_cc_nodes[c,:]).tobytes()
					child_edge_mask_key = np.asarray(ws_cc_edges[c,:]).tobytes()

					# add to dict to track node and edges
					if child_node_mask_key not in cc_node_mask_dict:
						# Must copy: workspace buffer is overwritten on the next connected_components call
						cc_node_mask_dict[child_node_mask_key] = np.array(ws_cc_nodes[c,:], dtype=MASK_DTYPE)
						# track ccs to idx
						css_to_id_dict[child_node_mask_key] = len(css_to_id_dict)

					if child_edge_mask_key not in cc_edge_mask_dict:
						# Must copy: workspace buffer is overwritten on the next connected_components call
						cc_edge_mask_dict[child_edge_mask_key] = np.array(ws_cc_edges[c,:], dtype=MASK_DTYPE)
						# Build active-edge list for this new fragment from the workspace result
						cc_edge_active_dict[child_edge_mask_key] = [
							j for j in range(num_edges) if ws_cc_edges[c, j] == 1
						]

					# track depth for each cc node
					if child_node_mask_key not in ccs_depth_dict:
						ccs_depth_dict[child_node_mask_key] = set([current_depth + 1])
					else:
						ccs_depth_dict[child_node_mask_key].add(current_depth + 1)

					ccs.append(child_node_mask_key)
					cc_edges.append(child_edge_mask_key)

					# add edge from node to node if mask are different
					# NOTE: it is possiable we have parent and child with the same mask
					# eg. first bond break on a ring
					if curent_node_mask_key != child_node_mask_key:

						edge_pair = (css_to_id_dict[curent_node_mask_key],css_to_id_dict[child_node_mask_key])
						if edge_pair not in dag_edge_dict:
							dag_edge_dict[edge_pair] = current_depth + 1

				num_ccs += new_num_ccs
				reached_depth = current_depth + 1
		# break outer loop
		if force_stop:
			# Timeout occurred during current_depth processing
			# Discard all incomplete work from current_depth and keep only fully completed depths
			# The last fully completed depth is (current_depth - 1)
			reached_depth = current_depth - 1 if current_depth > 0 else 0

			# Filter ccs_depth_dict: keep only nodes with at least one depth <= reached_depth
			filtered_ccs_depth_dict = {}
			for k, v in ccs_depth_dict.items():
				filtered_depths = set([i for i in v if i <= reached_depth])
				if filtered_depths:  # Only include if non-empty
					filtered_ccs_depth_dict[k] = filtered_depths
			ccs_depth_dict = filtered_ccs_depth_dict

			# Filter css_to_id_dict: keep only nodes that are in filtered_ccs_depth_dict
			filtered_css_to_id_dict = {cc: css_to_id_dict[cc] for cc in filtered_ccs_depth_dict}

			# Build reverse mapping (id -> cc) to validate edge endpoints
			id_to_css = {v: k for k, v in filtered_css_to_id_dict.items()}

			# Filter dag_edge_dict: keep only edges where BOTH endpoints exist in filtered nodes
			# AND the edge depth is <= reached_depth
			filtered_dag_edge_dict = {}
			for (src_id, dst_id), depth in dag_edge_dict.items():
				if depth <= reached_depth and src_id in id_to_css and dst_id in id_to_css:
					filtered_dag_edge_dict[(src_id, dst_id)] = depth
			dag_edge_dict = filtered_dag_edge_dict
			css_to_id_dict = filtered_css_to_id_dict
			break
		# finishing current depth, move to next
		ccs_idx = ccs_end

	# build graph
	num_un_ccs = len(css_to_id_dict)
	# create sorted node matrix with masks
	cdef list ordered_ccs = [0] * num_un_ccs
	for cc in css_to_id_dict:
		ordered_ccs[css_to_id_dict[cc]] = cc

	# create node masks — np.array is 2.4x faster than np.stack for list-of-arrays
	nodes_mask_matrix = np.array([np.asarray(cc_node_mask_dict[cc]) for cc in ordered_ccs], dtype=MASK_DTYPE)

	# create node depth matrix
	nodes_depth_matrix = np.zeros((num_un_ccs, max_depth + 1),dtype = MASK_DTYPE)
	nodes_min_depth = np.zeros((num_un_ccs),dtype = MASK_DTYPE)

	for idx,cc in enumerate(ordered_ccs):
		for d_id in ccs_depth_dict[cc]:
			nodes_depth_matrix[idx,d_id] = 1
		# set lowest depth on the dag for each node
		nodes_min_depth[idx] = min(ccs_depth_dict[cc])

	# create edge set:
	dag_edges_matrix = np.asarray(list(dag_edge_dict.keys()), dtype = np.int64)
	edges_min_depth = np.asarray(list(dag_edge_dict.values()), dtype = MASK_DTYPE)
	dag_frag_meta = {"reached_depth": reached_depth, "edges_min_depth": edges_min_depth, "nodes_min_depth": nodes_min_depth, "force_stopped": force_stop}
	# both edges_min_depth and node_min_depth should be ordered

	return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta

def compute_cc_h_floor(
		cnp.ndarray[cnp.int32_t, ndim=1] cc_atom_ids,
		cnp.ndarray[cnp.int32_t, ndim=1] ve_arr,
		cnp.ndarray[cnp.int32_t, ndim=1] sbond_arr,
		int num_radicals,
		cnp.ndarray[cnp.int32_t, ndim=2] bonds,
		dict atoms_to_bonds,
		cnp.ndarray[cnp.uint8_t, ndim=1] bond_mask_arr):  # Fix: uint8_t for boolean array

	"""Compute the minimum number of Hs a connected component (cc) can have."""

	assert num_radicals == 0

	# Compute the difference array (initial hydrogen deficit)P
	cdef cnp.ndarray[cnp.int32_t, ndim=1] diff_arr = np.maximum(ve_arr - sbond_arr, 0)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] h_arr = diff_arr.copy()

	# Define Cython integer variables
	cdef Py_ssize_t atom, bond_idx, other
	cdef list bond_idxs
	cdef tuple bond

	# Iterate over atoms in the connected component
	for atom in cc_atom_ids:
		bond_idxs = atoms_to_bonds[atom]  # List of bond indices for this atom

		for bond_idx in bond_idxs:
			if h_arr[atom] == 0:
				break
			if bond_mask_arr[bond_idx] == 0:  # Use explicit comparison for uint8_t
				continue

			other = bonds[bond_idx, 1] if bonds[bond_idx, 0] == atom else bonds[bond_idx, 0]

			# Ensure we don't form more than 3 bonds
			h_arr[atom] = max(0, h_arr[atom] - min(diff_arr[other], 2))

	# Compute the lower bound of hydrogen count
	cdef int cc_floor = max(h_arr[cc_atom_ids].sum() - num_radicals, 0)

	return cc_floor

def update_bonds(
		cnp.ndarray[cnp.int32_t, ndim=1] cc_atom_ids,
		cnp.ndarray[cnp.int32_t, ndim=1] sbond_arr,
		cnp.ndarray[cnp.uint8_t, ndim=1] bond_mask_arr,  # Using uint8_t for boolean
		cnp.ndarray[cnp.int32_t, ndim=2] bonds,
		dict atoms_to_bonds):

	"""Updates single bond counts and bond masks for a given connected component."""

	# Create new arrays with the same shape as sbond_arr and bond_mask_arr
	cdef cnp.ndarray[cnp.int32_t, ndim=1] new_sbond_arr = np.zeros_like(sbond_arr, dtype=np.int32)
	cdef cnp.ndarray[cnp.uint8_t, ndim=1] new_bond_mask_arr = np.zeros_like(bond_mask_arr, dtype=np.uint8)

	# Boolean lookup array: O(1) membership test instead of O(n) linear scan per atom
	cdef Py_ssize_t num_atoms = sbond_arr.shape[0]
	cdef cnp.ndarray[cnp.uint8_t, ndim=1] in_cc = np.zeros(num_atoms, dtype=np.uint8)
	cdef Py_ssize_t k
	for k in range(cc_atom_ids.shape[0]):
		in_cc[cc_atom_ids[k]] = 1

	# Define Cython integer variables
	cdef Py_ssize_t atom, other, bond_idx, i, j
	cdef list bond_idxs

	# Iterate through each atom in the connected component
	for i in range(cc_atom_ids.shape[0]):
		atom = cc_atom_ids[i]
		bond_idxs = atoms_to_bonds[atom]  # Get bond indices for this atom

		for j in range(len(bond_idxs)):
			bond_idx = bond_idxs[j]

			# Get bond endpoints
			other = bonds[bond_idx, 1] if bonds[bond_idx, 0] == atom else bonds[bond_idx, 0]

			# O(1) lookup instead of O(n) linear scan through cc_atom_ids
			if in_cc[other]:
				new_sbond_arr[atom] += 1
				new_bond_mask_arr[bond_idx] = 1  # Use 1 instead of True for uint8_t

	return new_sbond_arr, new_bond_mask_arr
