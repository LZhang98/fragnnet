# cython: profile=False
# cython: linetrace=False
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
import os
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

#cdef long node_mask_to_int(int num_nodes, char [::1] node_mask):

#	assert MAX_NUM_NODES < 64, MAX_NUM_NODES
#	cdef long node_mask_int = 0
#	cdef int i
#	for i in range(num_nodes):
#		assert node_mask[i] >= 0, node_mask[i]
#		node_mask_int += node_mask[i]*(2**i)
#	assert node_mask_int >= 0, node_mask_int
#	return node_mask_int

def mask_to_binary_hash(char [::1] mask):
	return np.asarray(mask).tobytes()


cdef object mask_to_int_memview(char [::1] arr):
	# pack a 0/1 mask from a memoryview into a Python int (up to 128 bits)
	# Use Python int throughout to avoid C sign extension issues
	cdef Py_ssize_t n = arr.shape[0]
	cdef Py_ssize_t i
	cdef object result = 0  # Python int
	cdef object one = 1     # Python int

	for i in range(n):
		if arr[i]:
			result |= (one << i)

	return result

def py_mask_to_int_memview(mask):
	"""Python wrapper for testing mask_to_int_memview.

	Converts input to uint8 array and calls the cdef version.
	"""
	cdef cnp.ndarray[cnp.uint8_t, ndim=1] arr = np.asarray(mask, dtype=np.uint8)
	return mask_to_int_memview(arr)

cdef int [:,::1] compute_node_to_edge_idx(int num_nodes, int num_edges, int [:,::1] edges):
	# use np fill instead of np.ones to speed up
	cdef int [:,::1] node_to_edge_idx = np.full((num_nodes, MAX_EDGE_PER_NODE), -1, dtype=np.intc)
	cdef int src_node, dst_node, j, k
	for j in range(num_edges):
		src_node = edges[j,0]
		dst_node = edges[j,1]
		#assert src_node < num_nodes and dst_node < num_nodes
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

cdef connected_components(
		int num_nodes,
		int num_edges,
		char [::1] node_mask,
		int [:,::1] edges,
		char [::1] edge_mask,
		int [:,::1] node_to_edge_idx
	):
	# print('>connected_components')
	# return the number of connected components
	cdef Py_ssize_t num_visited = 0
	cdef Py_ssize_t num_nonmask = np_sum(node_mask)
	cdef Py_ssize_t cur_node
	cdef Py_ssize_t num_ccs = 0
	cdef Py_ssize_t start_queue = 0
	cdef Py_ssize_t end_queue = 0
	cdef Py_ssize_t queue_size = 0
	cdef Py_ssize_t cur_edge
	cdef Py_ssize_t i
	cdef Py_ssize_t j
	cdef Py_ssize_t neighbour

	cdef int [::1] node_visited = np.zeros(num_nodes,dtype=np.intc)
	cdef int [::1] node_enqueued = np.zeros(num_nodes,dtype=np.intc)
	cdef int [::1] node_queue = np.full(num_nodes, -1, dtype=np.intc)

	cdef list comp_nodes = None

	# allocate with safe capacity: worst-case each node is its own component
	# using num_nodes rows prevents out-of-bounds writes when unexpected
	# logic paths create many components (observed in logs).
	# Keep the ndarray objects alive by holding references; memoryviews
	# alone may not own the underlying buffer, causing use-after-free when
	# returning views. Allocate ndarrays and create memoryviews that point
	# to them; return the ndarrays below so Python owns the buffers.
	cdef cnp.ndarray[cnp.uint8_t, ndim=2] cc_nodes_arr = np.zeros((num_nodes, num_nodes), dtype=MASK_DTYPE)
	cdef cnp.ndarray[cnp.uint8_t, ndim=2] cc_edges_arr = np.zeros((num_nodes, num_edges), dtype=MASK_DTYPE)
	cdef char[:,::1] cc_nodes = cc_nodes_arr
	cdef char[:,::1] cc_edges = cc_edges_arr

	# run BFS
	while num_visited < num_nonmask:
		if start_queue != end_queue:
			# dequeue
			cur_node = node_queue[start_queue]
			start_queue = (start_queue + 1) % num_nodes
			# maintain queue size
			if queue_size > 0:
				queue_size -= 1
			# mark as visited
			#assert node_enqueued[cur_node] == 1, cur_node
			#assert node_visited[cur_node] == 0, cur_node
			#assert node_mask[cur_node] == 1, cur_node
			node_visited[cur_node] = 1
			num_visited += 1
			# buffer node for this component; flush into cc_nodes when component completes
			if comp_nodes is None:
				# defensive fallback: ensure buffer exists
				comp_nodes = []
			comp_nodes.append(cur_node)
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
					#assert edges[cur_edge,1] == cur_node
					neighbour = edges[cur_edge,0]

				# if we have not seen node before and it is in the active mask
				if node_enqueued[neighbour] == 0 and node_mask[neighbour] == 1:
					# guard queue overflow when enqueuing neighbours
					if queue_size >= num_nodes:
						try:
							with open('frag_debug.log', 'a') as _f:
								_f.write(f'CONNECTED_COMPONENTS_QUEUE_OVERFLOW num_nodes={num_nodes} num_ccs={num_ccs}\n')
						except Exception:
							pass
						raise ValueError('queue overflow in connected_components')
					node_queue[end_queue] = neighbour
					end_queue = (end_queue + 1) % num_nodes
					node_enqueued[neighbour] = 1
					# maintain queue size
					queue_size += 1

				# add edge to cc
				cc_edges[num_ccs-1,cur_edge] = 1

		else:
			# if we have just finished a component (queue empty) flush buffered nodes
			if comp_nodes is not None:
				for _n in comp_nodes:
					cc_nodes[num_ccs-1, _n] = 1
				comp_nodes = None
			# check for new nodes to seed BFS
			for i in range(num_nodes):
				if node_mask[i] == 1 and node_visited[i] == 0:
					# enqueue (seed new BFS). Guard queue overflow.
					if queue_size >= num_nodes:
						try:
							with open('frag_debug.log', 'a') as _f:
								_f.write(f'CONNECTED_COMPONENTS_QUEUE_OVERFLOW num_nodes={num_nodes} num_ccs={num_ccs}\n')
						except Exception:
							pass
						raise ValueError('queue overflow in connected_components')
					node_queue[end_queue] = i
					end_queue = (end_queue + 1) % num_nodes
					#print(f"new end node {end_queue}")
					#assert node_enqueued[i] == 0
					node_enqueued[i] = 1
					queue_size += 1
					# start a fresh per-component buffer for nodes
					comp_nodes = []
					# safety: guard against runaway num_ccs (should be <= num_nodes)
					if num_ccs >= num_nodes:
						try:
							with open('frag_debug.log', 'a') as _f:
								_f.write(f'CONNECTED_COMPONENTS_OVERFLOW num_nodes={num_nodes} num_ccs={num_ccs}\n')
						except Exception:
							pass
						raise ValueError(f'unexpected num_ccs {num_ccs} >= {num_nodes}')
					num_ccs += 1
					break
	
	if num_visited != num_nonmask:
		raise ValueError(f'num_visited ({num_visited}) != num_nonmask ({num_nonmask})')
	
	# Flush any remaining buffered nodes for the final component
	if comp_nodes is not None and num_ccs > 0:
		for _n in comp_nodes:
			cc_nodes[num_ccs-1, _n] = 1
	
	if start_queue != end_queue:
		raise ValueError(f'start_queue ({start_queue}) != end_queue ({end_queue}), num_ccs {num_ccs}')

	# allow up to num_nodes connected components (safe upper bound)
	if num_ccs < 0 or num_ccs > num_nodes:
		# persist a detailed debug record before raising so we can inspect later
		try:
			with open('frag_debug.log', 'a') as _f:
				_f.write(f'CONNECTED_COMPONENTS_ERROR num_nodes={num_nodes} num_edges={num_edges} num_nonmask={num_nonmask} num_visited={num_visited} num_ccs={num_ccs}\n')
				try:
					_f.write('node_mask_sample=' + np.asarray(node_mask[:min(num_nodes,64)]).tobytes().hex() + '\n')
				except Exception:
					_f.write('node_mask_sample=FAILED\n')
				try:
					_f.write('edge_mask_sample=' + np.asarray(edge_mask[:min(num_edges,64)]).tobytes().hex() + '\n')
				except Exception:
					_f.write('edge_mask_sample=FAILED\n')
		except Exception:
			# best effort logging; ignore write errors
			pass
		raise ValueError(f'{num_ccs} num_ccs <0 or num_ccs > {num_nodes}')

	# Return the ndarray objects to ensure the buffers remain owned by Python
	return cc_nodes_arr, cc_edges_arr, num_ccs

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
	# use memoryview-based packing to avoid temporary numpy allocations
	node_mask_key = mask_to_int_memview(node_mask)
	edge_mask_key = mask_to_int_memview(edge_mask)

	# Preallocate lists to avoid repeated append/resizing in hot loops
	cdef int est_per_edge = max_depth + 1
	cdef Py_ssize_t est_capacity = 1 + (num_edges * est_per_edge * 2)
	cdef list ccs = [None] * est_capacity
	ccs[0] = node_mask_key
	cdef list cc_edges = [None] * est_capacity
	cc_edges[0] = edge_mask_key
	cdef list ccs_depths = [0]
	cdef Py_ssize_t ccs_len = 1
	cdef dict cc_node_mask_dict = {node_mask_key:node_mask}
	cdef dict cc_edge_mask_dict = {edge_mask_key:edge_mask}
	cdef dict has_seen_dict = {}
	# store depths as a bitmask integer: bit d set means node exists at depth d
	cdef dict ccs_depth_dict = {node_mask_key: (1 << 0)}
	cdef int num_ccs = 0
	
	# data for gaphs
	cdef dict css_to_id_dict = {node_mask_key:0}
	# needed by pyg
	# dag_edge_dict, used to keep track dag edges and min depth
	# Collect dag edges in preallocated lists; dedupe after nogil work
	dag_edge_dict = {} # final mapping (built later)
	cdef list dag_src = [0] * (num_edges * est_per_edge + 4)
	cdef list dag_dst = [0] * (num_edges * est_per_edge + 4)
	cdef list dag_depths = [0] * (num_edges * est_per_edge + 4)
	cdef Py_ssize_t dag_len = 0

	#print("test_list", test_list[0])
	force_stop = False
	reached_depth = 0
	# for each depth
	for current_depth in range(max_depth):
		# keep all the ccs mask for next round
		child_edge_mask_list = []		
		ccs_end = ccs_len
		
		for node_idx in range(ccs_idx, ccs_end):
			# let us check time
			cur_time = get_time()
			#sig_check()
			if interrupted:
				raise KeyboardInterrupt("Execution interrupted by user (Ctrl+C)")

			if cur_time - start_time > time_limit:
				force_stop = True
				#print(f"Timeout reached, {time_limit} seconds. Stopping at depth {current_depth}.")
				break
			
			# get mask hash and mask
			curent_node_mask_key = ccs[node_idx]
			current_edge_mask_key = cc_edges[node_idx]
			current_node_mask = cc_node_mask_dict[curent_node_mask_key]
			current_edge_mask = cc_edge_mask_dict[current_edge_mask_key]

			# let us check mask; there is more than one way to get each node
			# thus, we need to check if we have seen this node mask and edge mask before
			# pack node+edge into a single int key to avoid tuple allocations
			# use 128-bit shift (masks support up to 128 bits)
			node_full_mask = ( (curent_node_mask_key << 128) | current_edge_mask_key )

			# check if we have seen this configuration before (store min depth)
			need_skip = False
			if node_full_mask not in has_seen_dict:
				has_seen_dict[node_full_mask] = current_depth
			elif has_seen_dict[node_full_mask] > current_depth:
				has_seen_dict[node_full_mask] = current_depth
			else:
				need_skip = True

			# check if there any bond left to kill
			num_edges_nomask = np_sum(current_edge_mask)
			if need_skip or num_edges_nomask == 0:
				continue

			# now we need to break more bond
			#i = 0
			for i in range(num_edges):
				# check if edge is already masked
				# sorry Mr Bond, you already dead
				if current_edge_mask[i] == 0:
					continue
				# mask edge, kill bond
				current_edge_mask[i] = 0
				# compute ccs
				new_cc_nodes, new_cc_edges, new_num_ccs = connected_components(
					num_nodes,
					num_edges,
					current_node_mask,
					edges,
					current_edge_mask,
					node_to_edge_idx
				)

				# persist context for post-mortem debugging
				try:
					with open('frag_debug.log', 'a') as _f:
						_f.write(f'CALL_CONNECTED_COMPONENTS depth={current_depth} node_idx={node_idx} edge_i={i} new_num_ccs={new_num_ccs} num_nodes={num_nodes} num_edges={num_edges}\n')
						try:
							_f.write('parent_node_mask_sum=' + str(int(np.sum(current_node_mask))) + ' parent_edge_mask_sum=' + str(int(np.sum(current_edge_mask))) + '\n')
						except Exception:
							_f.write('mask_sums_failed\n')
				except Exception:
					# ignore logging errors
					pass

				# Temporary guard: if connected_components returns an unexpectedly large
				# number of components (likely a corruption or unexpected case), skip
				# processing this edge and continue; this avoids array OOB later and
				# allows gathering more logs for diagnosis.
				MAX_ALLOWED_CCS = 8
				if new_num_ccs < 0 or new_num_ccs > MAX_ALLOWED_CCS:
					try:
						with open('frag_debug.log', 'a') as _f:
							_f.write(f'WARNING_SKIP large_new_num_ccs depth={current_depth} node_idx={node_idx} edge_i={i} new_num_ccs={new_num_ccs}\n')
					except Exception:
						pass
					# restore edge mask and continue
					current_edge_mask[i] = 1
					continue
				# Debugging: if FRAG_DEBUG env var set, print returned shapes and counts
				if os.getenv('FRAG_DEBUG') == '1':
					try:
						print(f"DEBUG connected_components: depth={current_depth} node_idx={node_idx} edge_i={i} new_num_ccs={new_num_ccs}")
						print(f"DEBUG new_cc_nodes.shape={getattr(new_cc_nodes, 'shape', None)} new_cc_edges.shape={getattr(new_cc_edges, 'shape', None)}")
						# print masks for inspection
						for cc_i in range(new_num_ccs):
							print(f"DEBUG cc[{cc_i}] nodes_sum={int(np.sum(new_cc_nodes[cc_i,:]))} edges_sum={int(np.sum(new_cc_edges[cc_i,:]))}")
					except Exception as e:
						print('DEBUG logging failed:', e)

				# restore edge
				current_edge_mask[i] = 1
				# save data and save edge and node configure
				for c in range(new_num_ccs):
					# save cc_nodes
					child_node_mask_key = mask_to_int_memview(new_cc_nodes[c,:])
					child_edge_mask_key = mask_to_int_memview(new_cc_edges[c,:])

					# add to dict to track node and edges
					if child_node_mask_key not in cc_node_mask_dict:
						# store a copy to ensure the mask memory outlives the local
						# array returned from connected_components
						cc_node_mask_dict[child_node_mask_key] = np.asarray(new_cc_nodes[c,:]).copy()
						# track ccs to idx 
						css_to_id_dict[child_node_mask_key] = len(css_to_id_dict)
						
					if child_edge_mask_key not in cc_edge_mask_dict:
						cc_edge_mask_dict[child_edge_mask_key] = np.asarray(new_cc_edges[c,:]).copy()

					# track depth for each cc node as a bitmask
					if child_node_mask_key not in ccs_depth_dict:
						ccs_depth_dict[child_node_mask_key] = (1 << (current_depth + 1))
					else:
						ccs_depth_dict[child_node_mask_key] |= (1 << (current_depth + 1))

					# append to preallocated lists
					if ccs_len >= est_capacity:
						# fallback to append (unlikely) if we exceed estimate
						ccs.append(child_node_mask_key)
						cc_edges.append(child_edge_mask_key)
						ccs_len += 1
					else:
						ccs[ccs_len] = child_node_mask_key
						cc_edges[ccs_len] = child_edge_mask_key
						ccs_len += 1
					
					# add edge from node to node if mask are different
					# NOTE: it is possiable we have parent and child with the same mask
					# eg. first bond break on a ring
					if curent_node_mask_key != child_node_mask_key:
						# record dag edge in preallocated lists; dedupe later
						src_id = css_to_id_dict[curent_node_mask_key]
						dst_id = css_to_id_dict[child_node_mask_key]
						if dag_len >= len(dag_src):
							dag_src.append(src_id); dag_dst.append(dst_id); dag_depths.append(current_depth + 1)
							dag_len += 1
						else:
							dag_src[dag_len] = src_id
							dag_dst[dag_len] = dst_id
							dag_depths[dag_len] = current_depth + 1
							dag_len += 1
					
				num_ccs += new_num_ccs
				reached_depth = current_depth + 1
		# print(dag_edge_dict)
		# break outer loop
		if force_stop:
			# Timeout occurred during current_depth processing
			# Discard all incomplete work from current_depth and keep only fully completed depths
			# The last fully completed depth is (current_depth - 1)
			reached_depth = current_depth - 1 if current_depth > 0 else 0
			# Truncate any provisional CC entries collected for this depth
			ccs_len = ccs_end
			
			# Filter ccs_depth_dict: keep only nodes with at least one depth <= reached_depth
			filtered_ccs_depth_dict = {}
			# v is a bitmask; keep only bits <= reached_depth
			mask_limit = (1 << (reached_depth + 1)) - 1
			for k, v in ccs_depth_dict.items():
				filtered = v & mask_limit
				if filtered:
					filtered_ccs_depth_dict[k] = filtered
			ccs_depth_dict = filtered_ccs_depth_dict
			
			# Filter css_to_id_dict: keep only nodes that are in filtered_ccs_depth_dict
			filtered_css_to_id_dict = {cc: css_to_id_dict[cc] for cc in filtered_ccs_depth_dict}
			
			# Build reverse mapping (id -> cc) to validate edge endpoints
			id_to_css = {v: k for k, v in filtered_css_to_id_dict.items()}
			
			# Build dag_edge_dict from collected lists (take min depth for duplicates)
			built_dag = {}
			for edge_idx in range(dag_len):
				_p = (dag_src[edge_idx], dag_dst[edge_idx])
				_d = dag_depths[edge_idx]
				built_dag[_p] = _d
			# Filter built_dag: keep only edges where BOTH endpoints exist in filtered nodes
			# AND the edge depth is <= reached_depth
			filtered_dag_edge_dict = {}
			for (src_id, dst_id), depth in built_dag.items():
				if depth <= reached_depth and src_id in id_to_css and dst_id in id_to_css:
					filtered_dag_edge_dict[(src_id, dst_id)] = depth
			dag_edge_dict = filtered_dag_edge_dict
			css_to_id_dict = filtered_css_to_id_dict
			# print(f"Timeout reached. Stopping at depth {reached_depth}.")
			break
		# finishing current depth, move to next
		ccs_idx = ccs_end

	# build graph
	num_un_ccs = len(css_to_id_dict)

	# Build final dag_edge_dict from collected lists (take min depth for duplicates)
	if dag_len > 0:
		built_dag = {}
		for __i in range(dag_len):
			_p = (dag_src[__i], dag_dst[__i])
			_d = dag_depths[__i]
			if _p not in built_dag or built_dag[_p] > _d:
				built_dag[_p] = _d
		dag_edge_dict = built_dag
	# create sorted node matirx with masks
	cdef list ordered_ccs = [0] * num_un_ccs
	for cc in css_to_id_dict:
		ordered_ccs[css_to_id_dict[cc]] = cc

	# create node masks using a preallocated array and direct copies to avoid np.stack
	cdef int nn_nodes = num_nodes
	nodes_mask_matrix = np.empty((num_un_ccs, nn_nodes), dtype = MASK_DTYPE)
	for idx, cc in enumerate(ordered_ccs):
		nodes_mask_matrix[idx, : ] = cc_node_mask_dict[cc]

	# create node depth matrix
	nodes_depth_matrix = np.zeros((num_un_ccs, max_depth + 1),dtype = MASK_DTYPE)
	nodes_min_depth = np.zeros((num_un_ccs),dtype = MASK_DTYPE)

	for idx,cc in enumerate(ordered_ccs):
		v = ccs_depth_dict[cc]
		# iterate bits in bitmask v
		for d_id in range(max_depth + 1):
			if v & (1 << d_id):
				nodes_depth_matrix[idx,d_id] = 1
		# set lowest depth on the dag for each node (find lowest set bit)
		lowest = None
		for d_id in range(max_depth + 1):
			if v & (1 << d_id):
				lowest = d_id
				break
		if lowest is None:
			nodes_min_depth[idx] = 0
		else:
			nodes_min_depth[idx] = lowest

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

			#bond = bonds[bond_idx]
			#other = bond[1] if bond[0] == atom else bond[0]
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

	# Define Cython integer variables
	cdef Py_ssize_t  atom, other, bond_idx, i, j
	cdef list bond_idxs

	# Iterate through each atom in the connected component
	for i in range(cc_atom_ids.shape[0]):
		atom = cc_atom_ids[i]
		bond_idxs = atoms_to_bonds[atom]  # Get bond indices for this atom

		for j in range(len(bond_idxs)):
			bond_idx = bond_idxs[j]

			# Get bond endpoints
			other = bonds[bond_idx, 1] if bonds[bond_idx, 0] == atom else bonds[bond_idx, 0]

			# Check if both atoms are in the connected component
			if other in cc_atom_ids:
				new_sbond_arr[atom] += 1
				new_bond_mask_arr[bond_idx] = 1  # Use 1 instead of True for uint8_t

	return new_sbond_arr, new_bond_mask_arr
