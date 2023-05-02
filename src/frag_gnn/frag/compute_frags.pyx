# cython: profile=False
# cython: linetrace=False
# cython: language_level=3
# cython: boundscheck=False

import numpy as np
from cysignals.signals cimport sig_check
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from cpython.ref cimport PyObject

MAX_NUM_NODES = 60 # hard cap on 64!!
MAX_NUM_EDGES = 150 # ~4*MAX_NUM_NODES 
DTYPE = np.intc
MASK_DTYPE = np.byte
MAX_EDGE_PER_NODE = 6 # Sulfur

class TimeoutError(RuntimeError):
	pass

cdef char [::1] np_mask_copy(char [::1] array):
	cdef int num_elems = array.shape[0]
	cdef char [::1] new_array = np.zeros(num_elems,MASK_DTYPE)
	for i in range(num_elems):
		new_array[i] = array[i]
	return new_array

cdef void np_fill(int [::1] array, int val):
	cdef int num_elems = array.shape[0]
	for i in range(num_elems):
		array[i] = val

cdef int np_sum(char [::1] array):
	cdef int num_elems = array.shape[0]
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

cdef long node_mask_to_int(int num_nodes, char [::1] node_mask):

	assert MAX_NUM_NODES < 64, MAX_NUM_NODES
	cdef long node_mask_int = 0
	cdef int i
	for i in range(num_nodes):
		assert node_mask[i] >= 0, node_mask[i]
		node_mask_int += node_mask[i]*(2**i)
	assert node_mask_int >= 0, node_mask_int
	return node_mask_int

cdef mask_to_str(int num, char [::1] mask):
	#Not fastest but it will do for now
	mask_str = ''
	node_mask_int = 0
	cdef int i
	for i in range(num):
		node_mask_int += mask[i]*(2**i)
	mask_str = str(node_mask_int)
	return mask_str

cdef int [:,::1] compute_node_to_edge_idx(int num_nodes, int num_edges, int [:,::1] edges):

	cdef int [:,::1] node_to_edge_idx = -np.ones((MAX_NUM_NODES,MAX_EDGE_PER_NODE),dtype=DTYPE)
	cdef int src_node, dst_node, j, k
	for j in range(num_edges):
		src_node = edges[j,0]
		dst_node = edges[j,1]
		assert src_node < num_nodes and dst_node < num_nodes
		for k in range(MAX_EDGE_PER_NODE):
			if node_to_edge_idx[src_node,k] == -1:
				node_to_edge_idx[src_node,k] = j
				break
		for k in range(MAX_EDGE_PER_NODE):
			if node_to_edge_idx[dst_node,k] == -1:
				node_to_edge_idx[dst_node,k] = j
				break
	return node_to_edge_idx

cdef char [::1] compute_full_cc(int num_nodes):
	cdef char [::1] cc_nodes = np.zeros(MAX_NUM_NODES,dtype=MASK_DTYPE)
	for i in range(num_nodes):
		cc_nodes[i] = 1
	return cc_nodes

def py_compute_node_to_edge_idx(int num_nodes, int num_edges, int [:,::1] edges):
	return np.asarray(compute_node_to_edge_idx(num_nodes,num_edges,edges),dtype=DTYPE)

cdef connected_components(
		int num_nodes,
		int num_edges,
		char [::1] node_mask,
		int [:,::1] edges,
		char [::1] edge_mask,
		int [:,::1] node_to_edge_idx
	):
	# return the number of connected components
	cdef int num_visited = 0
	cdef int num_nonmask = np_sum(node_mask)
	cdef int cur_node
	cdef int num_ccs = 0
	cdef int start_queue = 0
	cdef int end_queue = 0
	cdef int cur_edge
	cdef int i
	cdef int j
	cdef int neighbour

	cdef int [::1] node_visited = np.zeros(MAX_NUM_NODES,dtype=np.intc)
	cdef int [::1] node_enqueued = np.zeros(MAX_NUM_NODES,dtype=np.intc)
	cdef int [::1] node_queue = np.zeros(MAX_NUM_NODES,dtype=np.intc)
	np_fill(node_queue,-1)

	cdef char [:,::1] cc_nodes = np.zeros((2,MAX_NUM_NODES),dtype=MASK_DTYPE)
	cdef char [:,::1] cc_edges = np.zeros((2,MAX_NUM_EDGES),dtype=MASK_DTYPE)

	# run BFS
	while num_visited < num_nonmask:
		if start_queue != end_queue:
			# dequeue
			cur_node = node_queue[start_queue]
			start_queue = (start_queue + 1) % MAX_NUM_NODES
			# mark as visited
			assert node_enqueued[cur_node] == 1, cur_node
			assert node_visited[cur_node] == 0, cur_node
			node_visited[cur_node] = 1
			num_visited += 1
			# add node to cc
			assert num_ccs > 0 and num_ccs <= 2, num_ccs
			cc_nodes[num_ccs-1,cur_node] = 1
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
					assert edges[cur_edge,1] == cur_node
					neighbour = edges[cur_edge,0]
				if node_enqueued[neighbour] == 0:
					assert node_visited[neighbour] == 0
					node_queue[end_queue] = neighbour
					end_queue = (end_queue + 1) % MAX_NUM_NODES
					node_enqueued[neighbour] = 1
				# add edge to cc
				cc_edges[num_ccs-1,cur_edge] = 1
		else:
			# check for new nodes to seed BFS
			for i in range(num_nodes):
				if node_mask[i] == 1 and node_visited[i] == 0:
					# enqueue
					node_queue[end_queue] = i
					end_queue = (end_queue + 1) % MAX_NUM_NODES
					assert node_enqueued[i] == 0
					node_enqueued[i] = 1
					num_ccs += 1
					break
	assert num_visited == num_nonmask, (num_visited,num_nonmask)
	assert start_queue == end_queue, (start_queue,end_queue)
	assert num_ccs > 0 and num_ccs <= 2
	# assert num_visited == np_sum(node_visited[:num_nodes]), (num_visited,np_sum(node_visited[:num_nodes]))
	# assert np_sum(cc_nodes[0,:num_nodes])+np_sum(cc_nodes[1,:num_nodes]) == num_nonmask
	return cc_nodes, cc_edges, num_ccs

cdef compute_ccs_recursive(
		int num_nodes,
		int num_edges,
		char [::1] node_mask,
		int [:,::1] edges,
		char [::1] edge_mask,
		int [:,::1] node_to_edge_idx,
		int depth,
		int max_depth,
		bint parents,
		bint depths,
		long time_limit,
		long start_time,
		dict has_seen
	):
	# start with a signal check
	sig_check()
	# and a time check
	cdef long cur_time = get_time()
	if cur_time - start_time > time_limit:
		raise TimeoutError
	
	# now the rest
	cdef char [::1] new_node_mask = np_mask_copy(node_mask)
	#cdef int [::1] new_edge_mask = np_mask_copy(edge_mask)
	cdef int num_edges_nomask = np_sum(edge_mask)
	cdef int num_ccs = 0
	cdef list ccs = []
	cdef list ccs_parents = []
	cdef list ccs_depths = []
	cdef long parent_cc_int = -1
	if parents:
		parent_cc_int = node_mask_to_int(num_nodes,node_mask)
	cdef int new_num_ccs
	cdef char [:,::1] new_cc_nodes
	cdef char [:,::1] new_cc_edges

	# lets check we have see this before
	# why not using c_str ? because I don't want to free it 
	node_full_mask_str = mask_to_str(num_nodes,node_mask) + '|' +  mask_to_str(num_edges,edge_mask) 
	#node_full_mask_str =  get_node_full_mask_str(num_nodes,node_mask, num_edges,edge_mask)

	need_skip = False
	if node_full_mask_str not in has_seen:
		has_seen[node_full_mask_str] = depth
	elif has_seen[node_full_mask_str] > depth:
		has_seen[node_full_mask_str] = depth
	else:
		need_skip = True
		
	if depth == max_depth or need_skip:
		pass
	elif num_edges_nomask == 0:
		# add leaf
		ccs = [new_node_mask]
		if parents:
			ccs_parents.append(parent_cc_int)
		if depths:
			ccs_depths.append(depth)
	else:
		for i in range(num_edges):
			# check if edge is already masked
			# sorry Mr Bond, you already dead
			if edge_mask[i] == 0:
				continue
			# mask edge, kill bond
			edge_mask[i] = 0
			# compute ccs
			new_cc_nodes, new_cc_edges, new_num_ccs = connected_components(
				num_nodes,
				num_edges,
				node_mask,
				edges,
				edge_mask,
				node_to_edge_idx
			)

			for c in range(new_num_ccs):
				ccs.append(new_cc_nodes[c,:])
				if parents:
					ccs_parents.append(parent_cc_int)
				if depths:
					ccs_depths.append(depth)
			num_ccs += new_num_ccs
			# recursive call
			for c in range(new_num_ccs):
				rec_ccs, rec_ccs_parents, rec_ccs_depths = compute_ccs_recursive(
					num_nodes,
					num_edges,
					new_cc_nodes[c,:],
					edges,
					new_cc_edges[c,:],
					node_to_edge_idx,
					depth+1,
					max_depth,
					parents,
					depths,
					time_limit,
					start_time,
					has_seen,
				)

				ccs.extend(rec_ccs)
				if parents:
					ccs_parents.extend(rec_ccs_parents)
				if depths:
					ccs_depths.extend(rec_ccs_depths)
				num_ccs += len(rec_ccs)
			# restore edge
			edge_mask[i] = 1

	return ccs, ccs_parents, ccs_depths

def py_compute_ccs_recursive(
		int num_nodes,
		int num_edges,
		char [::1] node_mask,
		int [:,::1] edges,
		char [::1] edge_mask,
		int [:,::1] node_to_edge_idx,
		int depth,
		int max_depth,
		bint convert_to_ints,
		bint parents,
		bint depths,
		long time_limit
	):

	assert num_nodes <= MAX_NUM_NODES, num_nodes
	assert num_edges <= MAX_NUM_EDGES, num_edges
	# at this point, there are duplicates
	# the latter two lists may be empty
	cdef long start_time = get_time()
	has_seen = {}
	ccs_list, ccs_parents_list, ccs_depth_list = compute_ccs_recursive(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		node_to_edge_idx,
		depth,
		max_depth,
		parents,
		depths,
		time_limit,
		start_time,
		has_seen,
	)
	# add the precursor
	ccs_list.append(compute_full_cc(num_nodes))
	if parents:
		ccs_parents_list.append(-1)
	if depths:
		ccs_depth_list.append(-1)
	# convert masks to ints
	if convert_to_ints:
		ccs_list = [node_mask_to_int(num_nodes,cc) for cc in ccs_list]

	return ccs_list, ccs_parents_list, ccs_depth_list
