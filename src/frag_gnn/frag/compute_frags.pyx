# cython: profile=False
# cython: linetrace=False
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False, 
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
from cysignals.signals cimport sig_check
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME


MAX_NUM_NODES = 60 # hard cap on 64!!
MAX_NUM_EDGES = 150 # ~4*MAX_NUM_NODES 
DTYPE = np.intc
MASK_DTYPE = np.byte
MAX_EDGE_PER_NODE = 6 # Sulfur

class TimeoutError(RuntimeError):
    pass

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

#cdef long node_mask_to_int(int num_nodes, char [::1] node_mask):

#    assert MAX_NUM_NODES < 64, MAX_NUM_NODES
#    cdef long node_mask_int = 0
#    cdef int i
#    for i in range(num_nodes):
#        assert node_mask[i] >= 0, node_mask[i]
#        node_mask_int += node_mask[i]*(2**i)
#    assert node_mask_int >= 0, node_mask_int
#    return node_mask_int

cdef mask_to_big_int(int num, char [::1] mask):
    #Not fastest but it will do for now
    mask_str = ''
    node_mask_int = 0
    cdef int i
    for i in range(num):
        node_mask_int += mask[i]*(2**i)
    #mask_str = str(node_mask_int)
    return node_mask_int

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

#cdef char [::1] compute_full_cc(int num_nodes):
#    cdef char [::1] cc_nodes = np.zeros(MAX_NUM_NODES,dtype=MASK_DTYPE)
#    for i in range(num_nodes):
#        cc_nodes[i] = 1
#    return cc_nodes

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

def compute_ccs(
        int num_nodes,
        int num_edges,
        char [::1] node_mask,
        int [:,::1] edges,
        char [::1] edge_mask,
        int [:,::1] node_to_edge_idx,
        int max_depth,
        bint return_parents,
        bint return_depths,
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
    node_mask_key = mask_to_big_int(num_nodes,node_mask)
    edge_mask_key = mask_to_big_int(num_edges,edge_mask)

    cdef list ccs = [node_mask_key]
    cdef list current_edge_masks_list = [edge_mask_key]
    cdef list ccs_parents = [-1]
    cdef list ccs_depths = [-1]
    cdef dict cc_node_mask_dict = {node_mask_key:node_mask}
    cdef dict cc_edge_mask_dict = {edge_mask_key:edge_mask}
    cdef dict has_seen = {}
    cdef int num_ccs = 0

    # for each depth
    for current_depth in range(max_depth):
        # keep all the ccs mask for next round
        #child_node_mask_list = []
        child_edge_mask_list = []
        
        #print(">ccs:", len(ccs))
        ccs_end = len(ccs)
        for node_idx in range(ccs_idx,ccs_end):

            # let us check time
            cur_time = get_time()
            sig_check()
            if cur_time - start_time > time_limit:
                raise TimeoutError
    
            current_node_mask = cc_node_mask_dict[ccs[node_idx]]
            current_edge_mask = cc_edge_mask_dict[current_edge_masks_list[node_idx - ccs_idx]]

            # let us check mask
            node_full_mask_str = str(ccs[node_idx]) + '|' +  str(current_edge_masks_list[node_idx - ccs_idx])

            # check if we have seen has configure before
            need_skip = False
            if node_full_mask_str not in has_seen:
                has_seen[node_full_mask_str] = current_depth
            elif has_seen[node_full_mask_str] > current_depth:
                has_seen[node_full_mask_str] = current_depth
            else:
                need_skip = True

            # check if there any bond left to kill
            num_edges_nomask = np_sum(current_edge_mask)
            if need_skip or num_edges_nomask == 0:
                continue

            # compute parent_cc_int for current_node, assign to child node
            if return_parents:
                parent_cc_int = mask_to_big_int(num_nodes,current_node_mask)

            # now we need to break more bond
            i = 0
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

                # restore edge
                current_edge_mask[i] = 1
                # save data and save edge and node configure
                c = 0
                for c in range(new_num_ccs):
             
                    if return_parents:
                        ccs_parents.append(parent_cc_int)
                    if return_depths:
                        ccs_depths.append(current_depth)

                    # save cc_nodes
                    # child_node_mask_list.append(np_mask_copy(new_cc_nodes[c,:]))
                    child_node_mask_key = mask_to_big_int(num_nodes,new_cc_nodes[c,:])
                    child_edge_mask_key = mask_to_big_int(num_edges,new_cc_edges[c,:])

                    if child_node_mask_key not in cc_node_mask_dict:
                        cc_node_mask_dict[child_node_mask_key] = new_cc_nodes[c,:]
                    
                    if child_edge_mask_key not in cc_edge_mask_dict:
                        cc_edge_mask_dict[child_edge_mask_key] = new_cc_edges[c,:]

                    ccs.append(child_node_mask_key)
                    child_edge_mask_list.append(child_edge_mask_key)
                
                num_ccs += new_num_ccs
        # finishing current depth, move to next
        ccs_idx = ccs_end
        current_edge_masks_list = child_edge_mask_list
        child_edge_mask_list = []
    
    return ccs, ccs_parents, ccs_depths