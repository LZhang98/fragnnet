import os
import sys
import subprocess
import time
import cProfile

# Profiling harness for src/fragnnet/frag/compute_frags.pyx
# Writes prof.out in workspace root and generates cython annotation HTML.

def build_extension():
    print('Attempting to build Cython extension (cythonize -a -i)...')
    cmd = [sys.executable, '-m', 'cythonize', '-a', '-i', 'src/fragnnet/frag/compute_frags.pyx']
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print('Build failed. Ensure Cython and a C compiler are installed.')
        raise


def import_module():
    try:
        from fragnnet.frag import compute_frags as cf
        return cf
    except Exception as e:
        print('Import failed, will try to build then import:', e)
        build_extension()
        from fragnnet.frag import compute_frags as cf
        return cf


def make_chain(n):
    import numpy as np
    num_nodes = n
    edges = [(i, i+1) for i in range(n-1)]
    edges = np.asarray(edges, dtype=np.intc)
    node_mask = np.ones(num_nodes, dtype=np.uint8)
    edge_mask = np.ones(edges.shape[0], dtype=np.uint8)
    # compute node_to_edge_idx via helper
    node_to_edge_idx = None
    try:
        from fragnnet.frag import compute_frags as cf
        node_to_edge_idx = cf.py_compute_node_to_edge_idx(num_nodes, edges.shape[0], edges)
    except Exception:
        # fallback simple construction
        node_to_edge_idx = np.full((num_nodes, 6), -1, dtype=np.intc)
        for j in range(edges.shape[0]):
            s, d = edges[j,0], edges[j,1]
            for k in range(6):
                if node_to_edge_idx[s,k] == -1:
                    node_to_edge_idx[s,k] = j
                    break
            for k in range(6):
                if node_to_edge_idx[d,k] == -1:
                    node_to_edge_idx[d,k] = j
                    break
    return num_nodes, edges, node_mask, edge_mask, node_to_edge_idx


def run_profile(iterations=200, nodes=50, max_depth=4):
    cf = import_module()
    num_nodes, edges, node_mask, edge_mask, node_to_edge_idx = make_chain(nodes)

    def workload():
        for _ in range(iterations):
            cf.compute_ccs(
                int(num_nodes),
                int(edges.shape[0]),
                node_mask,
                edges,
                edge_mask,
                node_to_edge_idx,
                int(max_depth),
                60,
            )

    print('Starting cProfile run...')
    cProfile.runctx('workload()', globals(), locals(), filename='prof.out')
    print('Wrote prof.out')

    # generate cython annotation if cython exists
    try:
        subprocess.check_call([sys.executable, '-m', 'cython', '-a', 'src/fragnnet/frag/compute_frags.pyx'])
        print('Generated compute_frags.html')
    except Exception as e:
        print('Could not run cython -a to generate HTML annotation:', e)


if __name__ == '__main__':
    # tune iteration count if you expect a long build
    run_profile(iterations=200, nodes=50, max_depth=4)
