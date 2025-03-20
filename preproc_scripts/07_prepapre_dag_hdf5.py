import bz2
import pickle
import tarfile
import h5py
import numpy as np
import frag_gnn.utils.frag_utils as frag_utils


def append_hdf5_rows(hdf5_fp,new_rows, num_features):
    
	with h5py.File(hdf5_fp, "a") as f:
		dataset = f["data"]
		current_size = dataset.shape[0]
		new_size = current_size + len(new_rows)  # Increase row count
		dataset.resize((new_size, num_features))  # Extend dataset
		dataset[current_size:new_size, :] = new_rows  # Write new batch

		# Flush changes to disk to ensure data is not lost
		f.flush()

def create_hdf5_file(hdf5_fp:str,num_features:int,batch_size:int):
    # Create an HDF5 file with an extendable dataset
    
	keys = ['max_depth', 'reached_depth', 'h_prior', 'max_h_transfer', 'formula_peak_mzs', 'formula_peak_probs', 'idx_to_formula', 'idx_by_h_delta', 'dag', 'edges_min_depth', 'nodes_min_depth', 'dag_num_edges', 'dag_num_nodes', 'dag_num_nodes_nb', 'dag_sparsity', 'formula_redundancy', 'node_feature_size', 'edge_feature_size', 'is_directed', 'dag_num_edges_by_depth', 'dag_num_nodes_by_depth']
    
	with h5py.File(hdf5_fp, "w") as f:
		dataset = f.create_dataset(
			"dags",
			shape=(0, num_features),  # Start with zero rows
			maxshape=(None, num_features),  # Allow unlimited rows
			dtype="f4",
			chunks=(batch_size, num_features),  # Write in small batches
			compression="gzip",  # Optional compression
			compression_opts=4
		)
if __name__ == "__main__":
	# Define dataset parameters
	num_features = 1  # Number of columns (features)
	batch_size = 1  # Number of rows to write in one batch

	#frag_d = pickle.loads(bz2.decompress(content))
	frag_l = []
	#frag_tar = 'data/frag/nist20mona23v3_d3_h4_isoTrue2.tar'
	fp = 'data/frag/nist20mona23v2_d3_h4_isoFalse_nb3/dags/1.pickle.bz2'
	with bz2.BZ2File(fp, 'rb') as filein: 
		frag_d = pickle.load(filein)

	print(frag_d)
	frag_l.append(frag_d)
	for k,v in frag_d.items():
		print(k,type(v))
	# Create an HDF5 file with an extendable dataset
	create_hdf5_file("incremental_data.h5", num_features, batch_size)

	# Simulate real-time data streaming (incremental writing)
	for i in range(5):  # Simulating 5 batches of incoming data
		new_batch = np.random.rand(batch_size, num_features)  # Generate a batch
		append_hdf5_rows(new_batch)  # Append to HDF5 file
		print(f"Batch {i+1} written to disk.")

	# Verify stored data
	#with h5py.File("incremental_data.h5", "r") as f:
	#    dataset = f["data"]
	#    print("Final dataset shape:", dataset.shape)  # Should be (50, 100)