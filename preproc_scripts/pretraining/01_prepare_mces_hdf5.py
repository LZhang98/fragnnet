import numpy as np
import argparse
from tqdm import tqdm
import h5py

def npz_to_hdf5(npz_fp, hdf5_fp, batch_size):
	"""
	Convert a .npz file to a .hdf5 file in batches.
	"""
	with np.load(npz_fp, allow_pickle=True) as data:
		with h5py.File(hdf5_fp, 'w') as hdf5_file:
			for key in data:
				arr = data[key]
				# Convert unicode arrays to bytes for HDF5 compatibility
				if arr.dtype.kind == 'U':
					arr = arr.astype('S')  # Convert to bytes
				dset = hdf5_file.create_dataset(
					key, shape=arr.shape, dtype=arr.dtype, compression='gzip', chunks=True
				)
				for i in tqdm(range(0, arr.shape[0], batch_size), desc=f"Writing {key}"):
					dset[i:i+batch_size] = arr[i:i+batch_size]
			print(f"Converted {npz_fp} to {hdf5_fp} in batches of {batch_size}")
   
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Filter MCES data and remove all -1 pairs.")
	parser.add_argument(
		"--mces_npz_fp",
		type=str,
		default="./data/mces-data/mces_distances.npz",
		help="Path to the MCES .npz file"
	)
	parser.add_argument(
		"--mces_hdf5_fp",
		type=str,
		default="./data/mces-data/mces_distances.hdf5",
		help="Path to the MCES .hdf5 file"
	)
	parser.add_argument(
		"--hdf5_batch_size",
		type=int,
		default=10000,
		help="Batch size for processing valid pairs"
	)
	args = parser.parse_args()
	npz_to_hdf5(args.mces_npz_fp, args.mces_hdf5_fp, args.hdf5_batch_size)