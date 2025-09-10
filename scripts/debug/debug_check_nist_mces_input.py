import argparse
import os
import pandas as pd
import h5py
from tqdm import tqdm

if __name__ == "__main__":
	h5_fp = "./data/mces-data/nist20_inputs/mces_input_1.h5"
	with h5py.File(h5_fp, "r") as f:
		print("Groups/Datasets in file:")
		def print_h5(name, obj):
			if isinstance(obj, h5py.Dataset):
				print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
			elif isinstance(obj, h5py.Group):
				print(f"Group: {name}")
		f.visititems(print_h5)

		# Example: Show first 5 items of each dataset
		for name, ds in f.items():
			if isinstance(ds, h5py.Dataset):
				print(f"\nFirst 5 entries in {name}:")
				print(ds[:5])