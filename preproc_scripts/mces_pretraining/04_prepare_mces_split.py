import argparse
import os
import pandas as pd
import json

from fragnnet.utils import frag_utils

def make_splits(args):

	ms_mol_df = pd.read_pickle(os.path.join(args.ms_proc_dp,"mol_df.pkl"))
	ms_split_d = {}
	for split_type in ["train","val","test"]:
		split_fp = os.path.join(args.ms_split_dp,f"{split_type}_ids.csv")
		split_df = pd.read_csv(split_fp)
		ms_split_d[split_type] = ms_mol_df[ms_mol_df['mol_id'].isin(split_df['mol_id'])]['inchikey_s'].unique()

	mol_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl"))
	mces_data = pd.read_pickle(os.path.join(args.proc_dp,"mces_pairs.pkl"))

	# Create MCES splits based on MS splits
	# Ensure molecules in MS train split are also in MCES train split
	split_d = {}
	
	# Filter mol_df by MS split assignments based on inchikey_s
	for split_type in ["train", "val", "test"]:
		# Get mol_ids from mol_df that have inchikey_s in the corresponding MS split
		split_mol_ids = mol_df[mol_df['inchikey_s'].isin(ms_split_d[split_type])].index.tolist()
		split_d[split_type] = split_mol_ids
		print(f"{split_type}: {len(split_mol_ids)} molecules")
	
 
	print(mol_df.head())
	total_mols = mol_df['inchikey_s'].nunique()
	print(f"\nTotal unique molecules in mol_df: {total_mols}")
	expected_train_mol_count = total_mols - int(total_mols * (args.test_frac + args.val_frac))
	print(f"Expected train molecules (approx): {expected_train_mol_count}")
	expected_val_mol_count = int(total_mols * args.val_frac)
	print(f"Expected val molecules (approx): {expected_val_mol_count}")
	expected_test_mol_count = int(total_mols * args.test_frac)
	print(f"Expected test molecules (approx): {expected_test_mol_count}")
   
	# Verify no overlap between test and train/val
	train_set = set(split_d['train'])
	val_set = set(split_d['val'])
	test_set = set(split_d['test'])
	
	print(f"\nChecking molecule overlaps:")
	print(f"Train ∩ Test: {len(train_set & test_set)} molecules (should be 0)")
	print(f"Val ∩ Test: {len(val_set & test_set)} molecules (should be 0)")
	print(f"Train ∩ Val: {len(train_set & val_set)} molecules")
	exit()
	# Filter MCES pairs to ensure both molecules in a pair belong to the same split
	# Remove any pairs that cross splits
	mces_split_d = {}
	
	print(f"\nFiltering MCES pairs (removing cross-split pairs)...")
	print(f"Total MCES pairs before filtering: {len(mces_data)}")
	
	for split_type in ["train", "val", "test"]:
		split_set = set(split_d[split_type])
		
		# Keep only pairs where both mol_id_1 and mol_id_2 are in the same split
		split_pairs = mces_data[
			(mces_data['mol_id_1'].isin(split_set)) & 
			(mces_data['mol_id_2'].isin(split_set))
		].copy()
		
		mces_split_d[split_type] = split_pairs
		print(f"{split_type}: {len(split_pairs)} MCES pairs")
	
	# Verify no pair overlap between splits
	train_pair_indices = set(mces_split_d['train'].index)
	val_pair_indices = set(mces_split_d['val'].index)
	test_pair_indices = set(mces_split_d['test'].index)
	
	print(f"\nChecking MCES pair overlaps:")
	print(f"Train ∩ Test pairs: {len(train_pair_indices & test_pair_indices)} (should be 0)")
	print(f"Val ∩ Test pairs: {len(val_pair_indices & test_pair_indices)} (should be 0)")
	print(f"Train ∩ Val pairs: {len(train_pair_indices & val_pair_indices)}")
	
	total_pairs_after = sum(len(mces_split_d[s]) for s in ["train", "val", "test"])
	removed_pairs = len(mces_data) - total_pairs_after
	print(f"\nRemoved {removed_pairs} cross-split pairs ({removed_pairs/len(mces_data)*100:.2f}%)")
	
	# Save splits
	os.makedirs(args.split_dp, exist_ok=True)
	
	# Save molecule IDs
	for split_type in ["train", "val", "test"]:
		split_fp = os.path.join(args.split_dp, f"{split_type}_ids.csv")
		split_df = pd.DataFrame({'mol_id': split_d[split_type]})
		split_df.to_csv(split_fp, index=False)
		print(f"\nSaved {split_type} mol_ids to {split_fp}")
	
	# Save MCES pairs
	for split_type in ["train", "val", "test"]:
		mces_split_fp = os.path.join(args.split_dp, f"{split_type}_mces_pairs.pkl")
		mces_split_d[split_type].to_pickle(mces_split_fp)
		print(f"Saved {split_type} MCES pairs to {mces_split_fp}")
	
	# Save split metadata
	metadata = {
		'split_type': args.split_type,
		'split_key': args.split_key,
		'ms_split_dp': args.ms_split_dp,
		'total_molecules': len(mol_df),
		'train_mol_size': len(split_d['train']),
		'val_mol_size': len(split_d['val']),
		'test_mol_size': len(split_d['test']),
		'total_mces_pairs': len(mces_data),
		'train_pairs_size': len(mces_split_d['train']),
		'val_pairs_size': len(mces_split_d['val']),
		'test_pairs_size': len(mces_split_d['test']),
		'removed_cross_split_pairs': removed_pairs,
		'cross_split_removal_rate': f"{removed_pairs/len(mces_data)*100:.2f}%"
	}
	
	metadata_fp = os.path.join(args.split_dp, "split_metadata.json")
	with open(metadata_fp, 'w') as f:
		json.dump(metadata, f, indent=2)
	print(f"\nSaved metadata to {metadata_fp}")

	return

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--split_type",type=str,choices=["random"],default="random")
	parser.add_argument("--split_key",type=str,choices=["inchikey_s","scaffold"],default="inchikey_s")
	parser.add_argument("--elements",type=str,nargs="+",default=frag_utils.ELEMENTS)

	# non-filtering args
	parser.add_argument("--meta_rseed",type=int,default=420420)
	parser.add_argument("--total_frac",type=float,default=1.0)
	parser.add_argument("--test_frac",type=float,default=0.2)
	parser.add_argument("--val_frac",type=float,default=0.2)
	parser.add_argument("--proc_dp",type=str, default="data/proc/mces")
	parser.add_argument("--split_dp",type=str, default="data/split/mces_nist23_ma_mi_inchikey")
	parser.add_argument("--ms_proc_dp",type=str, default="data/proc/nist23")
	parser.add_argument("--ms_split_dp",type=str, default="data/split/nist23_ma_mi_inchikey")
	#parser.add_argument("--proc_dp",type=str,required=True)
	#parser.add_argument("--split_dp",type=str,required=True)
	#parser.add_argument("--ms_proc_dp",type=str,required=True)
	#parser.add_argument("--ms_split_dp",type=str,required=True)
	args = parser.parse_args()

	make_splits(args)

