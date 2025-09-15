import pandas as pd
from fragnnet.utils.misc_utils import booltype
from tqdm import tqdm
import argparse
import os
import logging
import fragnnet.utils.data_utils as data_utils
import fragnnet.utils.frag_utils as frag_utils

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--start_row", type=int, default=0, help="Start row index (inclusive) for chunked processing")
	parser.add_argument("--end_row", type=int, default=None, help="End row index (exclusive) for chunked processing")
	# frag feats configs
	parser.add_argument("--max_depth",type=int,default=3)
	parser.add_argument("--max_time",type=int,default=150)
	parser.add_argument("--project_dp",type=str,default=os.getcwd())
	parser.add_argument("--frag_dp",type=str,required=True, help="Path to the fragmentation output directory")
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--allowed_elements",type=str,nargs="+",default=frag_utils.ELEMENTS)
	parser.add_argument("--nb_isomorphic",type=booltype,default=False)
	parser.add_argument("--max_iterations",type=int,default=-1)
	parser.add_argument("--isotopes",type=booltype,default=False)
	parser.add_argument("--proc_dp", type=str,required=True, help="Path to the preprocessed mol_df and spec_df directory")
	parser.add_argument("--allow_cached", type=booltype,default=True, help="")
	parser.add_argument("--compressed", type=booltype,default=True, help="")
	parser.add_argument("--show_tqdm", type=booltype, default=False, help="Disable tqdm progress bar")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
	logger = logging.getLogger("frag_gen")

	args = get_args()
	logger.info(f"Starting MS2C FragGen run with parameters:")
	for k, v in vars(args).items():
		logger.info(f"  {k}: {v}")

	mol_df_fp = os.path.join(args.proc_dp, "mol_df.pkl")
	logger.info(f"Looking for mol_df at {mol_df_fp}")
	if os.path.isfile(mol_df_fp):
		logger.info(f"Found mol_df: {mol_df_fp}")
	elif os.path.isfile(mol_df_fp + ".gz"):
		mol_df_fp = mol_df_fp + ".gz"
		logger.info(f"Found gzipped mol_df: {mol_df_fp}")
	else:
		logger.error(f"mol_df.pkl or mol_df.pkl.gz not found in {args.proc_dp}")
		raise ValueError(f"mol_df.pkl or mol_df.pkl.gz not found in {args.proc_dp}")

	mol_df = pd.read_pickle(mol_df_fp)
	logger.info(f"Loaded mol_df with {len(mol_df)} molecules")
	logger.info(f"Creating output directory {args.frag_dp}")
	os.makedirs(args.frag_dp, exist_ok=True)

	# Select chunk
	start_row = args.start_row
	end_row = args.end_row if args.end_row is not None else len(mol_df)
	if start_row < 0 or start_row >= len(mol_df):
		logger.error(f"start_row {start_row} is out of bounds for mol_df of length {len(mol_df)}")
		raise ValueError(f"start_row {start_row} is out of bounds for mol_df of length {len(mol_df)}")
	if end_row > len(mol_df):
		logger.warning(f"end_row {end_row} is greater than mol_df length {len(mol_df)}, setting to {len(mol_df)}")
		end_row = len(mol_df)
	if end_row <= start_row:
		logger.error(f"end_row {end_row} must be greater than start_row {start_row}")
		raise ValueError(f"end_row {end_row} must be greater than start_row {start_row}")
	mol_df_chunk = mol_df.iloc[start_row:end_row]
	logger.info(f"Processing rows {start_row} to {end_row} (total {len(mol_df_chunk)})")

	dag_feat_inputs_l = []
	logger.info("Preparing DAG feature input list...")
	for _, row in mol_df_chunk.iterrows():
		dag_feat_inputs_l.append([
			row['mol'],
			row['mol_id'],
			args.max_depth,
			True, # h_prior
			args.max_h_transfer,
			args.max_time,
			args.isotopes,
			args.nb_isomorphic,
			args.max_iterations,
			args.frag_dp,
			args.allow_cached,
			args.compressed, # compressed
		])

	logger.info(f"Starting fragment generation for {len(dag_feat_inputs_l)} molecules...")
	frag_results = data_utils.par_apply(iter(dag_feat_inputs_l), frag_utils.timed_get_dags, True, return_as_generator=True)
	n_done = 0
	if not args.show_tqdm:
		for res in frag_results:
			n_done += 1
			if n_done % 100 == 0:
				logger.info(f"Processed {n_done} molecules...")
	else:
		for res in tqdm(frag_results, total=len(dag_feat_inputs_l), desc="Compute Frags"):
			n_done += 1
			if n_done % 100 == 0:
				logger.info(f"Processed {n_done} molecules...")
	logger.info(f"Fragment generation complete. Processed {n_done} molecules.")