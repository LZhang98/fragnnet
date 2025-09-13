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
	# frag feats configs
	parser.add_argument("--max_depth",type=int,default=3)
	parser.add_argument("--max_time",type=int,default=150)
	parser.add_argument("--project_dp",type=str,default=os.getcwd())
	parser.add_argument("--frag_dp",type=str,default="data/ms2c/casmi2016_neg/frag")
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--allowed_elements",type=str,nargs="+",default=frag_utils.ELEMENTS)
	parser.add_argument("--nb_isomorphic",type=booltype,default=False)
	parser.add_argument("--max_iterations",type=int,default=-1)
	parser.add_argument("--isotopes",type=booltype,default=False)
	parser.add_argument("--proc_dp", type=str,default="data/ms2c/casmi2016_neg/proc", help="")
	parser.add_argument("--allow_cached", type=booltype,default=False, help="")
	parser.add_argument("--compressed", type=booltype,default=True, help="")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
	dag_feat_inputs_l = []
	logger.info("Preparing DAG feature input list...")
	for _, row in mol_df.iterrows():
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
	for res in tqdm(frag_results, total=len(dag_feat_inputs_l), desc="Compute Frags"):
		n_done += 1
		if n_done % 100 == 0:
			logger.info(f"Processed {n_done} molecules...")
	logger.info(f"Fragment generation complete. Processed {n_done} molecules.")