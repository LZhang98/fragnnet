import argparse
import pickle
import torch as th
from tqdm import tqdm

from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL
from frag_gnn.iceberg.pl_model import IcebergGenPL, IcebergIntenPL
from frag_gnn.utils.plot_utils import plot_spectra_sparse
from frag_gnn.utils.misc_utils import to_device
from frag_gnn.dataset import SpecMolFragDataset, init_dataloader, SpecMolDataset


import os
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_spectra(predicted_fp, save_dp):

	print(f"> load data from  {predicted_fp}")
	all_pred_data = []
	with (open(predicted_fp, "rb")) as file_in:
		all_pred_data = pickle.load(file_in)

	for pred_data in tqdm(all_pred_data):
		
		true_spec_mzs = pred_data['spec_mzs'].numpy()
		true_spec_ints = pred_data['spec_ints'].numpy()

		pred_spec_mzs = pred_data['pred_mzs'].numpy()
		pre_spec_ints = pred_data['pred_ints'].numpy()
		
		cos_sim_bin = pred_data['cos_sim_0.01'].cpu()
		cos_sim_hun = pred_data['cos_hun'].cpu()

		try:
			smiles = pred_data['smiles'][0]
			spec_unique_id = pred_data['spec_unique_id'].numpy()[0]
			fp = os.path.join(save_dp, f"{spec_unique_id}.png")

			# noramilze int to max
			true_spec_ints = true_spec_ints / true_spec_ints.max()
			pre_spec_ints = pre_spec_ints / pre_spec_ints.max()

			plot_spectra_sparse(true_spec_mzs,true_spec_ints,
						pred_spec_mzs,pre_spec_ints,smiles, fp)
		except:
			print("error",pred_data['spec_unique_id'].numpy()[0])
		
def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--predicted_fp",type=str,default="data/predicted/nist20_d4_wr00_inchikey/kme8fry7_fraggnn_d4_nodemlp_mnce_nocp.pkl.gz")
	parser.add_argument("--save_dp", type=str,default="figs/nist20_d4_wr00_inchikey/kme8fry7_fraggnn_d4_nodemlp_mnce_nocp")

	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	os.makedirs(args.save_dp, exist_ok=True)
	plot_spectra(predicted_fp = args.predicted_fp,
			  	save_dp = args.save_dp)
