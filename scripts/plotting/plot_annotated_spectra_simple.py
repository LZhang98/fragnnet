import argparse
import pickle
import pandas as pd
import torch as th
from tqdm import tqdm

from fragnnet.utils.frag_utils import load_frag_d, th_long_to_mask
from fragnnet.utils.plot_utils import plot_spectra_sparse, get_mol_im, get_highlighted_bonds
from fragnnet.utils.spec_utils import *

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem, Draw

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.collections as mc
from matplotlib.text import Text

def plot_spectra_axes(ax, 
					  true_mzs,
					  true_ints,
					  pred_mzs, 
					  pred_ints, 
					  bar_width=3, 
					  font_size = 24,  
					  true_ints_thresh = 0,
					  pred_ints_thresh = 0,
					  colors = ['green', 'red'], 
					  circle_pad = 0.1,
					  peak_labels = None):


	mz_max = 1500
	bar_width = 1
	#true_ints_thresh = 0
	#pred_ints_thresh = 0
	
	true_mask = true_ints > true_ints_thresh
	true_mzs = true_mzs[true_mask]
	true_ints = true_ints[true_mask]
	pred_mask = pred_ints > pred_ints_thresh
	pred_mzs = pred_mzs[pred_mask]
	pred_ints = pred_ints[pred_mask]
	true_mz_max = max(np.max(true_mzs),np.max(pred_mzs))

	mz_max = int(np.floor(1.05*min(mz_max,true_mz_max)))
	true_mask = true_mzs < mz_max
	pred_mask = pred_mzs < mz_max
	true_mzs = true_mzs[true_mask]
	true_ints = true_ints[true_mask]
	pred_mzs = pred_mzs[pred_mask]
	pred_ints = pred_ints[pred_mask]

	x_max = mz_max
	ints_max = np.max(true_ints)
	y_max = 1.05*ints_max


	#
	ax.set_ylim(-y_max,y_max)
	ax.set_xlim(0,x_max)
	
	# get peaks
	lines_top, styles_top = [], []

	mz_mask = np.ones_like(true_mzs,dtype=bool)
	for i in range(len(true_mzs)):
		mz = true_mzs[i]
		ints = true_ints[i]
		line_start = (mz,0)
		line_end = (mz,ints)
		lines_top.append((line_start,line_end))
		if mz_mask[i]:
			styles_top.append("solid")
		else:
			styles_top.append((0,(1,1)))
			
	# print(lines_top[:10])
	lc_top = mc.LineCollection(
		lines_top, 
		colors=colors[0], 
		linewidths=bar_width, 
		linestyles=styles_top
	)
	ax.add_collection(lc_top)

	# add botton peaks and peak ids
	lines_bottom = []
	bottom_annotations = []
	bottom_annotations_x = []
	bottom_annotations_y = []
	
	if peak_labels is None:
		peak_labels = []
	
	# we maye need change this to bars
	for i in range(len(pred_mzs)):
		mz = pred_mzs[i]
		ints = pred_ints[i]
		line_start = (mz,0)
		line_end = (mz,-ints)
		lines_bottom.append((line_start,line_end))
		
		if i in peak_labels:
			bottom_annotations.append(ax.text(mz, -ints - ints_max * 0.02, 
											  "$f_{" + str(peak_labels[i]) + "}$", 
											  size=font_size,
											  ha='left', va='top',
											  bbox=dict(boxstyle="circle",
												ec=(1., 0.5, 0.5),
												fc=(1., 0.8, 0.8),
												pad = circle_pad
												)
									  ))
			bottom_annotations_x.append(mz)
			bottom_annotations_y.append(-ints)

	lc_bottom = mc.LineCollection(lines_bottom, colors=colors[1], linewidths=bar_width)
	lc_bottom = ax.add_collection(lc_bottom)
	
	#bars = ax.bar(pred_mzs, -pred_ints, width=bar_width/2, color=colors[1])
			
	ax.set_axisbelow(True)
	ax.grid(color="black", linewidth=0.1)
	ax.tick_params(axis='x', labelsize=font_size)
	ax.tick_params(axis='y', labelsize=font_size)
	#ax.tick_params(axis='both', which='both', labelsize='small')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_position('zero')
	ax.spines['top'].set_color('none')
	
	# Note: Text overlaps may occur without adjust_text
		
	leg_font_size = 0.9*font_size
	leg_kws = {'ncol': 1, 'fontsize': leg_font_size, "loc": "upper left"}
	handles = [
		patches.Patch(facecolor=colors[0],label="Ground Truth",alpha=1.0),
		patches.Patch(facecolor=colors[1],label="Predicted",alpha=1.0),
	]
	ax.legend(handles=handles,**leg_kws)

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--predicted_fp", type=str, required=False, help="Predicted data pickle file path", default='output/inference_eval/at7r9y7u/predictions_test.pkl')
	parser.add_argument("--true_fp", type=str, required=False, help="True data pickle file path", default='data/proc/nist23_ei/spec_df.pkl')
	parser.add_argument("--mol_df_fp", type=str, required=False, help="Molecule dataframe pickle file path", default='data/proc/nist23_ei/mol_df.pkl')

	args = parser.parse_args()
	pred_df = pd.read_pickle(args.predicted_fp)
	true_df = pd.read_pickle(args.true_fp)
	print(true_df.columns)
	mol_df = pd.read_pickle(args.mol_df_fp)
	
	merge_df = pred_df.merge(true_df[['spec_id','peaks']], on='spec_id', how='left')
	merge_df = merge_df.merge(mol_df, on='mol_id', how='left')
 
	print(merge_df.head())
	print(merge_df.columns)
	# Create output directory for plots
	import os
	output_dir = "output/spectra_plots"
	os.makedirs(output_dir, exist_ok=True)
	merge_df = merge_df[merge_df['cos_sim_1.0'] >= 0.8]
	# Limit to first few examples for testing
	num_plots = min(10, len(merge_df))
	
	for idx, pred_data in tqdm(merge_df.head(num_plots).iterrows(), total=num_plots):
		spec_id = pred_data['spec_id']
		mol_id = pred_data['mol_id']
		
		# Extract predicted spectrum data
		pred_spec_mzs = np.array(pred_data['pred_mzs'])
		pred_spec_ints = np.array(pred_data['pred_ints'])
		pred_spec_mzs = pred_spec_mzs - 0.5  # Adjust for plotting
		# Parse true peaks from string format
		peaks_data = pred_data['peaks']
		
		# Handle different data types for peaks
		try:
			# Check if data is NaN/None (handle both scalar and array cases)
			is_na = peaks_data is None or (isinstance(peaks_data, (float, np.floating)) and np.isnan(peaks_data))
		except:
			is_na = False
		
		if is_na:
			# No true peaks available
			true_spec_mzs = np.array([])
			true_spec_ints = np.array([])
		elif isinstance(peaks_data, str):
			# String format: "mz1 int1\nmz2 int2\n..."
			if peaks_data.strip():
				peak_lines = peaks_data.strip().split('\n')
				true_mzs, true_ints = [], []
				for line in peak_lines:
					if line.strip():
						parts = line.strip().split()
						if len(parts) >= 2:
							try:
								mz = float(parts[0])
								intensity = float(parts[1])
								true_mzs.append(mz)
								true_ints.append(intensity)
							except ValueError:
								continue
				true_spec_mzs = np.array(true_mzs)
				true_spec_ints = np.array(true_ints)
			else:
				true_spec_mzs = np.array([])
				true_spec_ints = np.array([])
		elif isinstance(peaks_data, (list, np.ndarray)):
			# Array format: assume it's already parsed peaks data
			try:
				if len(peaks_data) > 0 and isinstance(peaks_data[0], (list, tuple)):
					# Format: [(mz1, int1), (mz2, int2), ...]
					true_mzs = [peak[0] for peak in peaks_data]
					true_ints = [peak[1] for peak in peaks_data]
					true_spec_mzs = np.array(true_mzs)
					true_spec_ints = np.array(true_ints)
				else:
					# Unknown format
					true_spec_mzs = np.array([])
					true_spec_ints = np.array([])
			except:
				true_spec_mzs = np.array([])
				true_spec_ints = np.array([])
		else:
			# Unknown data type
			true_spec_mzs = np.array([])
			true_spec_ints = np.array([])
		
		# Skip if no data available
		if len(pred_spec_mzs) == 0 and len(true_spec_mzs) == 0:
			continue
		
		# Create mirror plot
		fig, ax = plt.subplots(1, 1, figsize=(12, 8))
		
		# Plot spectra with mirror layout
		plot_spectra_axes(
			ax=ax,
			true_mzs=true_spec_mzs,
			true_ints=true_spec_ints,
			pred_mzs=pred_spec_mzs,
			pred_ints=pred_spec_ints,
			bar_width=2,
			font_size=12,
			true_ints_thresh=0,  # Filter low intensity peaks
			pred_ints_thresh=0,
			colors=['green', 'red'],
			circle_pad=0.1
		)
		
		# Add title with metadata
		cos_sim = pred_data.get('cos_sim_1.0', 'N/A')
		title = f"Spec ID: {spec_id}, Mol ID: {mol_id}\nCosine Similarity: {cos_sim:.3f}" if cos_sim != 'N/A' else f"Spec ID: {spec_id}, Mol ID: {mol_id}"
		ax.set_title(title, fontsize=14, pad=20)
		
		# Set axis labels
		ax.set_xlabel("m/z", fontsize=12)
		ax.set_ylabel("Relative Intensity", fontsize=12)
		
		# Save plot
		plot_filename = f"mirror_plot_spec_{spec_id}_mol_{mol_id}.png"
		plot_path = os.path.join(output_dir, plot_filename)
		plt.tight_layout()
		plt.savefig(plot_path, dpi=150, bbox_inches='tight')
		plt.close()
		
		print(f"Saved mirror plot: {plot_path}")
		print(f"  True peaks: {len(true_spec_mzs)}, Predicted peaks: {len(pred_spec_mzs)}")
		if cos_sim != 'N/A':
			print(f"  Cosine similarity: {cos_sim:.3f}")
	
	print(f"\nCompleted generating {num_plots} mirror plots in {output_dir}")
