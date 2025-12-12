import argparse
from math import ceil
import re
import os
from io import BytesIO


import pandas as pd
from tqdm import tqdm
import numpy as np
import torch as th
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.collections as mc
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from PIL import Image

from rdkit.Chem import Draw

from fragnnet.utils.frag_utils import th_long_to_mask
from fragnnet.utils.plot_utils import get_highlighted_bonds, get_mol_im

def plot_formula_axes(ax,
					  formula:str,
					  formula_label: str,
					  formula_prob = None,
					  font_size = None,
					  font ='DejaVu Sans Mono',
					  circle_pad = 0.1):
	
	fig = plt.gcf()

	bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
	ax_width, ax_height = bbox.width, bbox.height
	#print(ax_width, ax_height)
	ax_width *= fig.dpi
	ax_height *= fig.dpi

	ax.set_xlim(0, ax_width)
	ax.set_ylim(ax_height, 0)
		
	ax.axis('off')
	if font_size == None:
		font_size = int(bbox.height * 72 / 4)
	font_size_pixel = int(font_size * fig.dpi / 72)
	text_loc = font_size_pixel/3 * 2
	#ax.text(0, text_loc, f"Peak {peak_id}", size=font_size, horizontalalignment='left', verticalalignment = 'center')
	
	#circle = matplotlib.patches.Circle((0, 0), radius=1, color='rea')
	#ax.add_patch(circle)

	ax.text(10,text_loc, "$f_{"+ formula_label +"}$", size=font_size, horizontalalignment='left', verticalalignment = 'center',
			bbox=dict(boxstyle="circle",
				   ec=(1., 0.5, 0.5),
				   fc=(1., 0.8, 0.8),
				   pad = circle_pad,
				   ))
	
	#text_loc += font_size * 4
	#font_size = int((bbox.height / 4 + circle_pad * 2 ) * 72)
	
	formula_display_str = "$" + formula + "$"
	numbers = re.compile(r'(\d+)')
	formula_display_str = numbers.sub(r'_{\1}',formula_display_str)
	ax.text(1.25 * font_size_pixel,text_loc ,formula_display_str, size=font_size, horizontalalignment='left', verticalalignment = 'center', fontfamily = font)
	
	#if formula_prob is not None:
	text_loc += 1.25 * font_size_pixel
	formula_prob_display_str =  "$P(f_{"+ formula_label +"})=" + "{:.3f}".format(formula_prob) + "$"
	ax.text(1.25 * font_size_pixel,text_loc ,formula_prob_display_str, size=font_size, horizontalalignment='left', verticalalignment = 'center', fontfamily = font)
	
	text_loc = 0.85 * ax_height
	formula_node_display_str =  "$P(n|f_{"+ formula_label +"})=$"
	ax.text(1.25 * font_size_pixel,text_loc,formula_node_display_str, size=font_size, horizontalalignment='left', verticalalignment = 'center', fontfamily = font)
		
def plot_mol_frag_axes(ax, 
						mol: str, 
						atom_highlight_idx:list,
						bond_highlight_idx:list,
						frag_formula_prob:float,
						font_size = None,
						font = 'DejaVu Sans Mono'):


	fig = plt.gcf()

	bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
	ax_width, ax_height = bbox.width, bbox.height
	#print(ax_width, ax_height)
	ax_width *= fig.dpi
	ax_height *= fig.dpi

	ax.set_xlim(0, ax_width)
	ax.set_ylim(ax_height, 0)

	image_width = int(ax_width * 0.9)
	image_height = int(ax_height * 0.65)
	image_x_offset = int(0.5*(ax_width - image_width))
	image_y_offset = 0.0
	#print(image_width, image_height)
	d2d = Draw.MolDraw2DCairo(image_width, image_height)
	options = d2d.drawOptions()
	rgba_color = (0.529, 0.808, 0.98)# transparent blue
	options.fillHighlights=True
	options.setHighlightColour((rgba_color))
	options.highlightBondWidthMultiplier=20
	options.clearBackground = False
	d2d.DrawMolecule(mol,highlightAtoms=atom_highlight_idx, highlightBonds =  bond_highlight_idx)
	d2d.FinishDrawing()
	bio = BytesIO(d2d.GetDrawingText())
	mol_image = Image.open(bio)

	# (left, right, bottom, top)
	extent = (0 + image_x_offset, image_width + image_x_offset, image_height + image_y_offset, image_y_offset)
	ax.imshow(mol_image, extent = extent, interpolation = 'lanczos')
	
	if font_size == None:
		font_size = int(bbox.height * 72 / 4)
		
	text_loc = 0.85 * ax_height
	#formula_node_display_str =  "$P(n|f_{"+ str(peak_id) +"})=$"
	ax.text(ax_width/2,text_loc, "{:.3f}".format(frag_formula_prob), size=font_size, horizontalalignment='center', verticalalignment = 'center', fontfamily = font)
	ax.axis('off')

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
					  peak_label_dict = None):


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
	
	# peak_label_dict can be a dict {pred_peak_index: label_str}
	if peak_label_dict is None:
		peak_label_dict = {}
	
	# we maye need change this to bars
	for i in range(len(pred_mzs)):
		mz = pred_mzs[i]
		ints = pred_ints[i]
		line_start = (mz,0)
		line_end = (mz,-ints)
		lines_bottom.append((line_start,line_end))

		# annotate with provided formula label if available
		if i in peak_label_dict:
			label = str(peak_label_dict[i])
			# print(label)
			bottom_annotations.append(
				ax.text(
					mz,
					-ints - ints_max * 0.02,
					label,
					size=font_size,
					ha='left', va='top',
					bbox=dict(
						boxstyle="circle",
						ec=(1., 0.5, 0.5),
						fc=(1., 0.8, 0.8),
						pad=circle_pad,
					)
				)
			)
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

	# Remove 0 from x-axis ticks
	xticks = ax.get_xticks()
	ax.set_xticks([t for t in xticks if t != 0])

	# Show absolute values on y-axis (mirror plot)
	ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{abs(y):.0f}"))

def load_dataframe(file_path_gz, file_path_pkl, description):
	"""Load dataframe from .pkl.gz or .pkl file"""
	if os.path.exists(file_path_gz):
		df = pd.read_pickle(file_path_gz)
		print(f"Loaded {description} with {len(df)} entries from {file_path_gz}")
	elif os.path.exists(file_path_pkl):
		df = pd.read_pickle(file_path_pkl)
		print(f"Loaded {description} with {len(df)} entries from {file_path_pkl}")
	else:
		raise FileNotFoundError(f"Could not find {description} file. Looked for both {file_path_gz} and {file_path_pkl}")
	return df

def load_and_prepare_data(args):
	"""Load and merge all reference data"""
	proc_dp = args.proc_dp
	
	# Load mol_df
	mol_df_path_gz = os.path.join(proc_dp, "mol_df.pkl.gz")
	mol_df_path_pkl = os.path.join(proc_dp, "mol_df.pkl")
	mol_df = load_dataframe(mol_df_path_gz, mol_df_path_pkl, "mol_df")
	
	# Load spec_df
	spec_df_path_gz = os.path.join(proc_dp, "spec_df.pkl.gz")
	spec_df_path_pkl = os.path.join(proc_dp, "spec_df.pkl")
	spec_df = load_dataframe(spec_df_path_gz, spec_df_path_pkl, "spec_df")
	
	# Merge with mol_df
	has_input_mol_id = 'input_mol_id' in mol_df.columns
	if has_input_mol_id:
		print("mol_df contains 'input_mol_id' column")
		spec_df = spec_df.merge(mol_df[['mol_id', 'smiles', 'inchikey_s', 'input_mol_id', 'mol']], on='mol_id', how='left')
	else:
		spec_df = spec_df.merge(mol_df[['mol_id', 'smiles', 'inchikey_s', 'mol']], on='mol_id', how='left')
	
	# Load pred_df
	pred_df = pd.read_pickle(args.predicted_fp)
	print(f"Loaded pred_df with {len(pred_df)} predictions")
	if 'mol_id' not in pred_df.columns:
		pred_df = pred_df.merge(spec_df, on='spec_id', how='left')
	else:
		pred_df = pred_df.merge(spec_df, on=['spec_id','mol_id'], how='left')
	print(f"Merged pred_df now has {len(pred_df)} rows")
	
	# Add pred_peaks column
	pred_df.loc[:, "pred_peaks"] = pred_df.apply(lambda x: np.stack([np.array(x['pred_mzs']), np.array(x['pred_ints'])], axis=1), axis=1)
	
	return pred_df

def parse_peaks_from_string(peaks_data):
	"""Parse true peaks from various data formats"""
	# Check if data is NaN/None (handle both scalar and array cases)
	try:
		is_true_peak_na = peaks_data is None or (isinstance(peaks_data, (float, np.floating)) and np.isnan(peaks_data))
	except:
		is_true_peak_na = False
	
	if is_true_peak_na:
		# No true peaks available
		return np.array([]), np.array([])
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
			return np.array(true_mzs), np.array(true_ints)
		else:
			return np.array([]), np.array([])
	elif isinstance(peaks_data, (list, np.ndarray)):
		# Array format: assume it's already parsed peaks data
		try:
			if len(peaks_data) > 0 and isinstance(peaks_data[0], (list, tuple)):
				# Format: [(mz1, int1), (mz2, int2), ...]
				true_mzs = [peak[0] for peak in peaks_data]
				true_ints = [peak[1] for peak in peaks_data]
				return np.array(true_mzs), np.array(true_ints)
			else:
				# Unknown format
				return np.array([]), np.array([])
		except:
			return np.array([]), np.array([])
	else:
		# Unknown data type
		return np.array([]), np.array([])

def process_prediction_data(pred_data):

	"""Process prediction data to extract formulas, nodes, and probabilities"""
	# Get formula dict
	# TODO double check isotope handling
	formula_dict = {i:{"formula":s,"mz":m, "peak_idx":pi,"prob":p} for i,s,m,pi,p in zip(pred_data['pred_formula_formula_idxs'],
		pred_data['pred_formula_strs'],
		pred_data['frag_formula_peak_mzs'], # not sure what will heappen if isotope is enabled
		pred_data['frag_formula_peak_idxs'], # not sure what will heappen if isotope is enabled
		pred_data['pred_formula_probs'])}
	
	# Sort by prob
	formula_dict = dict(sorted(formula_dict.items(), key=lambda item: item[1]["prob"], reverse=True))
	
	# Map formula to nodes
	formula_node_map = {i:[] for i in set(pred_data['pred_joint_formula_idxs'])}
	#for f_id, n_id in zip(pred_data['pred_joint_formula_idxs'], pred_data['pred_joint_node_idxs']):
	#	formula_node_map[f_id].append(n_id)
	
	pred_formula_node_probs = pred_data['pred_formula_node_probs']
	pred_node_formula_probs = pred_data['pred_node_formula_probs']
	pred_joint_formula_idxs = pred_data['pred_joint_formula_idxs']
	pred_joint_node_idxs = pred_data['pred_joint_node_idxs']
 
	node_formula_prob_dict = {}
	formula_node_prob_dict = {}
	for f_id, n_id, f_n_prob, n_f_prob in zip(pred_joint_formula_idxs,
											pred_joint_node_idxs,
											pred_formula_node_probs,
											pred_node_formula_probs):
		node_formula_prob_dict[(f_id, n_id)] = n_f_prob
		formula_node_prob_dict[(f_id, n_id)] = f_n_prob
		formula_node_map[f_id].append(n_id)
	
	# Get node masks
	frag_node_masks = th_long_to_mask(pred_data['pred_node_ccs'])
	all_atom_ids = th.arange(0, frag_node_masks.shape[1]).to(th.long)
	node_dict = {i:{"pred_node_probs":p,"pred_node_ccs": all_atom_ids[c].tolist()} for i,p,c in zip(pred_data['pred_node_node_idxs'],
		pred_data['pred_node_probs'],
		frag_node_masks)}
	
	return formula_dict, formula_node_map, node_dict, formula_node_prob_dict

def build_display_data(formula_dict, formula_node_map, node_dict, formula_node_prob_dict, mol):
	"""Build display data structure for plotting"""
	display_data = []
	
	for _, f_id in enumerate(formula_dict.keys()):
		
		formula_data = {
			"formula_idx": f_id,
			"formula":  formula_dict[f_id]["formula"],
			"formula_prob": formula_dict[f_id]["prob"],
			"formula_mass": formula_dict[f_id]["mz"],
			"peak_idx": formula_dict[f_id]["peak_idx"],
			"frag": []
		}
		
		for n_id in formula_node_map[f_id]:
			if n_id not in node_dict:
				continue
			pred_node_prob = node_dict[n_id]["pred_node_probs"]
			frag_atom_ids = node_dict[n_id]["pred_node_ccs"]
			formula_data["frag"].append(
				{
					"node_atom_indices": frag_atom_ids,
					"node_bond_indices": get_highlighted_bonds(mol, frag_atom_ids),
					"node_prob": pred_node_prob,
					"formula_node_prob": formula_node_prob_dict.get((f_id, n_id), 0.0)
				}
			)
		# sort fragments by formula_node_prob
		formula_data["frag"] = sorted(formula_data["frag"], key=lambda x: x['formula_node_prob'], reverse=True)
		display_data.append(formula_data)

	return display_data

def plot_mirror_spectra(pred_data, display_data, output_dir):
	"""Generate and save mirror plot for a single spectrum"""
	smiles = pred_data['smiles']
	mol = pred_data['mol']
	spec_id = pred_data['spec_id']
	mol_id = pred_data['mol_id']
	
	# Extract predicted spectrum data
	pred_spec_mzs = np.array(pred_data['pred_mzs'])
	pred_spec_ints = np.array(pred_data['pred_ints'])
	
	# Parse true peaks
	peaks_data = pred_data.get('peaks', None)
	true_spec_mzs, true_spec_ints = parse_peaks_from_string(peaks_data)
	
	# Skip if no data available
	if len(pred_spec_mzs) == 0 and len(true_spec_mzs) == 0:
		return False
	
	# Get molecule image
	img = get_mol_im(smiles)
	w, h = img.size
	ratio = w / h
	
	# Setup plot layout
	right_ncols = 3 + 1
	right_rows = 8
	figsize = (24, 16)
	frag_col_w = figsize[1] / right_rows * 0.5 * ratio * right_ncols
	frag_col_w = int(ceil(frag_col_w)) + 2
	
	# Build title with additional metadata
	title_parts = [f'Spec ID: {spec_id}', f'Mol ID: {mol_id}']
	
	# Add collision energy if available
	col_energy = pred_data.get('col_energy', None)
	if col_energy is not None and pd.notna(col_energy) and str(col_energy).strip():
		title_parts.append(f'CE: {col_energy}')

	# Prefer normalized CE (NCE) or absolute CE (ACE) if present
	nce = pred_data.get('nce', None)
	ace = pred_data.get('ace', None)
	if nce is not None and pd.notna(nce) and str(nce).strip():
		title_parts.append(f'NCE: {nce}')
	elif ace is not None and pd.notna(ace) and str(ace).strip():
		title_parts.append(f'ACE: {ace}')
	
	# Add adduct/precursor type if available
	prec_type = pred_data.get('prec_type', None)
	if prec_type is not None and pd.notna(prec_type) and str(prec_type).strip():
		title_parts.append(f'Adduct: {prec_type}')
	
	# Add instrument type if available
	inst_type = pred_data.get('inst_type', None)
	if inst_type is not None and pd.notna(inst_type) and str(inst_type).strip():
		title_parts.append(f'Inst: {inst_type}')
	
	title = ' | '.join(title_parts)
	
	fig = plt.figure(figsize=figsize, dpi=300)
	fig.suptitle(title, fontsize=28, fontweight='bold')
	grand_gs = GridSpec(ncols=figsize[0], nrows=1, figure=fig)
	left_gs = grand_gs[0:figsize[0]-frag_col_w].subgridspec(ncols=1, nrows=1)
	right_gs = grand_gs[figsize[0]-frag_col_w:].subgridspec(ncols=right_ncols, nrows=right_rows)
	
	# Build peak label map from display_data: {peak_idx: formula}
	peak_label_dict = {}
	for fd in display_data:		
		p_idx = fd.get('peak_idx', None)
		f_idx = fd.get('formula_idx', None) 
		if p_idx is not None and f_idx is not None and str(f_idx).strip():
			peak_label_dict[int(p_idx)] =  r"$f_{"+str(f_idx)+r"}$"

	# print(peak_label_dict)
	# Plot spectrum
	ax = fig.add_subplot(left_gs[0], facecolor="white")
	plot_spectra_axes(ax,
					true_spec_mzs,
					true_spec_ints,
					pred_spec_mzs,
					pred_spec_ints,
					bar_width=1,
					font_size=24,
					true_ints_thresh=0,
					pred_ints_thresh=0,
					colors=['green', 'red'],
					peak_label_dict=peak_label_dict)

	# Overlay NCE/ACE on the spectrum axes if available
	ann_text = None
	if nce is not None and pd.notna(nce) and str(nce).strip():
		ann_text = f'NCE: {nce}'
	elif ace is not None and pd.notna(ace) and str(ace).strip():
		ann_text = f'ACE: {ace}'
	if ann_text is not None:
		ax.text(0.99, 0.02, ann_text, transform=ax.transAxes,
				ha='right', va='bottom', fontsize=16,
				bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
	
	# Plot formulas and fragments
	for f_idx, formula_data in enumerate(display_data):
		if f_idx >= right_rows:
			break
		formula = formula_data['formula']
		# null formula check (this should not happen)
		if formula is None or formula == "":
			continue

		ax = fig.add_subplot(right_gs[f_idx, 0])
		#print(f"Plotting formula {f_idx}: {formula} for peak {peak_id}")
		plot_formula_axes(ax, formula, str(formula_data['formula_idx']), formula_data['formula_prob'])
		
		for frag_idx, frag in enumerate(formula_data['frag']):
			if frag_idx + 1 >= right_ncols:
				break
			ax = fig.add_subplot(right_gs[f_idx, frag_idx + 1])
			h_atom_list = frag['node_atom_indices']
			h_bond_list = frag['node_bond_indices']
			plot_mol_frag_axes(ax, mol, h_atom_list, h_bond_list, frag['formula_node_prob'])
	
	# Save plot
	plot_filename = f"mirror_plot_spec_{spec_id}_mol_{mol_id}.png"
	plot_path = os.path.join(output_dir, plot_filename)
	#try:
	#	plt.tight_layout()
	#except ValueError as e:
		# tight_layout may fail if there are math parsing errors in text
	#	print(f"Warning: tight_layout failed with error: {e}")
	plt.savefig(plot_path, dpi=150, bbox_inches='tight')
	plt.close()
	
	print(f"Saved mirror plot: {plot_path}")
	#print(f"  True peaks: {len(true_spec_mzs)}, Predicted peaks: {len(pred_spec_mzs)}")
	
	return True

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--predicted_fp", type=str, required=False, help="Predicted data pickle file path", default='output/inference_eval/32n55f2m/predictions_test.pkl')
	parser.add_argument("--proc_dp", type=str, required=False, help="Processed data directory",  default="data/proc/nist20")
	parser.add_argument("--save_dp", type=str, required=False, help="Directory to save plots", default="output/spectra_plots_test")
	parser.add_argument("--has_truth", type=str, required=False, help="True data pickle file path")
	parser.add_argument("--num_plots", type=int, required=False, help="Number of plots to generate", default=20)
	
	args = parser.parse_args()
	
	# Load and prepare data
	pred_df = load_and_prepare_data(args)
	
	output_dir = args.save_dp
	os.makedirs(output_dir, exist_ok=True)
   
	# Limit to first few examples for testing
	num_plots = min(args.num_plots, len(pred_df))
	
	for idx, pred_data in tqdm(pred_df.head(num_plots).iterrows(), total=num_plots):
     
		print(pred_data)
		#exit()
		# Process prediction data
		formula_dict, formula_node_map, node_dict, formula_node_prob_dict = process_prediction_data(pred_data)
		
		# Build display data
		display_data = build_display_data(formula_dict, formula_node_map, node_dict, formula_node_prob_dict, pred_data['mol'])
		
		# Generate and save plot
		plot_mirror_spectra(pred_data, display_data, output_dir)
	
	print(f"\nCompleted generating {num_plots} mirror plots in {output_dir}")
