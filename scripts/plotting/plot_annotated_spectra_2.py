import argparse
import pandas as pd
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mc

import os
 
 
def plot_formula_axes(ax,
					  formula:str,
					  peak_id:str|int,
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

	ax.text(10,text_loc, r"$f_{"+str(peak_id)+"}$", size=font_size, horizontalalignment='left', verticalalignment = 'center',
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
	formula_prob_display_str =  "$P(f_{"+ str(peak_id) +"})=" + "{:.3f}".format(formula_prob) + "$"
	ax.text(1.25 * font_size_pixel,text_loc ,formula_prob_display_str, size=font_size, horizontalalignment='left', verticalalignment = 'center', fontfamily = font)
	
	text_loc = 0.85 * ax_height
	formula_node_display_str =  "$P(n|f_{"+ str(peak_id) +"})=$"
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
	parser.add_argument("--predicted_fp", type=str, required=False, help="Predicted data pickle file path", default='output/inference_eval/32n55f2m/predictions_test.pkl')
	parser.add_argument("--proc_dp", type=str, required=False, help="Processed data directory",  default="data/proc/nist20")
	parser.add_argument("--save_dp", type=str, required=False, help="Directory to save plots", default="output/spectra_plots_test")
	parser.add_argument("--has_truth", type=str, required=False, help="True data pickle file path")
	parser.add_argument("--num_plots", type=int, required=False, help="Number of plots to generate", default=400)
	
	args = parser.parse_args()
  	# Load reference data once - try .pkl.gz first, then .pkl, then error
	proc_dp = args.proc_dp
	mol_df_path_gz = os.path.join(proc_dp, "mol_df.pkl.gz")
	mol_df_path_pkl = os.path.join(proc_dp, "mol_df.pkl")
	
	if os.path.exists(mol_df_path_gz):
		mol_df = pd.read_pickle(mol_df_path_gz)
		print(f"Loaded mol_df with {len(mol_df)} molecules from {mol_df_path_gz}")
	elif os.path.exists(mol_df_path_pkl):
		mol_df = pd.read_pickle(mol_df_path_pkl)
		print(f"Loaded mol_df with {len(mol_df)} molecules from {mol_df_path_pkl}")
	else:
		raise FileNotFoundError(f"Could not find mol_df file in {proc_dp}. Looked for both mol_df.pkl.gz and mol_df.pkl")
	
	spec_df_path_gz = os.path.join(proc_dp, "spec_df.pkl.gz")
	spec_df_path_pkl = os.path.join(proc_dp, "spec_df.pkl")
	
	if os.path.exists(spec_df_path_gz):
		spec_df = pd.read_pickle(spec_df_path_gz)
		print(f"Loaded spec_df with {len(spec_df)} spectra from {spec_df_path_gz}")
	elif os.path.exists(spec_df_path_pkl):
		spec_df = pd.read_pickle(spec_df_path_pkl)
		print(f"Loaded spec_df with {len(spec_df)} spectra from {spec_df_path_pkl}")
	else:
		raise FileNotFoundError(f"Could not find spec_df file in {proc_dp}. Looked for both spec_df.pkl.gz and spec_df.pkl")

	output_dir = "output/spectra_plots"
	os.makedirs(output_dir, exist_ok=True)
 
	has_input_mol_id = 'input_mol_id' in mol_df.columns
	if has_input_mol_id:
		print("mol_df contains 'input_mol_id' column")
		spec_df = spec_df.merge(mol_df[['mol_id', 'smiles', 'inchikey_s', 'input_mol_id']], on='mol_id', how='left')
	else:
		spec_df = spec_df.merge(mol_df[['mol_id', 'smiles', 'inchikey_s']], on='mol_id', how='left')
	
	pred_df = pd.read_pickle(args.predicted_fp)
	print(f"Loaded pred_df with {len(pred_df)} predictions")
	if 'mol_id' not in pred_df.columns:
		pred_df = pred_df.merge(spec_df, on='spec_id', how='left')
	else:
		pred_df = pred_df.merge(spec_df, on=['spec_id','mol_id'], how='left')
	print(f"Merged pred_df now has {len(pred_df)} rows")
	pred_df.loc[:, "pred_peaks"] = pred_df.apply(lambda x: np.stack([np.array(x['pred_mzs']), np.array(x['pred_ints'])], axis=1), axis=1)
   
	# Limit to first few examples for testing
	num_plots = min(args.num_plots, len(pred_df))
	
	for idx, pred_data in tqdm(pred_df.head(num_plots).iterrows(), total=num_plots):
		# print(pred_data)
		spec_id = pred_data['spec_id']
		mol_id = pred_data['mol_id']
		
		# Extract predicted spectrum data
		pred_spec_mzs = np.array(pred_data['pred_mzs'])
		pred_spec_ints = np.array(pred_data['pred_ints'])
		pred_spec_mzs = pred_spec_mzs # Adjust for plotting
  
		# Parse true peaks from string format
		if 'peaks' not in pred_data or pred_data['peaks'] is None:
			peaks_data = None
		else:
			peaks_data = pred_data['peaks']
		
		# Handle different data types for peaks
		try:
			# Check if data is NaN/None (handle both scalar and array cases)
			is_true_peak_na = peaks_data is None or (isinstance(peaks_data, (float, np.floating)) and np.isnan(peaks_data))
		except:
			is_true_peak_na = False
		
		if is_true_peak_na:
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
