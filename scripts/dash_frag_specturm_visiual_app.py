import os
import pandas as pd

from frag_gnn.formula_utils import formula_to_peak_mzs
from dash import Dash, Patch, dcc, html, Input, Output, no_update
from rdkit.Chem.Draw import rdMolDraw2D
from dgl.data.utils import load_graphs

import dgl
import torch as th
import base64
import plotly.graph_objects as go
import numpy as np

# global dict for easy access
# This is not safe for more than one process
# it will do for now
dag_d = {}

def get_base64_image(mol, highlight = []):
	drawer = rdMolDraw2D.MolDraw2DCairo(200,200)
	drawer.DrawMolecule(mol, highlightAtoms = highlight)
	drawer.FinishDrawing()
	img_str = base64.b64encode(drawer.GetDrawingText()).decode('utf8')
	return img_str

def get_dag_fig(dag_d, mol):

	idx_to_formula = dag_d["idx_to_formula"]
	G = dag_d["nx_dag"]
	pos_dict = {}
	customdata = []
	for node in G.nodes(data = True):
		#print(node)
		node_depth_mask = node[1]['depth']

		# compute depth and add pos
		depth = 0
		for depth_value in node_depth_mask:
			if depth_value == 1:
				break
			depth += 1
		if depth not in pos_dict:
			pos_dict[depth] = 0
		else:
			pos_dict[depth] += 1
		node[1]['pos'] = (depth,pos_dict[depth])
		
		# add image
		node_atom_mask = node[1]['mask']
		highlight_atom = []
		atom_idx = 0
		for value in node_atom_mask.tolist():
			if value == 1:
				highlight_atom.append(atom_idx)
			atom_idx += 1

		# add MWs
		formulas = []
		for id in node[1]['h_formulae_idx'].tolist():
			if id != 0:
				mass = formula_to_peak_mzs(idx_to_formula[id], "[M+H]+", isotopes = False)
				#print(idx_to_formula[id], mass[0])
				formulas.append(idx_to_formula[id] + ':{0:.4f}Da'.format(mass[0]))
		#print(formulas)
		customdata.append([get_base64_image(mol, highlight_atom),formulas])

	# add lines
	# note line is not clickable or hoverable
	edge_x = []
	edge_y = []
	for edge in G.edges():
		x0, y0 = G.nodes[edge[0]]['pos']
		x1, y1 = G.nodes[edge[1]]['pos']
		edge_x.append(x0)
		edge_x.append(x1)
		edge_x.append(None)
		edge_y.append(y0)
		edge_y.append(y1)
		edge_y.append(None)

	edge_trace = go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=0.5, color='#888'),
		hoverinfo='none',
		marker= dict(size=10,symbol= "arrow-bar-up", angleref="previous"),
		mode='lines+markers')

	node_x = []
	node_y = []
	for node in G.nodes():
		x, y = G.nodes[node]['pos']
		node_x.append(x)
		node_y.append(y)

	node_trace = go.Scatter(
		x=node_x, y=node_y,
		mode='markers',
		hoverinfo='none',
		marker=dict(
			reversescale=True,
			color=[],
			size=10,
			line_width=2),
		customdata = customdata	
		)
	

	fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Frag Graph',
                titlefont_size=32,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
  
	return fig

def get_specturm_fig(peaks):

	x_values = []
	y_values = []
	print(peaks)
	for peak in peaks:
		x_values.append(peak[0])
		y_values.append(peak[1])

	print(x_values)
	print(y_values)
	fig = go.Figure([go.Bar(x=x_values, y=y_values, width = 1.0)],
		              layout=go.Layout(
                title='<br>Spectrum',
                titlefont_size=32,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

	#fig.show()
	return fig

def load_dgl_graph(dgl_saved_graph):
	glist, label_dict = load_graphs(dgl_saved_graph) # glist will be [g1, g2]
	return glist

def find_nearest(np_array, value):
    #array = np.asarray(array)
    idx = (np.abs(np_array - value)).argmin()
    return np_array[idx]

def get_updated_color(dag_d, peak_mws):

	idx_to_formula = dag_d["idx_to_formula"]
	G = dag_d["nx_dag"]
	colors = []

	peak_mws = np.asarray(peak_mws)
	colour_set = ["rgb(31, 119, 180)", "rgb(255, 127, 14)", "rgb(44, 160, 44)"]

	node_color = colour_set[1]
	for node in G.nodes(data = True):
		# add MWs
		#formulas = []
		for id in node[1]['h_formulae_idx'].tolist():
			if id != 0:
				mass = formula_to_peak_mzs(idx_to_formula[id], "[M+H]+", isotopes = False)
				closet = find_nearest(peak_mws, mass)
				if abs(mass - closet) < 0.01:
					node_color = colour_set[1]
					break
				#print(idx_to_formula[id], mass[0])
				#formulas.append(idx_to_formula[id] + ':{0:.4f}Da'.format(mass[0]))
		colors.append(node_color)
	return colors


app = Dash(__name__)

@app.callback(
	Output("dag-graph-tooltip", "show"),
	Output("dag-graph-tooltip", "bbox"),
	Output("dag-graph-tooltip", "children"),
	Input("dag-graph", "hoverData"),
	)
def display_hover(hoverData):
	if hoverData is None:
		return False, no_update, no_update
	
	pt = hoverData["points"][0]
	bbox = pt["bbox"]

	if "customdata" not in pt:
		return False, no_update, no_update

	encoded_image = pt["customdata"][0]
	img_src = "data:image/jpeg;base64, " + encoded_image
	formula_str_list = pt["customdata"][1]
	#desc = "des" #df_row['DESC'] 

	children_elemenets = [html.Img(src=img_src, style={"width": "100%"})]
	#children_elemenets.append(html.Img(src=img_src, style={"width": "100%"}))
	for formula_str in formula_str_list:
		children_elemenets.append(html.P(f"{formula_str}"))
		#children_elemenets.append(html.Br)

	children = [
		html.Div(children=children_elemenets,
			
		style={'width': '200px', 'white-space': 'normal'})
		]

	return True, bbox, children


@app.callback(
Output('spectrum_fig', 'figure'),
#Output('dag-graph', 'figure'),
Input('spectrum_dropdown', 'value')
)
def update_spectrum_fig(value):
	if value is None:
		return no_update
	spec_id = value
	#print(spec_df[spec_df['spec_id'] == spec_id])
	peaks = spec_df[spec_df['spec_id'] == spec_id]['peaks'].values[0]
	specturm_fig = get_specturm_fig(peaks)

	#colors = get_updated_color(dag_d, [p[0] for p in peaks])
	# Creating a Patch object
	#patched_dag_fig = Patch()
	#patched_figure["layout"]["title"]["font"]["color"] = new_color
	return specturm_fig #, patched_dag_fig

@app.callback(
	Output('dag-graph', 'figure'),
	Output('spectrum_dropdown','options'),
	Input('mol_dropdown', 'value')
)
def update_dag_graph_and_specturm_list(value):
	print('update_dag_graph_and_specturm_list',value)
	if value is None:
		return no_update, no_update
	mol_id = value

	dgl_file = os.path.join(project_dp,frag_stats_dp,'dags',f'{mol_id}_dag.bin')
	th_file = os.path.join(project_dp,frag_stats_dp,'dags',f'{mol_id}_dag_meta.pt')
	dag_fig = None
	if os.path.exists(dgl_file) and os.path.exists(th_file):
		print("file exists", dgl_file, th_file)
		print(">load dgl", dgl_file)
		graph = load_dgl_graph(dgl_file)
		#print(graph)
		print(">load meta", th_file)
		dag_d = th.load(th_file)
		#print(graph)
		dag_d["nx_dag"] = dgl.to_networkx(graph[0],
						node_attrs=['depth','h_formulae_idx','base_formula','mask'], 
						#edge_attrs=['base_formula','mask']
						)
		dag_fig = get_dag_fig(dag_d, mol_df[mol_df['mol_id'] == mol_id]['mol'].values[0])
		print(dag_fig)
		#print(type(dag_fig))
	else:
		print("file not exists", dgl_file, th_file)
	#print(dag_fig)
	specturm_ids = spec_df[spec_df['mol_id'] == mol_id]['spec_id'].to_list()
	#print(specturm_ids)
	return dag_fig, specturm_ids

# read in the molecule data
project_dp = os.getcwd()
frag_dp = "data/frag"
frag_stats_dp = "data/frag/debug"
proc_dp = "data/proc"

print(">read in mol df")
mol_df = pd.read_pickle(os.path.join(project_dp,proc_dp,"mol_df.pkl"))
print(">read in spec df")
spec_df = pd.read_pickle(os.path.join(project_dp,proc_dp,"spec_df.pkl"))
print(">finish loading dfs")

print(">listing saved files")
spectrum_dropdown_values = []
dag_fir_list = os.listdir(os.path.join(project_dp,frag_stats_dp,'dags'))

print(">populate mol drop down")
mol_dropdown_values = [] #spec_df['mol_id'].unique().tolist()

for mol_id in spec_df['mol_id'].unique().tolist():
	dgl_file = f'{mol_id}_dag.bin'
	th_file = f'{mol_id}_dag_meta.pt'
	if dgl_file in dag_fir_list and th_file in dag_fir_list:
		mol_dropdown_values.append(mol_id)
	
#for mol_id in spec_df['mol_id'].unique().values.tolist():
#	mol_id = spec_df[spec_df['spec_id'] == specturm_id]['mol_id'].values[0]
#	if '{mol_id}_dag.bin' in dag_fir_list and\
#		f'{mol_id}_dag_meta.pt' in dag_fir_list:
#		spectrum_dropdown_values.append(specturm_id)

print(">setup layout")
app.layout = html.Div([
	dcc.Dropdown(mol_dropdown_values, id='mol_dropdown'),
	dcc.Dropdown(id='spectrum_dropdown'),
	html.Div([
		#html.Button("Label M/Z on DAG", id="update-button"),
		dcc.Graph(id="spectrum_fig", 
			#figure=specturm_fig, 
			style={'height': '90vh'},
			clear_on_unhover=True),
],
style={'width': '49%', 'display': 'inline-block'}),
html.Div([
		dcc.Graph(id="dag-graph", 
			#figure=dag_fig, 
			style={'height': '90vh'},
			clear_on_unhover=True),
	dcc.Tooltip(id="dag-graph-tooltip"),
],
style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),
])

if __name__ == "__main__":
	app.run_server(debug=True)