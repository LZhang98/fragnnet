import numpy as np
import argparse

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.misc_utils import timeit, booltype, get_dgl_memory_usage

import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, no_update
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64
from rdkit.Chem.Draw import rdMolDraw2D

def get_base64_image(mol, highlight = []):

	drawer = rdMolDraw2D.MolDraw2DCairo(200,200)
	drawer.DrawMolecule(mol,highlightAtoms = highlight)
	drawer.FinishDrawing()
	img_str = base64.b64encode(drawer.GetDrawingText()).decode('utf8')
	return img_str


def get_dag_fig(dag_d):

	smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
	mol = Chem.MolFromSmiles(smiles)
	idx_to_formula = dag_d["idx_to_formula"]
	G = dag_d["nx_dag"]
	pos_dict = {}
	node_text = []
	customdata = []
	for node in G.nodes(data = True):
		print(node)
		node_depth_mask = node[1]['depth']

		depth = 0
		for depth_value in node_depth_mask:
			if depth_value == 1:
				break
			depth += 1
		#print(depth)
		if depth not in pos_dict:
			pos_dict[depth] = 0
		else:
			pos_dict[depth] += 1
		node[1]['pos'] = (depth,pos_dict[depth])

		#node_text_str = ''
		#for idx in node[1]['h_formulae_idx']:
		#	node_text_str += idx_to_formula[idx] + ' '
		#node_text.append(node_text_str[:-1]) 
		#print(get_base64_image(mol)

		node_atom_mask = node[1]['mask']
		highlight_atom = []
		atom_idx = 0
		for value in node_atom_mask:
			if value == 1:
				highlight_atom.append(atom_idx)
			atom_idx += 1

		customdata.append(get_base64_image(mol, highlight_atom))

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
		mode='lines')

	node_x = []
	node_y = []
	for node in G.nodes():
		x, y = G.nodes[node]['pos']
		node_x.append(x)
		node_y.append(y)

	node_trace = go.Scatter(
		x=node_x, y=node_y,
		mode='markers',
		hoverinfo='text',
		marker=dict(
			showscale=True,
			# colorscale options
			#'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
			#'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
			#'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
			colorscale='YlGnBu',
			reversescale=True,
			color=[],
			size=10,
			colorbar=dict(
				thickness=15,
				title='Node Connections',
				xanchor='left',
				titleside='right'
			),
			line_width=2),
		customdata = customdata	
		)
	
	#node_adjacencies = []
	#node_text = []
	#for node, adjacencies in enumerate(G.adjacency()):
	#	node_adjacencies.append(len(adjacencies[1]))
	#	node_text.append('# of connections: '+str(len(adjacencies[1])))

	#node_trace.marker.color = node_adjacencies
	#node_trace.text = node_text


	fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Frag Graph',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    #return fig
	#fig.show()
	return fig

def run_frag(max_depth, h_prior, max_h_transfer, frag_max_time, smiles):
	mol_d = frag_utils.extract_mol_info(smiles)
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	print(num_nodes)

	include_nx_dag = True
	dag_d = timeit(frag_utils.compute_dags)(mol_d,max_depth,h_prior,max_h_transfer,frag_max_time, False, False, include_nx_dag)
	#print(dag_d)
	dag = dag_d["dag"]
	print("> dag")
	dag_nn = dag.number_of_nodes()
	dag_ne = dag.number_of_edges()
	dag_indeg = dag.in_degrees().numpy()
	print(f"num_nodes = {dag_nn}, num_edges = {dag_ne}, edge_frac = {dag_ne/(dag_nn*(dag_nn-1))}")
	print(f"avg_indeg = {np.mean(dag_indeg)}, std_indeg = {np.std(dag_indeg)}")
	dag_memory = get_dgl_memory_usage(dag)
	print(f"memory = {dag_memory/1e6} MB")

	return dag_d
	#print(dag_d['formula_peak_mzs'])
	#print(dag_d["nx_dag"])
	#plot_dag(dag_d)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--compound_idx",type=int,default=2)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	args = parser.parse_args()

	np.random.seed(420)

	COMPOUND_TO_SMILES = {
		0: "CCO", # ethanol
		1: "c1ccccc1", # benzene
		2: "CC(=O)OC1=CC=CC=C1C(=O)O", # aspirin
		3: "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", # testosterone
		4: "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O", # geddic acid
		5: "CSCCC(C(=O)O)N", # methionine, S=2
		6: "C1=CC(=CC=C1N)S(=O)(=O)N", # sulfanilamide, S=6
		7: "COP(C1=CC=CC=C1)C2=CC=CC=C2", # methoxydiphenylphosphine, P=3
		8: "CC1C(O1)P(=O)(O)O", # phosphomycin, P=5
		9: 'CC(=O)NC1C(OC2CCC3(C)C(CCC4(C)C3CC=C3C5CC(C)(C)CCC5(C(=O)O)CCC34C)C2(C)C)OC(CO)C(O)C1O', # big one (47 atoms)
		10: 'CC(=O)NC1C(OC2OC(CC(O)C3OC(n4ccc(=O)[nH]c4=O)C(O)C3O)C(O)C(O)C2NC(=O)C=CCCCCCCCCCCCC(C)C)OC(CO)C(O)C1O' # big one (60 atoms)
	}

	max_depth = args.max_depth
	h_prior = args.h_prior
	max_h_transfer = args.max_h_transfer
	frag_max_time = None
	smiles = COMPOUND_TO_SMILES[args.compound_idx]
	
	dag_d = run_frag(max_depth, h_prior, max_h_transfer, frag_max_time, smiles)
	fig = get_dag_fig(dag_d)
	
	app = Dash(__name__)

	app.layout = html.Div([
		dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
		dcc.Tooltip(id="graph-tooltip"),
	])


	@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
	)
	def display_hover(hoverData):
		if hoverData is None:
			return False, no_update, no_update
		
		#print(hoverData)
		# demo only shows the first point, but other points may also be available
		pt = hoverData["points"][0]
		bbox = pt["bbox"]
		num = pt["pointNumber"]

		#df_row = df.iloc[num]
		encoded_image = pt["customdata"]
		img_src = "data:image/jpeg;base64, " + encoded_image
		#img_src = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
		#name = "name"
		#desc = "des" #df_row['DESC'] 
	
		children = [
			html.Div(children=[
				html.Img(src=img_src, style={"width": "100%"}),
				#html.H2(f"{name}", style={"color": "darkblue"}),
				#html.P(f"{desc}"),
			],
			style={'width': '200px', 'white-space': 'normal'})
		    ]

		return True, bbox, children

	app.run_server(debug=True)