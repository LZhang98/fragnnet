import numpy as np
import torch as th
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import torch._dynamo as th_dynamo
import torch_geometric as pyg
import time
from pprint import pprint, pformat
import argparse
from copy import deepcopy

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.utils.misc_utils import timeit, booltype, th_temp_seed, get_pyg_memory_usage
from frag_gnn.utils.feat_utils import get_mol_graph, get_frag_graph, batch_mols_frags
from frag_gnn.nn.nn_utils import GNN, JointPredictor, scatter_normalize
from frag_gnn.nn.loss import get_sparse_cross_entropy_fn
from frag_gnn.utils.spec_utils import calculate_recalls, calculate_spectrum_entropy, calculate_topk_weighted_recall, \
	calculate_match_mzs, calculate_cosine_similarity, batch_func
from frag_gnn.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF, PERIODIC_TABLE, parse_formula
from frag_gnn.utils.data_utils import mol_from_smiles, mol_to_formula
from frag_gnn.utils.plot_utils import plot_dual_axis, plot_spectra_sparse

# print(f"> frag memory usage = {get_pyg_memory_usage(frag_pyg)/1e6:.3f} MB")

# print(f"> mol memory usage = {get_pyg_memory_usage(mol_pyg)/1e6:.3f} MB")

def torch_time(func):
	def _time_func(*args,**kwargs):
		if th.cuda.is_available():
			start = th.cuda.Event(enable_timing=True)
			end = th.cuda.Event(enable_timing=True)
			start.record()
			result = func(*args,**kwargs)
			end.record()
			th.cuda.synchronize()
			elapsed_time = start.elapsed_time(end)
		else:
			start = time.time_ns()
			result = func(*args,**kwargs)
			end = time.time_ns()
			elapsed_time = (end-start) / 1e6
		print(f"> func {func.__name__}: time elapsed = {elapsed_time:.2f} ms")
		# print(f"> func: time elapsed = {elapsed_time:.2f} ms")
		return result
	return _time_func

def calc_prec_mz(smiles,prec_type):

	mol = mol_from_smiles(smiles)
	formula = mol_to_formula(mol)
	element_counts = parse_formula(formula)
	mass_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
	peak_mz = 0.
	for k, v in element_counts.items():
		peak_mz += v*PERIODIC_TABLE.GetMostCommonIsotopeMass(k)
	peak_mz += mass_diff
	return peak_mz


if __name__ == "__main__":

	# to get compound 0 to work use the following configuration: 
	# max_h_transfer=4
	# num_isotopes=1
	# tolerance=1e-3
	# out_dist=gaussian
	# lr=1e-4
	# num_iters=1000 (should see convergence around 500)
	# batch_size=2 (just used something small to speed up training)
	# other parameters don't matter
	
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--compound_idx",
		type=int,
		default=0,
		help="Index of compound to run"
	)
	parser.add_argument(
		"--max_depth",
		type=int,
		default=4,
		help="Maximum depth of fragments to consider"
	)
	parser.add_argument(
		"--max_h_transfer",
		type=int,
		default=4,
		help="Maximum number of hydrogens to transfer"	
	)
	parser.add_argument(
		"--vectorized",
		type=booltype,
		default=False,
		help="Whether to use vectorized (vmap) implementation of cross entropy loss (this will be slower unless your batch is very big)"
	)
	parser.add_argument(
		"--device_id",
		type=int,
		choices=[-1,0],
		default=-1,
		help="Which device to use (-1 for CPU, 0 for GPU)"
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=2,
		help="Minibatch size for training and prediction (when batch_size>1, the input compound is duplicated)"
	)
	parser.add_argument(
		"--jit",
		type=booltype,
		default=True,
		help="Whether to use JIT compilation for the cross entropy loss"
	)
	parser.add_argument(
		"--num_iters",
		type=int,
		default=10,
		help="Number of training iterations"
	)
	parser.add_argument(
		"--tolerance",
		type=float,
		default=1e-3,
		help="Tolerance for matching peaks (see --relative), representing the standard deviation of the output gaussian"
	)
	parser.add_argument(
		"--relative",
		type=booltype,
		default=False,
		help="Whether to interpret tolerance as relative (percentage of peak mass) instead of absolute"
	)
	parser.add_argument(
		"--random_targ",
		type=booltype,
		default=False,
		help="Whether to use random spectrum targets"
	)
	parser.add_argument(
		"--time",
		type=booltype,
		default=True,
		help="Whether to time each iteration"
	)
	parser.add_argument(
		"--lr",
		type=float,
		default=1e-4, #1e-3,
		help="Learning rate"
	)
	parser.add_argument(
		"--out_dist",
		type=str,
		default="gaussian",
		choices=["gaussian","mol_uniform"],
		help="The form of the output distribution (gaussian or mollified uniform)"
	)
	parser.add_argument(
		"--num_isotopes",
		type=int,
		default=1, #frag_utils.MAX_NUM_MZS_PER_FORMULA,
		help="Maximum number of isotopic trace peaks to consider"
	)
	parser.add_argument(
		"--undirected_dag",
		type=booltype,
		default=False,
		help="Whether to use undirected edges in the fragment DAG"
	)
	args = parser.parse_args()

	assert not (args.jit and args.vectorized)

	np.random.seed(420)
	th.manual_seed(420)

	IDX_TO_SMILES = {
		# compounds with spectra
		0: "C1=CC=C(C=C1)C2=CSC(=N2)NC(=O)C3=CC=NC=C3",
		1: "CCCCCCCCC=CCCCCCCCC(=O)NCC",
		2: "CC1=C(N=C(N=C1OC(=O)N(C)C)NC)C",
	}

	# just hard-coding some examples
	# 0: mol_id=22041, smiles=O=C(Nc1nc(-c2ccccc2)cs1)c1ccncc1, wrecall=0.9814, depth=3
	# 1: mol_id=7602, smiles=CCCCCCCCC=CCCCCCCCC(=O)NCC, wrecall=0.9947, depth=3
	# 2: mol_id=11009, smiles=CNc1nc(C)c(C)c(OC(=O)N(C)C)n1, wrecall=0.9614, depth=3
	IDX_TO_SPEC = {
		0: [
			(50.0149, 41.36), (50.015, 11.19), (51.0227, 300.4), (51.0228, 9.19), (51.0229, 2.7), 
			(52.018, 24.37), (52.0295, 3.1), (52.0305, 270.12), (52.0306, 15.38), (52.0307, 5.09),
			(53.0384, 393.31000000000006), (53.0385, 5.8), (62.0149, 2.3), (63.0227, 33.769999999999996),
			(65.0361, 3.7), (65.0383, 408.99), (65.0384, 5.59), (67.0416, 2.2), (68.9792, 16.48), 
			(69.0057, 5.99), (69.0081, 781.3100000000001), (76.0305, 2.7), (77.0133, 3.7), (77.0383, 37.06), 
			(78.0335, 144.66000000000003), (78.0336, 361.25), (79.0393, 9.59), (79.0414, 4138.26), 
			(80.0476, 23.28), (80.0492, 6937.459999999999), (81.0525, 25.79), (85.9692, 66.72),
			(89.0383, 297.3), (90.046, 3.0), (90.0461, 59.44), (91.0514, 10.89), (91.0539, 2997.0), 
			(91.054, 581.11), (92.0572, 10.39), (92.0573, 6.79), (92.0574, 7.29), (92.9791, 3.0), 
			(93.0444, 2.2), (93.057, 37.37), (93.0571, 12.889999999999999), (95.0488, 21.08), (95.0489, 4.0), 
			(96.0441, 152.84), (96.0442, 1.6), (102.0461, 14.19), (102.0462, 26.37), (104.0492, 61.45), 
			(104.0493, 19.08), (105.0444, 3.7), (105.0445, 256.14000000000004), (106.0285, 334.97), 
			(107.0363, 38.77), (107.0364, 15.09), (108.0026, 15.28), (111.0312, 7.99), (111.0313, 91.41), 
			(116.0493, 70.52000000000001), (116.0494, 16.18), (117.0571, 5.1), (117.0572, 5.49), 
			(118.0523, 2.7), (118.0524, 2.4), (120.0554, 39.56), (120.0555, 108.28999999999999), 
			(121.0104, 69.63), (121.0105, 144.95), (124.0213, 1.8), (131.0601, 4.3), (131.0603, 6.59), 
			(132.0442, 19.68), (132.0443, 6.79), (133.0105,56.65), (133.0106, 28.28), (134.0182, 1.5), 
			(134.0183, 564.53), (134.0184, 94.3), (135.0261, 62.339999999999996), (135.0262, 820.88), 
			(136.0296, 2.7), (139.0542, 2.0), (148.0214, 124.47999999999999), (148.0215,75.73), 
			(148.0216, 7.69), (148.0503, 3.0), (148.0504, 35.46), (148.0505, 18.18), (159.0139, 3.1), 
			(160.0213, 2.0), (160.0214, 53.54), (160.0215, 17.09), (161.0167, 2.6), (161.0169, 4.9), 
			(175.0324, 11.889999999999999), (175.0325, 3.0), (176.0163, 34.97), (176.0164, 93.41), 
			(176.0166, 3.6), (176.0401, 25.18), (176.0402, 27.369999999999997), (176.0403, 16.78), 
			(178.0319, 4.2), (178.032, 5.0), (183.0916, 2.0), (187.0325, 1.6), (193.0763, 4.9), 
			(194.0839, 4.7), (194.084, 6.89), (195.0917, 7.29), (195.0918, 3.5), (200.0526, 2.7), 
			(200.0527, 2.9), (203.0272, 136.87), (203.0273, 1296.6000000000001), (203.0278, 4.4), 
			(204.0301, 2.4), (204.0305, 2.2), (204.0308, 3.5), (212.0527, 4.3), (212.0529, 1.4), 
			(212.053, 2.3), (220.0868, 2.2), (221.0943, 2.1), (221.0947, 5.19), (221.0948, 5.29), 
			(227.0634, 1.6), (227.0635, 10.59), (227.0637, 183.03), (227.064, 2.7), (239.0635, 31.07), 
			(239.0636, 94.31), (239.0637, 229.26999999999998), (249.0894, 4.6), (249.0896, 8.690000000000001), 
			(249.0899, 2.3), (254.0679, 2.6), (254.0745, 762.74), (254.0746, 210.69), (254.0747, 9.39), 
			(255.0778, 10.899999999999999), (264.0588, 52.040000000000006), (264.0589, 25.27), (282.0515, 2.2), 
			(282.0591, 2.0), (282.0693, 1969.53), (282.0694, 1383.42), (282.0702, 3.4), (283.0728, 36.36), (283.0729, 19.38)
		],
		1: [
			(53.0387, 89.22), (55.0179, 9.29), (55.018, 209.7), (55.0181, 242.86), (55.0544, 2353.15), 
			(55.0545, 999.0), (57.0686, 71.93), (57.0687, 17.88), (57.0698, 753.34), (57.07, 2065.34), 
			(67.0542, 361.94), (67.0543, 225.78), (69.0697, 2188.21), (69.0699, 199.7), (71.0854, 900.4), 
			(71.0855, 86.01), (72.0425, 25.87), (72.0426, 39.35), (72.0442, 5324.06), (72.0443, 286.51), 
			(79.0542, 126.07), (79.0543, 65.24), (81.0697, 82.62), (81.0698, 232.57000000000002), 
			(81.0699, 98.8), (83.049, 3.2), (83.0854, 794.51), (83.0855, 38.66), (83.0856, 122.18), 
			(85.1011, 450.33), (93.0698, 19.28), (93.0699, 131.07), (95.0855, 319.29), (95.0856, 29.17), 
			(97.0648, 143.95999999999998), (97.0649, 25.57), (97.1011, 407.30000000000007), (97.1012, 183.51), 
			(100.0757, 172.73000000000002), (100.0758, 519.67), (107.0855, 24.78), (107.0856, 47.36), 
			(107.0857, 31.57), (109.1012, 79.12), (109.1013, 65.53), (111.0805, 46.849999999999994), 
			(111.0806, 34.57), (111.0807, 8.69), (111.1168, 61.14), (111.1169, 65.44), (111.117, 64.74), 
			(114.0913, 35.36), (114.0914, 840.55), (121.1012, 38.26), (121.1013, 88.91), (123.1168, 9.49), 
			(123.1169, 7.29), (125.0961, 2.2), (125.0962, 7.59), (125.0963, 4.6), (128.1071, 696.2099999999999), 
			(135.117, 116.88), (139.1119, 4.9), (142.1227, 506.98999999999995), (142.1228, 109.49), 
			(149.1326, 42.75), (149.1327, 6.89), (156.1384, 372.02), (156.1385, 4.8), (163.1484, 3.1), 
			(163.1485, 5.0), (170.154, 115.39), (170.1541, 121.27), (184.1695, 19.08), (184.1698, 129.37), 
			(198.1854, 19.28), (198.1855, 75.22999999999999), (198.1856, 7.49), (206.271, 4.8), (212.201, 18.08), 
			(212.2012, 12.59), (212.2013, 8.29), (226.2168, 8.29), (226.217, 17.38), (240.2323, 11.59), 
			(240.2324, 11.39), (240.2325, 17.98), (247.242, 4.7), (247.2423, 25.37), (247.2424, 22.18), 
			(254.2481, 30.37), (254.2483, 19.08), (254.2484, 22.68), (265.253, 10.69), (265.2531, 7.19), 
			(268.2635, 18.68), (268.2637, 8.09), (268.2641, 6.49), (310.3106, 579.52), (310.3107, 1998.0)
		],
		2: [
			(52.0181, 2.2), (53.0259, 2.0), (53.0385, 16.689999999999998), (53.0386, 26.07), 
			(54.0337, 11.59), (54.0338, 31.97), (55.0289, 1.2), (55.029, 16.29), (55.0541, 2.2), 
			(55.0542, 9.3), (56.0117, 4.0), (56.0118, 9.59), (56.0129, 421.28), (56.013, 6.1), 
			(57.0445, 23.17), (57.0446, 35.86), (57.0447, 3.1), (65.0385, 5.4), (68.0492, 39.77), 
			(68.0493, 16.88), (69.0081, 80.02000000000001), (69.057, 22.88), (69.0571, 41.46), (69.0572, 3.6), 
			(70.0649, 47.65), (71.0601, 129.17000000000002), (71.0602, 200.30999999999997), (72.0423, 34.46), 
			(72.0424, 25.27), (72.0425, 14.49), (72.0441, 11511.470000000001), (79.0288, 5.0), (80.0366, 12.59), 
			(80.0367, 5.09), (80.0492, 1.6), (82.0649, 2.5), (83.0236, 1.1), (83.0237, 236.07), 
			(85.0757, 65.35000000000001), (85.0758, 72.11999999999999), (92.0492, 2.2), (92.0493, 1.4), 
			(95.0601, 211.79000000000002), (96.044, 1.3), (96.0441, 5.0), (96.0679, 2.4), (98.0596, 1.4), 
			(98.0597, 15.28), (98.0598, 333.46999999999997), (109.0757, 2.7), (109.0758, 84.02), (109.0759, 5.79), 
			(110.8948, 2.2), (110.8949, 1.2), (112.0754, 1.0), (112.0755, 3.9), (123.055, 5.0), (123.0551, 6.29), 
			(123.0552, 13.379999999999999), (124.0391, 39.57000000000001), (124.0392, 56.64), (124.0628, 2.9), 
			(124.0629, 26.57), (124.063, 31.47), (124.0868, 1.7), (125.0706, 3.9), (125.0708, 16.79), 
			(136.0868, 17.89), (136.0869, 19.78), (137.0707, 6.3999999999999995), (137.0708, 113.58), 
			(137.0709, 122.48), (138.0661, 1.3), (141.0657, 18.28), (141.0658, 21.58), (152.0816, 2.4), 
			(152.0817, 2.5), (152.0819, 1.4), (153.0894, 6.5), (153.0895, 13.29), (153.0896, 19.88), (153.0897, 6.79), 
			(168.113, 3275.82), (168.1131, 717.58), (180.0766, 669.73), (180.0767, 117.99), 
			(181.1446, 141.85999999999999), (181.1447, 56.739999999999995), (181.145, 1.8), (225.1247, 3.0), 
			(225.1276, 1.7), (225.1291, 3.3), (225.1344, 770.03), (225.1345, 2359.1400000000003), (225.1347, 6.89)
		]

	}

	compound_idx = args.compound_idx
	max_depth = args.max_depth
	h_prior = True 
	max_h_transfer = args.max_h_transfer
	vectorized = th.tensor(args.vectorized,dtype=th.bool)
	batch_size = th.tensor(args.batch_size,dtype=th.int32)
	tolerance = th.tensor(args.tolerance,dtype=th.float32)
	frag_max_time = None
	smiles = IDX_TO_SMILES[compound_idx]
	mol_d = frag_utils.extract_mol_info(smiles)
	hidden_size = 256
	spec = IDX_TO_SPEC[compound_idx]
	mzs, ints = zip(*spec)
	mzs = th.tensor(mzs,dtype=th.float32)
	ints = F.normalize(th.tensor(ints,dtype=th.float32),p=1,dim=0)
	prec_mz = calc_prec_mz(smiles,"[M+H]+")
	undirected_dag = args.undirected_dag

	if args.device_id == 0:
		device = th.device("cuda:0")
	else:
		device = th.device("cpu")

	# prepare data
	batch_size = args.batch_size
	atom_feats = [
		"a_onehot",
		"a_degree",
		"a_hybrid",
		"a_formal",
		"a_radical",
		"a_ring",
		"a_mass",
		"a_chiral",
	]
	bond_feats = ["b_degree"]
	pe_embed_k = 0
	bigraph = True
	mol_pyg = get_mol_graph(mol_d["mol"],atom_feats,bond_feats,pe_embed_k,bigraph)
	frag_d = timeit(frag_utils.compute_dags)(mol_d,max_depth,h_prior,max_h_transfer,frag_max_time)
	frag_pyg = get_frag_graph(frag_d["dag"],undirected_dag)
	# batch mol/dag stuff
	mol_pygs = [deepcopy(mol_pyg) for i in range(batch_size)]
	frag_pygs = [deepcopy(frag_pyg) for i in range(batch_size)]
	mass_diff = PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
	formula_mzses = [frag_d["formula_peak_mzs"][:,:args.num_isotopes].detach().clone()+mass_diff for i in range(batch_size)]
	formula_probses = [frag_d["formula_peak_probs"][:,:args.num_isotopes].detach().clone() for i in range(batch_size)]
	batched_frag_d = batch_mols_frags(mol_pygs,frag_pygs,formula_mzses,formula_probses)
	for k in batched_frag_d.keys():
		batched_frag_d[k] = batched_frag_d[k].to(device)
	# batch spectrum stuff
	spec_mzs = [mzs.detach().clone() for i in range(batch_size)]
	spec_ints = [ints.detach().clone() for i in range(batch_size)]
	spec_mzs, spec_ints, spec_batch_idxs = batch_func(spec_mzs,spec_ints)
	spec_mzs, spec_ints, spec_batch_idxs = spec_mzs.to(device), spec_ints.to(device), spec_batch_idxs.to(device)
	batched_frag_d["spec_mzs"] = spec_mzs
	batched_frag_d["spec_ints"] = spec_ints
	batched_frag_d["spec_idxs"] = spec_batch_idxs

	# prepare_model
	mol_kwargs = {
		"hidden_size": hidden_size,
		"gnn_node_feats": mol_pyg.x.shape[1],
		"gnn_edge_feats": mol_pyg.edge_attr.shape[1],
		"mpnn_type": "GINE"
	}
	frag_kwargs = {
		"hidden_size": hidden_size,
		"mpnn_type": "GINE"
		# node/edge feats are inferred
	}
	mlp_kwargs = {
		"hidden_size": hidden_size,
		"dropout": 0.0,
		"num_layers": 2,
		"use_residuals": True
		# input/output sizes inferred
	}
	num_depth = frag_d["max_depth"]+1 # +1 for the root/precursor node
	num_hs = frag_d["max_h_transfer"]
	num_elements = frag_utils.NUM_ELEMENTS
	int_embedder = "abs-sines"
	joint_model = JointPredictor(
		num_depth,
		num_hs,
		num_elements,
		mol_kwargs,
		frag_kwargs,
		mlp_kwargs,
		int_embedder
	)
	joint_model.to(device)

	sparse_cross_entropy_fn = get_sparse_cross_entropy_fn(
		args.out_dist,
		vectorized,
		args.tolerance,
		args.relative
	)
	
	if args.jit:
		th_dynamo.config.verbose = True
		dynamo_prof = th_dynamo.utils.CompileProfiler()
		# don't compile the model for now
		joint_model_forward = joint_model.forward
		loss_fn = th.compile(sparse_cross_entropy_fn,dynamic=True,backend=dynamo_prof)
	else:
		joint_model_forward = joint_model.forward
		loss_fn = sparse_cross_entropy_fn

	if args.time:
		joint_model_forward = torch_time(joint_model_forward)
		loss_fn = torch_time(loss_fn)

	optim = th.optim.AdamW(joint_model.parameters(), lr=args.lr)
	optim.zero_grad()

	metrics = {
		"step": [],
		"cross_entropy_loss": [],
		"recall": [],
		"weighted_recall": [],
		"topk_weighted_recall": [],
		"topk_frac": [],
		"cosine_similarity": []
	}

	def get_pred_true():
		pred_mzs, pred_logits, pred_batch_idxs = joint_model_forward(
			**deepcopy(batched_frag_d)
		)
		if args.random_targ or spec is None:
			with th_temp_seed(420):
				# select random subset of pred peaks
				true_mask = th.rand_like(pred_mzs)<0.1
				true_batch_idxs = pred_batch_idxs[true_mask]
				# add slight mass jitter
				true_mzs = pred_mzs[true_mask] + 0.0002*th.rand_like(pred_mzs[true_mask]) - 0.0001
				# random intensities
				true_logits = scatter_normalize(
					th.log(th.rand_like(pred_logits[true_mask])),
					true_batch_idxs,
					dim=0
				)
		else:
			true_mzs = batched_frag_d["spec_mzs"]
			true_logits = th.log(batched_frag_d["spec_ints"])
			true_batch_idxs = batched_frag_d["spec_idxs"]
		return pred_mzs, pred_logits, pred_batch_idxs, true_mzs, true_logits, true_batch_idxs

	best_d = {}

	for i in range(args.num_iters):
		print(f">> iter {i}")
		pred_mzs, pred_logits, pred_batch_idxs, true_mzs, true_logits, true_batch_idxs = get_pred_true()
		pred_entropy = th.mean(calculate_spectrum_entropy(pred_logits,pred_batch_idxs))
		true_entropy = th.mean(calculate_spectrum_entropy(true_logits,true_batch_idxs))
		topk_weighted_recall, topk_frac = calculate_topk_weighted_recall(
			true_mzs,
			true_logits,
			true_batch_idxs,
			pred_mzs,
			pred_logits,
			pred_batch_idxs
		)
		topk_weighted_recall = th.mean(topk_weighted_recall)
		topk_frac = th.mean(topk_frac)
		recall, weighted_recall = calculate_recalls(
			true_mzs,
			true_logits,
			true_batch_idxs,
			pred_mzs,
			pred_logits,
			pred_batch_idxs
		)
		recall = th.mean(recall)
		weighted_recall = th.mean(weighted_recall)
		cosine_similarity = calculate_cosine_similarity(
			true_mzs,
			th.exp(scatter_normalize(true_logits,true_batch_idxs,dim=0)),
			true_batch_idxs,
			pred_mzs,
			th.exp(scatter_normalize(pred_logits,pred_batch_idxs,dim=0)),
			pred_batch_idxs
		)
		cosine_similarity = th.mean(cosine_similarity)
		print(f"> pred logits = {pred_logits.mean().item()}, true logits = {true_logits.mean().item()}")
		print(f"> pred entropy = {pred_entropy.item()}, true entropy = {true_entropy.item()}")
		print(f"> recall = {recall.item()}, weighted recall = {weighted_recall.item()}")
		print(f"> topk weighted recall = {topk_weighted_recall.item()}, topk frac = {topk_frac.item()}")
		print(f"> cosine similarity = {cosine_similarity.item()}")
		# calculate loss
		loss = loss_fn(
			true_mzs,
			true_logits,
			true_batch_idxs,
			pred_mzs,
			pred_logits,
			pred_batch_idxs,
			batch_size
		)
		print(f"> loss = {loss:.4f}")
		# update metrics
		metrics["step"].append(i+1)
		metrics["cross_entropy_loss"].append(loss.item())
		metrics["recall"].append(recall.item())
		metrics["weighted_recall"].append(weighted_recall.item())
		metrics["topk_weighted_recall"].append(topk_weighted_recall.item())
		metrics["topk_frac"].append(topk_frac.item())
		metrics["cosine_similarity"].append(cosine_similarity.item())
		# save best model
		if loss.item() < best_d.get("cross_entropy_loss",np.inf):
			best_d["iter"] = i
			best_d["params"] = deepcopy(joint_model.state_dict())
			best_d["cross_entropy_loss"] = loss.item()
			best_d["recall"] = recall.item()
			best_d["weighted_recall"] = weighted_recall.item()
			best_d["topk_weighted_recall"] = topk_weighted_recall.item()
			best_d["cosine_similarity"] = cosine_similarity.item()
			best_d["pred_mzs"] = pred_mzs.detach().clone()
			best_d["pred_logits"] = pred_logits.detach().clone()
			best_d["pred_batch_idxs"] = pred_batch_idxs.detach().clone()
			best_d["true_mzs"] = true_mzs.detach().clone()
			best_d["true_logits"] = true_logits.detach().clone()
			best_d["true_batch_idxs"] = true_batch_idxs.detach().clone()
		# backprop
		loss.backward()
		# update parameters
		optim.step()
		# zero out gradients
		optim.zero_grad()
	
	# plot spectrum
	pred_mzs = best_d["pred_mzs"]
	pred_logits = best_d["pred_logits"]
	pred_batch_idxs = best_d["pred_batch_idxs"]
	true_mzs = best_d["true_mzs"]
	true_logits = best_d["true_logits"]
	true_batch_idxs = best_d["true_batch_idxs"]
	best_iter = best_d["iter"]
	loss = best_d["cross_entropy_loss"]
	recall = best_d["recall"]
	weighted_recall = best_d["weighted_recall"]
	b_true_mask = true_batch_idxs==0
	b_true_mzs = true_mzs[b_true_mask]
	b_true_ints = th.exp(true_logits[b_true_mask]) / th.exp(th.logsumexp(true_logits[b_true_mask],0))
	b_pred_mask = pred_batch_idxs==0
	b_pred_mzs = pred_mzs[b_pred_mask]
	b_pred_ints = th.exp(pred_logits[b_pred_mask]) / th.exp(th.logsumexp(pred_logits[b_pred_mask],0))
	print("> prec_mz",prec_mz)
	print("> prec_mz in true_mzs",th.any(calculate_match_mzs(th.tensor([prec_mz],device=b_true_mzs.device),b_true_mzs,use_ppm=True)))
	print("> prec_mz in pred_mzs",th.any(calculate_match_mzs(th.tensor([prec_mz],device=b_pred_mzs.device),b_pred_mzs,use_ppm=True)))
	plot_spectra_sparse(
		b_true_mzs.cpu().numpy(),
		b_true_ints.cpu().numpy(),
		b_pred_mzs.cpu().numpy(),
		b_pred_ints.cpu().numpy(),
		prec_mz,
		smiles,
		f"figs/fit_model_{compound_idx}_spectra.png",
		custom_title=f"iter = {best_iter}, loss = {loss:.5f}, recall = {recall:.3f}, w_recall = {weighted_recall:.3f}"
	)

	# plot training metrics
	plot_dual_axis(
		metrics["step"],
		{
			"ce_loss": metrics["cross_entropy_loss"]
		},
		{
			"topk_wrecall": metrics["topk_weighted_recall"],
			"cos_sim": metrics["cosine_similarity"]
		},
		f"figs/fit_model_{compound_idx}_curves.png"
	)

	# get memory usage
	if args.device_id != -1 and th.cuda.is_available():
		cuda_max_memory = th.cuda.max_memory_allocated(device=device)/1e9
		print(f"> GPU memory: {cuda_max_memory:.2f} GB")

	if args.jit:
		print(dynamo_prof.report())