from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL
from frag_gnn.dataset import SpecMolFragDataset, init_dataloader
from frag_gnn.utils.misc_utils import to_device, scatter_reduce, scatter_var, scatter_argmax, flatten_lol, print_shapes, get_tensor_memory_usage, LOG_ZERO

import numpy as np
import torch as th
from tqdm import tqdm
import wandb
import tempfile
import os

# load configs
print(">>> Loading configs")
template_fp = "config/template.yml"
custom_fp = "config/nb_entropy/d3_baseline.yml"
config_d = load_config(template_fp, custom_fp)
# for debugging, use small dataset
config_d["spec_fp"] = "data/proc/debug_small/spec_df.pkl"
config_d["mol_fp"] = "data/proc/debug_small/mol_df.pkl"
config_d["split_dp"] = "data/split/debug_small"

# load datasets
print(">>> Loading datasets")
ds = SpecMolFragDataset(split="val", **config_d)
dl = init_dataloader(ds, config_d)

model_to_run_id = {
	"d3_baseline": "45csl4n4",
	"d3_posn": "31dlwe6a"
}

model_to_vals = {}
num_models = len(model_to_run_id.keys())

for model, run_id in model_to_run_id.items():

	# load model
	print(f">>> Loading model {model} - {run_id}")
	with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dp:
		api = wandb.Api()
		run = api.run(f"frag-gnn/frag-gnn/{run_id}")
		ckpt_file_local = run.file("ckpt/last.ckpt").download(root=temp_dp,replace=False)
		ckpt_fp = ckpt_file_local.name
		model = FragGNNPL.load_from_checkpoint(ckpt_fp)
	device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
	model.to(device)
	model.eval()

	# make predictions on validation data
	print(">>> Iterating over data")
	batch_cutoff = 3
	cum_num_datapoints = 0
	cum_num_nodes = 0
	cum_num_formulas = 0
	cum_num_nb_nodes = 0
	BATCH_IDX_VARS = [
		"pred_batch_idxs",
		"pred_formula_batch_idxs",
		"pred_node_batch_idxs",
		"pred_joint_batch_idxs",
		"pred_nb_node_batch_idxs",
		"pred_nb_joint_batch_idxs",
		"pred_nb_node_node_batch_idxs"
	]
	NODE_IDX_VARS = [
		"pred_node_node_idxs",
		"pred_joint_node_idxs"
	]
	FORMULA_IDX_VARS = [
		"pred_formula_formula_idxs",
		"pred_joint_formula_idxs",
		"pred_nb_joint_formula_idxs"
	]
	NB_NODE_IDX_VARS = [
		"pred_nb_node_node_idxs",
		"pred_nb_joint_node_idxs",
		"pred_nb_node_node_node_idxs"
	]
	vals = {}
	with th.no_grad():    
		for b_idx, b_input in tqdm(enumerate(dl),total=len(dl)):
			# transfer data, make predictions
			b_input = to_device(b_input, device)
			b_output = model._common_step(b_input, split="val")
			b_output = to_device(b_output, "cpu")
			# print(b_output["pred_formula_logprobs"].shape[0])
			# print(b_output["pred_joint_formula_idxs"].max()+1)
			# rebatch stuff
			b_num_datapoints = b_output["pred_batch_idxs"].max()+1
			b_num_nodes = b_output["pred_node_node_idxs"].max()+1
			b_num_formulas = b_output["pred_formula_formula_idxs"].max()+1
			b_num_nb_nodes = b_output["pred_nb_node_node_idxs"].max()+1
			for k in BATCH_IDX_VARS:
				b_output[k] = b_output[k] + cum_num_datapoints
			for k in NODE_IDX_VARS:
				b_output[k] = b_output[k] + cum_num_nodes
			for k in FORMULA_IDX_VARS:
				b_output[k] = b_output[k] + cum_num_formulas
			for k in NB_NODE_IDX_VARS:
				b_output[k] = b_output[k] + cum_num_nb_nodes
			# update counts
			cum_num_datapoints += b_num_datapoints
			cum_num_nodes += b_num_nodes
			cum_num_formulas += b_num_formulas
			cum_num_nb_nodes += b_num_nb_nodes
			# update dict
			for k in b_output.keys():
				if b_output[k] is None:
					continue
				if k not in vals.keys():
					assert b_idx == 0
					vals[k] = [b_output[k]]
				else:
					assert b_idx > 0
					vals[k].append(b_output[k])
			# early exit
			if b_idx >= batch_cutoff:
				break
	print(f"> num_datapoints = {cum_num_datapoints}")
	print(f"> num_nodes = {cum_num_nodes}")
	print(f"> num_formulas = {cum_num_formulas}")
	print(f"> num_nb_nodes = {cum_num_nb_nodes}")
	for k in vals.keys():
		if isinstance(vals[k][0], list):
			vals[k] = flatten_lol(vals[k])
		else:
			assert isinstance(vals[k][0], th.Tensor), type(vals[k][0])
			if vals[k][0].dim() == 0:
				vals[k] = th.stack(vals[k], dim=0)
			else:
				vals[k] = th.cat(vals[k], dim=0)
	# print_shapes(vals)
	print(vals["pred_formula_logprobs"].shape[0])
	print(vals["pred_joint_formula_idxs"].max()+1)
	model_to_vals[model] = vals

print("> Calculating metrics")
ensemble_metrics = {}

print("> Top-1 Agreement P(n|f) and P(n'|f)")
# top-1 agreement P(n|f) and P(n'|f)
formula_node_argmaxes = []
# formula_node_formula_idxses = []
nb_formula_node_argmaxes = []
# nb_formula_node_formula_idxses = []
for model, vals in model_to_vals.items():
	# top-1 agreement P(n|f)
	formula_node_logprobs = vals["pred_formula_node_logprobs"]
	formula_node_formula_idxs = vals["pred_joint_formula_idxs"]
	formula_node_node_idxs = vals["pred_joint_node_idxs"]
	formula_formula_idxs = vals["pred_formula_formula_idxs"]
	formula_node_argmax = scatter_argmax(
		formula_node_logprobs, 
		formula_node_formula_idxs,
		formula_node_node_idxs,
		dim_size=formula_formula_idxs.max()+1
	)
	formula_node_argmaxes.append(formula_node_argmax)
	# formula_node_formula_idxses.append(formula_node_formula_idxs)
	# top-1 agreement P(n'|f)
	nb_formula_node_logprobs = vals["pred_nb_formula_node_logprobs"]
	nb_formula_node_formula_idxs = vals["pred_nb_joint_formula_idxs"]
	nb_formula_node_node_idxs = vals["pred_nb_joint_node_idxs"]
	nb_formula_node_argmax = scatter_argmax(
		nb_formula_node_logprobs, 
		nb_formula_node_formula_idxs,
		nb_formula_node_node_idxs,
		dim_size=formula_formula_idxs.max()+1
	)
	nb_formula_node_argmaxes.append(nb_formula_node_argmax)
	# nb_formula_node_formula_idxses.append(nb_formula_node_formula_idxs)
# calculate majority vote and agreements
# P(n|f)
formula_node_argmaxes = th.stack(formula_node_argmaxes, dim=0)
# formula_node_formula_idxses = th.stack(formula_node_formula_idxses, dim=0)
# assert th.all(th.all(formula_node_formula_idxses == formula_node_formula_idxses[0:1],dim=0))
formula_node_argmax_majority = th.mode(formula_node_argmaxes,dim=0).values
formula_node_argmax_agrees = formula_node_argmaxes == formula_node_argmax_majority.unsqueeze(0)
# P(n'|f)
nb_formula_node_argmaxes = th.stack(nb_formula_node_argmaxes, dim=0)
# nb_formula_node_formula_idxses = th.stack(nb_formula_node_formula_idxses, dim=0)
# assert th.all(th.all(nb_formula_node_formula_idxses == nb_formula_node_formula_idxses[0:1],dim=0))
nb_formula_node_argmax_majority = th.mode(nb_formula_node_argmaxes,dim=0).values
nb_formula_node_argmax_agrees = nb_formula_node_argmaxes == nb_formula_node_argmax_majority.unsqueeze(0)
mean_agreements = []
for model_idx, (model, vals) in enumerate(model_to_vals.items()):
	formula_node_argmax_agree = formula_node_argmax_agrees[model_idx]    
	formula_node_logprobs = vals["pred_formula_node_logprobs"]
	formula_node_formula_idxs = vals["pred_joint_formula_idxs"]
	formula_logprobs = vals["pred_formula_logprobs"]
	formula_batch_idxs = vals["pred_formula_batch_idxs"]
	# weighted average over formulae, using formula probs as weights
	mean_argmax_agree = scatter_reduce(
		formula_node_argmax_agree.float() * th.exp(formula_logprobs), 
		formula_batch_idxs,
		reduce="sum",
		dim_size=th.max(formula_batch_idxs)+1
	)
	mean_agreements.append(mean_argmax_agree)
	# # average over datapoints
	# mean_argmax_agree = th.mean(mean_argmax_agree, dim=0)
	# mean_agreements.append(mean_argmax_agree)
mean_agreements = th.stack(mean_agreements, dim=0)
print(mean_agreements.shape)
print(th.mean(mean_agreements, dim=1))

# mean and variance of H(n|f) and H(n'|f)
print("> Mean and Variance of H(n|f)")
# get H(n|f) and H(n'|f) per datapoint per model
formula_node_es = []
# formula_node_batch_idxses = []
nb_formula_node_es = []
# nb_formula_node_batch_idxses = []
for model, vals in model_to_vals.items():
	formula_node_e = vals["formula_node_e"]
	formula_node_batch_idxs = vals["pred_formula_batch_idxs"]
	nb_formula_node_e = vals["nb_formula_node_e"]
	nb_formula_node_batch_idxs = vals["pred_nb_joint_batch_idxs"]
	formula_node_es.append(formula_node_e)
	# formula_node_batch_idxses.append(formula_node_batch_idxs)
	nb_formula_node_es.append(nb_formula_node_e)
	# nb_formula_node_batch_idxses.append(nb_formula_node_batch_idxs)
formula_node_es = th.stack(formula_node_es, dim=0)
# formula_node_batch_idxses = th.stack(formula_node_batch_idxses, dim=0)
# assert th.all(th.all(formula_node_batch_idxses == formula_node_batch_idxses[0:1],dim=0))
nb_formula_node_es = th.stack(nb_formula_node_es, dim=0)
# nb_formula_node_batch_idxses = th.stack(nb_formula_node_batch_idxses, dim=0)
# assert th.all(th.all(nb_formula_node_batch_idxses == nb_formula_node_batch_idxses[0:1],dim=0))
# calculate mean and std of H(n|f) per datapoint
formula_node_es_mean = th.mean(formula_node_es, dim=0)
formula_node_es_std = th.std(formula_node_es, dim=0)
# average over datapoints
print(th.mean(formula_node_es_mean), th.mean(formula_node_es_std))

# # JSS(n|f) and JSS(n'|f)
# # get P(n|f), P(f), P(n'|f)
formula_node_logprobses = []
formula_node_formula_idxses = []
formula_node_node_idxses = []
nb_formula_node_logprobses = []
nb_formula_node_formula_idxses = []
nb_formula_node_node_idxses = []
formula_logprobses = []
formula_formula_idxses = []
formula_batch_idxes = []
for model, vals in model_to_vals.items():
	formula_node_logprobs = vals["pred_formula_node_logprobs"]
	formula_node_formula_idxs = vals["pred_joint_formula_idxs"]
	formula_node_node_idxs = vals["pred_joint_node_idxs"]
	nb_formula_node_logprobs = vals["pred_nb_formula_node_logprobs"]
	nb_formula_node_formula_idxs = vals["pred_nb_joint_formula_idxs"]
	nb_formula_node_node_idxs = vals["pred_nb_joint_node_idxs"]
	formula_logprobs = vals["pred_formula_logprobs"]
	formula_formula_idxs = vals["pred_formula_formula_idxs"]
	formula_batch_idxs = vals["pred_formula_batch_idxs"]
	formula_node_logprobses.append(formula_node_logprobs)
	formula_node_formula_idxses.append(formula_node_formula_idxs)
	formula_node_node_idxses.append(formula_node_node_idxs)
	nb_formula_node_logprobses.append(nb_formula_node_logprobs)
	nb_formula_node_formula_idxses.append(nb_formula_node_formula_idxs)
	nb_formula_node_node_idxses.append(nb_formula_node_node_idxs)
	formula_logprobses.append(formula_logprobs)
	formula_formula_idxses.append(formula_formula_idxs)
	formula_batch_idxes.append(formula_batch_idxs)
formula_node_logprobses = th.stack(formula_node_logprobses, dim=0)
formula_node_formula_idxses = th.stack(formula_node_formula_idxses, dim=0)
formula_node_node_idxses = th.stack(formula_node_node_idxses, dim=0)
nb_formula_node_logprobses = th.stack(nb_formula_node_logprobses, dim=0)
nb_formula_node_formula_idxses = th.stack(nb_formula_node_formula_idxses, dim=0)
nb_formula_node_node_idxses = th.stack(nb_formula_node_node_idxses, dim=0)
formula_logprobses = th.stack(formula_logprobses, dim=0)
formula_formula_idxses = th.stack(formula_formula_idxses, dim=0)
formula_batch_idxes = th.stack(formula_batch_idxes, dim=0)

# define sparse kl divergence
def sparse_kl(log_p,log_q,b_idx,dim_size):
	p = th.exp(log_p)
	kl = scatter_reduce(
		p * (log_p - th.clamp(log_q,min=LOG_ZERO)),
		b_idx,
		reduce="sum",
		dim_size=dim_size
	)
	kl = th.clamp(kl,min=0.0)
	return kl

# calculate pairwise comparisons
pairwise_jss = th.zeros(
	(num_models,num_models),
	dtype=formula_logprobs.dtype,
	device=formula_logprobs.device
)
for i in range(num_models):
	for j in range(i+1,num_models):
		# merge formula predictions
		assert th.all(formula_formula_idxses[i] == formula_formula_idxses[j],dim=0)
		m_formula_logprobs_ij = th.stack([formula_logprobses[i],formula_logprobses[j]],dim=0)
		m_formula_logprobs_ij = th.logsumexp(m_formula_logprobs_ij,dim=0) - np.log(2.0)
		# merge formula-node predictions
		assert th.all(formula_node_formula_idxses[i] == formula_node_formula_idxses[j],dim=0)
		assert th.all(formula_node_node_idxses[i] == formula_node_node_idxses[j],dim=0)
		m_formula_node_logprobs_ij = th.stack([formula_node_logprobses[i],formula_node_logprobses[j]],dim=0)
		m_formula_node_logprobs_ij = th.logsumexp(m_formula_node_logprobs_ij,dim=0) - np.log(2.0)
		# compute kls
		kl_ij = sparse_kl(
			formula_node_logprobses[i], 
			m_formula_node_logprobs_ij, 
			formula_node_formula_idxses[i],
			formula_formula_idxses[i].max()+1
		)
		kl_ji = sparse_kl(
			formula_node_logprobses[j], 
			m_formula_node_logprobs_ij, 
			formula_node_formula_idxses[j],
			formula_formula_idxses[j].max()+1
		)
		# compute jsds
		import pdb; pdb.set_trace()
		jsd_ij = 0.5 * (kl_ij + kl_ji)
		jsd_b_ij = scatter_reduce(
			jsd_ij * th.exp(m_formula_logprobs_ij),
			formula_batch_idxes[i],
			reduce="sum",
			dim_size=formula_batch_idxes[i].max()+1
		)
		jsd_m_ij = th.mean(jsd_b_ij,dim=0)
		pairwise_jss[i,j] = 1. - jsd_m_ij
print(pairwise_jss)
import pdb; pdb.set_trace()

