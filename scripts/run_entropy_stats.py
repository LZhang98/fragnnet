from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL
from frag_gnn.dataset import SpecMolFragDataset, init_dataloader
from frag_gnn.utils.misc_utils import to_device, scatter_reduce, scatter_var, scatter_argmax, flatten_lol, print_shapes, get_tensor_memory_usage, LOG_ZERO, EPS, booltype

import numpy as np
import torch as th
from tqdm import tqdm
import wandb
import tempfile
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=booltype, default=False)
parser.add_argument("--mode", type=str, choices=["baseline","entropy"], default="entropy")
parser.add_argument("--batch_cutoff", type=int, default=int(1e6))
args = parser.parse_args()

debug = args.debug
mode = args.mode
batch_cutoff = args.batch_cutoff
pd.options.display.float_format = '{:.4f}'.format

# load configs
print(">>> Loading configs")
template_fp = "config/template.yml"
custom_fp = "config/nb_entropy/d3_baseline.yml"
config_d = load_config(template_fp, custom_fp)

if debug:
	# for debugging, use small dataset
	config_d["spec_fp"] = "data/proc/debug_small/spec_df.pkl"
	config_d["mol_fp"] = "data/proc/debug_small/mol_df.pkl"
	config_d["split_dp"] = "data/split/debug_small"

# load datasets
print(">>> Loading datasets")
ds = SpecMolFragDataset(split="val", **config_d)
dl = init_dataloader(ds, config_d)

if debug:
	if mode == "entropy":
		model_to_run_id = {
			"d3_baseline": "45csl4n4",
			"d3_posfn": "gzb25hmk",
			"d3_negfn": "phg6ta1y"
		}
	else:
		assert mode == "baseline"
		model_to_run_id = {
			"d3_baseline":  "45csl4n4",
			"d3_baseline_s0": "u9wy1bez",
			"d3_baseline_s1": "b41o0odi"
		}
else:
	if mode == "entropy":
		model_to_run_id = {
			"d3_baseline": "45csl4n4",
			# "d3_posn": "31dlwe6a",
			# "d3_negn": "7y090yc5",
			# "d3_posnf": "rwoixgf0",
			# "d3_negnf": "63qysyr1",
			"d3_posfn": "gzb25hmk",
			"d3_negfn": "phg6ta1y",
			# "d3_nbposn": "q90tr6zy",
			# "d3_nbnegn": "6yo3kgf5",
			# "d3_nbposnf": "6yo3kgf5",
			# "d3_nbnegnf": "b73djt3d",
			"d3_nbposfn": "j8nixjvp",
			"d3_nbnegfn": "xf93r2oe"
		}
	else:
		assert mode == "baseline"
		model_to_run_id = {
			"d3_baseline":  "45csl4n4",
			"d3_baseline_s0": "u9wy1bez",
			"d3_baseline_s1": "b41o0odi",
			"d3_baseline_s2": "1i6owhp1",
			"d3_baseline_s3": "kduczhxt",
			# "d3_baseline_s4": "ppiru3q2",
			# "d3_baseline_s5": "5anh3g2x",
			# "d3_baseline_s6": "4oaj4vf6",
			# "d3_baseline_s7": "p05lunqn",
			# "d3_baseline_s8": "kf2fuiya",
			# "d3_baseline_s9": "xzvo8kly"
		}

model_to_vals = {}
num_models = len(model_to_run_id.keys())

for model_name, run_id in model_to_run_id.items():

	# load model
	print(f">>> Loading model {model_name} - {run_id}")
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
			b_input = to_device(b_input, device,non_blocking=False)
			b_output = model._common_step(b_input, split="val")
			b_output = to_device(b_output, "cpu",non_blocking=False)
			# print(th.any(b_output["nb_formula_node_ne"] < -0.001))
			if th.any(b_output["nb_formula_node_ne"] < -0.001):
				print(b_idx)
				print(th.any(b_output["nb_formula_node_ne"] < -0.001))
				print(th.nonzero(b_output["nb_formula_node_ne"] < -0.001))
				import pdb; pdb.set_trace()
				# import pdb; pdb.set_trace()
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
	model_to_vals[model_name] = vals

print("> Calculating metrics")
ensemble_metrics = {}

def select_model_vals(model_to_vals, keys):

	select_d = {k: [] for k in keys}
	for model, vals in model_to_vals.items():
		for k in keys:
			select_d[k].append(vals[k])
	select_d = {k: th.stack(v, dim=0) for k,v in select_d.items()}
	return select_d

def log_mean(log_p,support_size,dim=0):
	return th.logsumexp(log_p,dim=dim) - np.log(support_size)

def log_std(log_p,support_size,dim=0,correction=1):
	log_mean_p = log_mean(log_p,support_size,dim=dim)
	log_mean_p_sq = log_mean(2*log_p,support_size,dim=dim)
	log_var = th.log(th.exp(log_mean_p_sq) - th.exp(2*log_mean_p))
	log_std = 0.5*(log_var + np.log(support_size) - np.log(support_size - correction))
	return log_std

### spectrum similarity metrics
print("> Spectrum Similarity Metrics")
spec_sims = th.stack([vals["cos_hun"] for vals in model_to_vals.values()], dim=0)
ensemble_metrics["spec_sims"] = spec_sims
print(th.mean(spec_sims, dim=1))
spec_sims_mean = th.mean(spec_sims, dim=0)
spec_sims_std = th.std(spec_sims, dim=0)
spec_sims_cv = spec_sims_std / th.clamp(spec_sims_mean, min=EPS)
ensemble_metrics["spec_sims_mean"] = spec_sims_mean
ensemble_metrics["spec_sims_std"] = spec_sims_std
ensemble_metrics["spec_sims_cv"] = spec_sims_cv

### top-1 agreement P(n|f) and P(n'|f)
print("> Top-1 Agreement P(n|f) and P(n'|f)")

def unique_counts(x,dim=0):
	x = th.sort(x,dim=dim)[0]
	k = x.shape[dim]
	c = th.ones_like(th.index_select(x,dim,th.tensor(0,device=x.device)).squeeze(dim),dtype=th.long)
	for i in range(1,k):
		cur = th.index_select(x,dim,th.tensor(i,device=x.device)).squeeze(dim)
		prev = th.index_select(x,dim,th.tensor(i-1,device=x.device)).squeeze(dim)
		c += (cur != prev).long()
	assert th.all(c >= 1)
	assert th.all(c <= k)
	return c

fn_argmaxes = []
nb_fn_argmaxes = []
for model, vals in model_to_vals.items():
	# top-1 agreement P(n|f)
	fn_logprobs = vals["pred_formula_node_logprobs"]
	fn_formula_idxs = vals["pred_joint_formula_idxs"]
	fn_node_idxs = vals["pred_joint_node_idxs"]
	f_formula_idxs = vals["pred_formula_formula_idxs"]
	fn_argmax = scatter_argmax(
		fn_logprobs, 
		fn_formula_idxs,
		fn_node_idxs,
		dim_size=f_formula_idxs.max()+1
	)
	fn_argmaxes.append(fn_argmax)
	# top-1 agreement P(n'|f)
	nb_fn_logprobs = vals["pred_nb_formula_node_logprobs"]
	nb_fn_formula_idxs = vals["pred_nb_joint_formula_idxs"]
	nb_fn_node_idxs = vals["pred_nb_joint_node_idxs"]
	nb_fn_argmax = scatter_argmax(
		nb_fn_logprobs, 
		nb_fn_formula_idxs,
		nb_fn_node_idxs,
		dim_size=f_formula_idxs.max()+1
	)
	nb_fn_argmaxes.append(nb_fn_argmax)
# calculate majority vote and agreements
# P(n|f)
fn_argmaxes = th.stack(fn_argmaxes, dim=0)
fn_argmax_majority = th.mode(fn_argmaxes,dim=0).values
fn_argmax_agrees = fn_argmaxes == fn_argmax_majority.unsqueeze(0)
fn_argmax_counts = unique_counts(fn_argmaxes,dim=0)
fn_argmax_probs = th.mean(fn_argmax_agrees.float(),dim=0)
fn_argmax_cons = th.all(fn_argmax_agrees,dim=0)
# P(n'|f)
nb_fn_argmaxes = th.stack(nb_fn_argmaxes, dim=0)
nb_fn_argmax_majority = th.mode(nb_fn_argmaxes,dim=0).values
nb_fn_argmax_agrees = nb_fn_argmaxes == nb_fn_argmax_majority.unsqueeze(0)
nb_fn_argmax_counts = unique_counts(nb_fn_argmaxes,dim=0)
nb_fn_argmax_probs = th.mean(nb_fn_argmax_agrees.float(),dim=0)
nb_fn_argmax_cons = th.all(nb_fn_argmax_agrees,dim=0)
# compute agreements
keys = [
	"pred_formula_logprobs",
	"pred_formula_batch_idxs",
	"pred_joint_node_idxs",
	"pred_joint_formula_idxs",
	"pred_joint_batch_idxs"
]
select_d = select_model_vals(model_to_vals, keys)
f_logprobses = select_d["pred_formula_logprobs"]
f_batch_idxses = select_d["pred_formula_batch_idxs"]
fn_node_idxses = select_d["pred_joint_node_idxs"]
fn_formula_idxses = select_d["pred_joint_formula_idxs"]
fn_batch_idxses = select_d["pred_joint_batch_idxs"]
del select_d
# compute averages
fn_argmax_agree_fws, fn_argmax_agree_uws = [], []
nb_fn_argmax_agree_fws, nb_fn_argmax_agree_uws = [], []
for model_idx in range(num_models):
	# P(f)
	f_logprobs = f_logprobses[model_idx]
	f_batch_idxs = f_batch_idxses[model_idx]
	# argmax
	fn_argmax_agree = fn_argmax_agrees[model_idx]
	# average with formula probs
	fn_argmax_agree_fw = scatter_reduce(
		fn_argmax_agree.float() * th.exp(f_logprobs),
		f_batch_idxs,
		reduce="sum",
		dim_size=th.max(f_batch_idxs)+1
	)
	fn_argmax_agree_fws.append(fn_argmax_agree_fw)
	# average with uniform probs
	fn_argmax_agree_uw = scatter_reduce(
		fn_argmax_agree.float(),
		f_batch_idxs,
		reduce="mean",
		dim_size=th.max(f_batch_idxs)+1,
		include_self=False
	)
	fn_argmax_agree_uws.append(fn_argmax_agree_uw)
	# argmax
	nb_fn_argmax_agree = nb_fn_argmax_agrees[model_idx]
	# average with formula probs
	nb_fn_argmax_agree_fw = scatter_reduce(
		nb_fn_argmax_agree.float() * th.exp(f_logprobs),
		f_batch_idxs,
		reduce="sum",
		dim_size=th.max(f_batch_idxs)+1
	)
	nb_fn_argmax_agree_fws.append(nb_fn_argmax_agree_fw)
	# average with uniform probs
	nb_fn_argmax_agree_uw = scatter_reduce(
		nb_fn_argmax_agree.float(),
		f_batch_idxs,
		reduce="mean",
		dim_size=th.max(f_batch_idxs)+1,
		include_self=False
	)
	nb_fn_argmax_agree_uws.append(nb_fn_argmax_agree_uw)
fn_argmax_agree_fws = th.stack(fn_argmax_agree_fws, dim=0)
fn_argmax_agree_uws = th.stack(fn_argmax_agree_uws, dim=0)
nb_fn_argmax_agree_fws = th.stack(nb_fn_argmax_agree_fws, dim=0)
nb_fn_argmax_agree_uws = th.stack(nb_fn_argmax_agree_uws, dim=0)
fn_argmax_agree_fw_mean = th.mean(fn_argmax_agree_fws,dim=0)
fn_argmax_agree_uw_mean = th.mean(fn_argmax_agree_uws,dim=0)
nb_fn_argmax_agree_fw_mean = th.mean(nb_fn_argmax_agree_fws,dim=0)
nb_fn_argmax_agree_uw_mean = th.mean(nb_fn_argmax_agree_uws,dim=0)

argmax_d = {}
argmax_d["fn_argmax_agree_fw_mean"] = fn_argmax_agree_fw_mean
argmax_d["fn_argmax_agree_uw_mean"] = fn_argmax_agree_uw_mean
argmax_d["nb_fn_argmax_agree_fw_mean"] = nb_fn_argmax_agree_fw_mean
argmax_d["nb_fn_argmax_agree_uw_mean"] = nb_fn_argmax_agree_uw_mean

num_formulae = th.max(f_batch_idxses[0])+1

# deal with counts
f_logprobs_mean = log_mean(f_logprobses,num_models)
fn_argmax_counts_fw = scatter_reduce(
	fn_argmax_counts.float() * th.exp(f_logprobs_mean),
	f_batch_idxses[0],
	reduce="sum",
	dim_size=num_formulae
)
nb_fn_argmax_counts_fw = scatter_reduce(
	nb_fn_argmax_counts.float() * th.exp(f_logprobs_mean),
	f_batch_idxs,
	reduce="sum",
	dim_size=num_formulae
)
fn_argmax_counts_uw = scatter_reduce(
	fn_argmax_counts.float(),
	f_batch_idxses[0],
	reduce="mean",
	dim_size=num_formulae,
	include_self=False
)
nb_fn_argmax_counts_uw = scatter_reduce(
	nb_fn_argmax_counts.float(),
	f_batch_idxses[0],
	reduce="mean",
	dim_size=num_formulae,
	include_self=False
)
argmax_d["n_argmax_counts_fw"] = fn_argmax_counts_fw
argmax_d["fn_argmax_counts_uw"] = fn_argmax_counts_uw
argmax_d["nb_fn_argmax_counts_fw"] = nb_fn_argmax_counts_fw
argmax_d["nb_fn_argmax_counts_uw"] = nb_fn_argmax_counts_uw

# deal with probs
fn_argmax_probs_fw = scatter_reduce(
	fn_argmax_probs.float() * th.exp(f_logprobs_mean),
	f_batch_idxses[0],
	reduce="sum",
	dim_size=num_formulae
)
nb_fn_argmax_probs_fw = scatter_reduce(
	nb_fn_argmax_probs.float() * th.exp(f_logprobs_mean),
	f_batch_idxses[0],
	reduce="sum",
	dim_size=num_formulae
)
fn_argmax_probs_uw = scatter_reduce(
	fn_argmax_probs.float(),
	f_batch_idxses[0],
	reduce="mean",
	dim_size=num_formulae,
	include_self=False
)
nb_fn_argmax_probs_uw = scatter_reduce(
	nb_fn_argmax_probs.float(),
	f_batch_idxses[0],
	reduce="mean",
	dim_size=num_formulae,
	include_self=False
)
argmax_d["fn_argmax_probs_fw"] = fn_argmax_probs_fw
argmax_d["fn_argmax_probs_uw"] = fn_argmax_probs_uw
argmax_d["nb_fn_argmax_probs_fw"] = nb_fn_argmax_probs_fw
argmax_d["nb_fn_argmax_probs_uw"] = nb_fn_argmax_probs_uw

# deal with consensus
fn_argmax_cons_fw = scatter_reduce(
	fn_argmax_cons.float() * th.exp(f_logprobs_mean),
	f_batch_idxses[0],
	reduce="sum",
	dim_size=num_formulae
)
nb_fn_argmax_cons_fw = scatter_reduce(
	nb_fn_argmax_cons.float() * th.exp(f_logprobs_mean),
	f_batch_idxses[0],
	reduce="sum",
	dim_size=num_formulae
)
fn_argmax_cons_uw = scatter_reduce(
	fn_argmax_cons.float(),
	f_batch_idxses[0],
	reduce="mean",
	dim_size=num_formulae,
	include_self=False
)
nb_fn_argmax_cons_uw = scatter_reduce(
	nb_fn_argmax_cons.float(),
	f_batch_idxses[0],
	reduce="mean",
	dim_size=num_formulae,
	include_self=False
)
argmax_d["fn_argmax_cons_fw"] = fn_argmax_cons_fw
argmax_d["fn_argmax_cons_uw"] = fn_argmax_cons_uw
argmax_d["nb_fn_argmax_cons_fw"] = nb_fn_argmax_cons_fw
argmax_d["nb_fn_argmax_cons_uw"] = nb_fn_argmax_cons_uw

argmax_df = []
for k, v in argmax_d.items():
	v_mean = th.mean(v).item()
	argmax_df.append({
		"metric": k,
		"mean": v_mean,
	})
argmax_df = pd.DataFrame(argmax_df)
print(argmax_df)
argmax_df.to_csv(f"{mode}_argmax_df.csv",index=False)	

### H(n|f) and H(n'|f)
print("> Entropies")

# get H(n|f) and H(n'|f) per datapoint per model
entropies = [
	"formula_e",
	"node_e",
	"node_formula_e",
	"formula_node_e",
	"nb_node_e",
	"nb_node_formula_e",
	"nb_formula_node_e",
	"nb_node_node_e",
	"formula_ne",
	"node_ne",
	"node_formula_ne",
	"formula_node_ne",
	"nb_node_ne",
	"nb_node_formula_ne",
	"nb_formula_node_ne",
	"nb_node_node_ne"
]
entropy_df = []
for entropy in entropies:
	all_entropy = th.stack([vals[entropy] for vals in model_to_vals.values()], dim=0)
	ensemble_metrics[entropy+"s"] = all_entropy
	entropy_mean = th.mean(all_entropy, dim=0)
	entropy_std = th.std(all_entropy, dim=0)
	entropy_cv = entropy_std / th.clamp(entropy_mean, min=EPS)
	if th.mean(entropy_cv) > 100:
		import pdb; pdb.set_trace()
	entropy_min = th.min(all_entropy, dim=0).values
	entropy_max = th.max(all_entropy, dim=0).values
	entropy_range = entropy_max - entropy_min
	entropy_expmax = th.exp(th.max(all_entropy, dim=0).values)
	if th.mean(entropy_expmax) > 100:
		import pdb; pdb.set_trace()
	entropy_expmin = th.exp(th.min(all_entropy, dim=0).values)
	entropy_exprange = entropy_expmax - entropy_expmin
	entropy_exprange2 = th.exp(entropy_range)
	ensemble_metrics[entropy+"_mean"] = entropy_mean
	ensemble_metrics[entropy+"_std"] = entropy_std
	ensemble_metrics[entropy+"_cv"] = entropy_cv
	entropy_df.append({
		"entropy": entropy,
		"mean": th.mean(entropy_mean).item(),
		"std": th.mean(entropy_std).item(),
		"cv": th.mean(entropy_cv).item(),
		"min": th.mean(entropy_min).item(),
		"max": th.mean(entropy_max).item(),
		"range": th.mean(entropy_range).item(),
		"exp_min": th.mean(entropy_expmin).item(),
		"exp_max": th.mean(entropy_expmax).item(),
		"exp_range": th.mean(entropy_exprange).item(),
		"exp_range_2": th.mean(entropy_exprange2).item()
	})
entropy_df = pd.DataFrame(entropy_df)
print(entropy_df)
entropy_df.to_csv(f"{mode}_entropy_df.csv",index=False)

import pdb; pdb.set_trace()

### JSS(n|f) and JSS(n'|f)
print("> Pairwise JSS(n|f) and JSS(n'|f)")

# get P(n|f), P(f), P(n'|f)
select_keys = [
	"pred_formula_node_logprobs",
	"pred_joint_formula_idxs",
	"pred_joint_node_idxs",
	"pred_nb_formula_node_logprobs",
	"pred_nb_joint_formula_idxs",
	"pred_nb_joint_node_idxs",
	"pred_formula_logprobs",
	"pred_formula_formula_idxs",
	"pred_formula_batch_idxs"
]
select_d = select_model_vals(model_to_vals, select_keys)
formula_node_logprobses = select_d["pred_formula_node_logprobs"]
formula_node_formula_idxses = select_d["pred_joint_formula_idxs"]
formula_node_node_idxses = select_d["pred_joint_node_idxs"]
nb_formula_node_logprobses = select_d["pred_nb_formula_node_logprobs"]
nb_formula_node_formula_idxses = select_d["pred_nb_joint_formula_idxs"]
nb_formula_node_node_idxses = select_d["pred_nb_joint_node_idxs"]
formula_logprobses = select_d["pred_formula_logprobs"]
formula_formula_idxses = select_d["pred_formula_formula_idxs"]
formula_batch_idxes = select_d["pred_formula_batch_idxs"]
del select_d

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
pairwise_jss = []
pairwise_ids = []
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
		jsd_ij = 0.5 * (kl_ij + kl_ji)
		jsd_b_ij = scatter_reduce(
			jsd_ij * th.exp(m_formula_logprobs_ij),
			formula_batch_idxes[i],
			reduce="sum",
			dim_size=formula_batch_idxes[i].max()+1
		)
		pairwise_jss_ij = 1. - jsd_b_ij
		pairwise_jss.append(pairwise_jss_ij)
		pairwise_ids.append(th.tensor((i,j),dtype=th.long))
pairwise_jss = th.stack(pairwise_jss, dim=0)
pairwise_ids = th.stack(pairwise_ids, dim=0)
pairwise_jss_mean = th.mean(pairwise_jss,dim=0)
pairwise_jss_std = th.std(pairwise_jss,dim=0)
ensemble_metrics["pairwise_jss"] = pairwise_jss
ensemble_metrics["pairwise_ids"] = pairwise_ids
ensemble_metrics["pairwise_jss_mean"] = pairwise_jss_mean
ensemble_metrics["pairwise_jss_std"] = pairwise_jss_std
ensemble_metrics["pairwise_jss_cv"] = pairwise_jss_std / th.clamp(pairwise_jss_mean, min=EPS)

### Correlation between spec_sims and the other metrics
print("> Correlation between sims and other metrics")

def pearson_r(x,y):
	x_mean = th.mean(x,dim=0)
	y_mean = th.mean(y,dim=0)
	x_std = th.std(x,dim=0)
	y_std = th.std(y,dim=0)
	r = th.mean((x-x_mean)*(y-y_mean),dim=0) / (x_std * y_std)
	return r

# # per-model correlations
# r_agrees, r_nb_agrees = [], []
# for model_idx, model in enumerate(model_to_vals.keys()):
# 	spec_sims = ensemble_metrics["spec_sims"][model_idx]
# 	argmax_agree = ensemble_metrics["formula_node_argmax_agree"][model_idx]
# 	nb_argmax_agree = ensemble_metrics["nb_formula_node_argmax_agree"][model_idx]
# 	r_agree = pearson_r(spec_sims, argmax_agree)
# 	r_nb_agree = pearson_r(spec_sims, nb_argmax_agree)
# 	r_agrees.append(r_agree)
# 	r_nb_agrees.append(r_nb_agree)
# r_agrees = th.stack(r_agrees, dim=0)
# r_nb_agrees = th.stack(r_nb_agrees, dim=0)
# print(th.mean(r_agrees))
# print(th.mean(r_nb_agrees))

# ensemble correlations
ensemble_r_metrics = []
for agg in ["mean","std","cv"]:
	ensemble_r_metrics.extend([entropy+f"_{agg}" for entropy in entropies])
	ensemble_r_metrics.extend([f"pairwise_jss_{agg}"])
	ensemble_r_metrics.extend([f"formula_node_argmax_agree_{agg}"])
	ensemble_r_metrics.extend([f"nb_formula_node_argmax_agree_{agg}"])
ensemble_r_metrics.extend(["formula_node_argmax_counts","nb_formula_node_argmax_counts"])
ensemble_r_df = []
for ensemble_r_metric in ensemble_r_metrics:
	ensemble_mean = th.mean(ensemble_metrics[ensemble_r_metric],dim=0)
	ensemble_std = th.std(ensemble_metrics[ensemble_r_metric],dim=0)
	ensemble_cv = ensemble_std / th.clamp(ensemble_mean, min=EPS)
	ensemble_r_simmean = pearson_r(ensemble_metrics["spec_sims_mean"], ensemble_metrics[ensemble_r_metric])
	ensemble_r_simstd = pearson_r(ensemble_metrics["spec_sims_std"], ensemble_metrics[ensemble_r_metric])
	ensemble_r_simcv = pearson_r(ensemble_metrics["spec_sims_cv"], ensemble_metrics[ensemble_r_metric])
	ensemble_r_df.append({
		"metric": ensemble_r_metric,
		"mean": ensemble_mean.item(),
		"std": ensemble_std.item(),
		"cv": ensemble_cv.item(),
		"r_sim_mean": ensemble_r_simmean.item(),
		"r_sim_std": ensemble_r_simstd.item(),
		"r_sim_cv": ensemble_r_simcv.item()
	})
ensemble_r_df = pd.DataFrame(ensemble_r_df)
print(ensemble_r_df)
ensemble_r_df.to_csv(f"{mode}_ensemble_r_df.csv",index=False)

# TODO: per-formula ensemble correlations
# get P(f) for each formula
# get each metric by formula (where applicable) for each model
# measure correlation between mean/std/cv P(f) and mean/std/cv metric
# also ger formula masses and measure correlation between mean/std masses and mean/std metric

# is it something about the scales of the metrics??

### formula probabilities P(F) and conditional entropies H(N|F=f)
print("> Formula shit")
select_keys = [
	"pred_formula_logprobs",
	"pred_formula_formula_idxs",
	"pred_formula_batch_idxs",
	"pred_formula_node_logprobs",
	"pred_joint_formula_idxs",
	"pred_joint_node_idxs",
	"pred_joint_batch_idxs"
]
select_d = select_model_vals(model_to_vals, select_keys)
formula_logprobses = select_d["pred_formula_logprobs"]
formula_formula_idxses = select_d["pred_formula_formula_idxs"]
formula_batch_idxses = select_d["pred_formula_batch_idxs"]
formula_node_logprobses = select_d["pred_formula_node_logprobs"]
formula_node_formula_idxses = select_d["pred_joint_formula_idxs"]
formula_node_node_idxses = select_d["pred_joint_node_idxs"]
formula_node_batch_idxses = select_d["pred_joint_batch_idxs"]
del select_d

# calculate H(N|F=f)
formula_node_ces = []
for i in range(num_models):
	formula_node_logprobs = formula_node_logprobses[i]
	formula_node_formula_idxs = formula_node_formula_idxses[i]
	formula_formula_idxs = formula_formula_idxses[i]
	formula_node_ce = -scatter_reduce(
		th.exp(formula_node_logprobs) * formula_node_logprobs,
		formula_node_formula_idxs,
		reduce="sum",
		dim_size=formula_formula_idxs.max()+1
	)
	formula_node_ces.append(formula_node_ce)
formula_node_ces = th.stack(formula_node_ces, dim=0)

# calculate correlation
formula_node_logprobs_mean = th.exp(log_mean(formula_node_logprobses,num_models,dim=0))
formula_node_ces_mean = th.mean(formula_node_ces,dim=0)
formula_node_logprobs_std = th.exp(log_std(formula_node_logprobses,num_models,dim=0))
formula_node_ces_std = th.std(formula_node_ces,dim=0)
formula_node_logprobs_cv = formula_node_logprobs_std / th.clamp(formula_node_logprobs_mean, min=EPS)
formula_node_ces_cv = formula_node_ces_std / th.clamp(formula_node_ces_mean, min=EPS)

formula_logprobs_mean = th.exp(log_mean(formula_logprobses,num_models,dim=0))
formula_logprobs_std = th.exp(log_std(formula_logprobses,num_models,dim=0))
formula_logprobs_cv = formula_logprobs_std / th.clamp(formula_logprobs_mean, min=EPS)

for agg in ["mean","std","cv"]:
	key1 = "formula_node_ce_"+agg
	key2 = "formula_logprobs_"+agg
	if agg == "mean":
		val1 = formula_node_ces_mean
		val2 = formula_logprobs_mean
	elif agg == "std":
		val1 = formula_node_ces_std
		val2 = formula_logprobs_std
	elif agg == "cv":
		val1 = formula_node_ces_cv
		val2 = formula_logprobs_cv
	ensemble_metrics[key1] = val1
	ensemble_metrics[key2] = val2

ensemble_r_metrics = []
for agg in ["mean","std","cv"]:
	ensemble_r_metrics.extend([f"formula_node_ce_{agg}"])

ensemble_r_df = []
for ensemble_r_metric in ensemble_r_metrics:
	ensemble_mean = th.mean(ensemble_metrics[ensemble_r_metric],dim=0)
	ensemble_std = th.std(ensemble_metrics[ensemble_r_metric],dim=0)
	ensemble_cv = ensemble_std / th.clamp(ensemble_mean, min=EPS)
	ensemble_r_pfmean = pearson_r(ensemble_metrics["formula_logprobs_mean"], ensemble_metrics[ensemble_r_metric])
	ensemble_r_pfstd = pearson_r(ensemble_metrics["formula_logprobs_std"], ensemble_metrics[ensemble_r_metric])
	ensemble_r_pfcv = pearson_r(ensemble_metrics["formula_logprobs_cv"], ensemble_metrics[ensemble_r_metric])
	ensemble_r_df.append({
		"metric": ensemble_r_metric,
		"mean": ensemble_mean.item(),
		"std": ensemble_std.item(),
		"cv": ensemble_cv.item(),
		"r_pf_mean": ensemble_r_pfmean.item(),
		"r_pf_std": ensemble_r_pfstd.item(),
		"r_pf_cv": ensemble_r_pfcv.item()
	})
ensemble_r_df = pd.DataFrame(ensemble_r_df)
print(ensemble_r_df)
ensemble_r_df.to_csv(f"{mode}_ensemble_r_df_2.csv",index=False)
import pdb; pdb.set_trace()


