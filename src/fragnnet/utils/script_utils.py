import multiprocessing
import pandas as pd
import torch as th
import numpy as np
from tqdm import tqdm
import tempfile
import os

from fragnnet.runner import init_dataloader
from fragnnet.utils.misc_utils import to_device, scatter_reduce, flatten_lol, LOG_ZERO
from fragnnet.pl_model import FragGNNPL
from fragnnet.iceberg.pl_model import IcebergIntenPL
from fragnnet.utils.nn_utils import decompile_jit_ckpt
from fragnnet.pl_model import FragGNNPL, NeimsPL, GNNPL
from fragnnet.massformer.pl_model import MassFormerPL
from fragnnet.iceberg.pl_model import IcebergGenPL, IcebergIntenPL
from fragnnet.iceberg.dataset import SpecMolMagmaIntenDataset,SpecMolMagmaGenDataset
from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.misc_utils import to_device
from fragnnet.utils.nn_utils import decompile_jit_ckpt
from fragnnet.dataset import SpecMolFragDataset, SpecMolDataset
from fragnnet.utils.pl_utils import get_pl_hparams
from fragnnet.utils.setup_utils import get_model_cls, get_dataset_cls

def init_wandb_ckpt(run_id,last_ckpt,model_cls,config_d):
	import wandb
	# load model
	with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dp:
		api = wandb.Api()
		run = api.run(f"frag-gnn/frag-gnn/{run_id}")
		if last_ckpt:
			ckpt_wandb_fp = "ckpt/last.ckpt"
		else:
			files = run.files(per_page=100)
			best_ckpt_wandb_fp, best_epoch = None, None
			for file in files:
				if file.name.endswith(".ckpt") and "last" not in file.name:
					ckpt_wandb_fp = file.name
					epoch = int(os.path.basename(file.name).removeprefix("model-epoch=").removesuffix(".ckpt"))
					if best_ckpt_wandb_fp is None or epoch > best_epoch:
						best_ckpt_wandb_fp = ckpt_wandb_fp
						best_epoch = epoch
		assert ckpt_wandb_fp is not None
		ckpt_file_local = run.file(ckpt_wandb_fp).download(root=temp_dp,replace=False)
		ckpt_fp = ckpt_file_local.name
		ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)
		# decompile
		assert not config_d["compile"]
		ckpt = decompile_jit_ckpt(ckpt)
		# init
		model = model_cls(**config_d)
		try:
			model.load_state_dict(ckpt["state_dict"], strict=True)
		except RuntimeError as e:
			print(f"> error when loading from checkpoint id {run_id}, will try with strict=False")
			model.load_state_dict(ckpt["state_dict"], strict=False)
	return model


def run_inference(dl, model, device, eval_split, batch_cutoff, nb_iso, output_subset=None, untransform_spec=False):
	
	# make predictions on validation data
	print(">>> Iterating over data")
	cum_num_datapoints = 0
	cum_num_nodes = 0
	cum_num_formulas = 0
	cum_num_nb_nodes = 0
	fraggnn_model = isinstance(model, FragGNNPL)
	iceberg_model = isinstance(model, IcebergIntenPL)
	if not (fraggnn_model or iceberg_model):
		assert not nb_iso
		BATCH_IDX_VARS = [
			"pred_batch_idxs",
			"true_batch_idxs"
		]
		NODE_IDX_VARS = []
		FORMULA_IDX_VARS = []
		NB_NODE_IDX_VARS = []
	elif iceberg_model:
		assert not nb_iso
		BATCH_IDX_VARS = [
			"pred_batch_idxs",
			"true_batch_idxs",
		]
		if model.model.output_formula_str:
			BATCH_IDX_VARS.extend([
				"pred_formula_batch_idxs"
			])
		NODE_IDX_VARS = []
		FORMULA_IDX_VARS = []
		NB_NODE_IDX_VARS = []
	else:
		BATCH_IDX_VARS = [
			"pred_batch_idxs",
			"true_batch_idxs",
			"pred_formula_batch_idxs",
			"pred_node_batch_idxs",
			"pred_joint_batch_idxs"
		]
		NODE_IDX_VARS = [
			"pred_node_node_idxs",
			"pred_joint_node_idxs"
		]
		FORMULA_IDX_VARS = [
			"pred_formula_formula_idxs",
			"pred_joint_formula_idxs"
		]
		NB_NODE_IDX_VARS = []
		if nb_iso:
			BATCH_IDX_VARS.extend([
				"pred_nb_node_batch_idxs",
				"pred_nb_joint_batch_idxs",
				"pred_nb_node_node_batch_idxs"
			])
			FORMULA_IDX_VARS.extend([
				"pred_nb_joint_formula_idxs"
			])
			NB_NODE_IDX_VARS.extend([
				"pred_nb_node_node_idxs",
				"pred_nb_joint_node_idxs",
				"pred_nb_node_node_node_idxs"
			])
	if output_subset is not None:
		BATCH_IDX_VARS = [k for k in BATCH_IDX_VARS if k in output_subset]
		NODE_IDX_VARS = [k for k in NODE_IDX_VARS if k in output_subset]
		FORMULA_IDX_VARS = [k for k in FORMULA_IDX_VARS if k in output_subset]
		NB_NODE_IDX_VARS = [k for k in NB_NODE_IDX_VARS if k in output_subset]
	vals = {}
	if model.training:
		print(f"> Warning: model {model} is in training mode!")
	with th.inference_mode(): 
		for b_idx, b_input in tqdm(enumerate(dl),total=min(len(dl),batch_cutoff)):
			# transfer data, make predictions
			b_input = to_device(b_input, device,non_blocking=False)
			b_output = model.inference_with_ground_truth_step(b_input, split=eval_split, untransform_spec=untransform_spec)
			b_output = to_device(b_output, "cpu",non_blocking=False)
			if output_subset is not None:
				b_output = {k: b_output[k] for k in b_output if k in output_subset}
			output_keys = b_output.keys()
			# rebatch stuff
			if "pred_batch_idxs" in output_keys:
				b_num_datapoints = b_output["pred_batch_idxs"].max()+1
			elif "true_batch_idxs" in output_keys:
				b_num_datapoints = b_output["true_batch_idxs"].max()+1
			else:
				b_num_datapoints = 0
			if fraggnn_model and "pred_node_node_idxs" in output_keys:
				b_num_nodes = b_output["pred_node_node_idxs"].max()+1
			else:
				b_num_nodes = 0
			if fraggnn_model and "pred_formula_formula_idxs" in output_keys:
				b_num_formulas = b_output["pred_formula_formula_idxs"].max()+1
			else:
				b_num_formulas = 0
			if fraggnn_model and nb_iso and "pred_nb_node_node_idxs" in output_keys:
				b_num_nb_nodes = b_output["pred_nb_node_node_idxs"].max()+1
			else:
				b_num_nb_nodes = 0
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
		elif isinstance(vals[k][0], th.Tensor):
			if vals[k][0].dim() == 0:
				vals[k] = th.stack(vals[k], dim=0)
			else:
				vals[k] = th.cat(vals[k], dim=0)
		elif isinstance(vals[k][0], np.ndarray):
			if vals[k][0].ndim == 0:
				vals[k] = np.stack(vals[k], axis=0)
			else:
				vals[k] = np.concatenate(vals[k], axis=0)
		else:
			raise ValueError(f"unexpected type {type(vals[k][0])}")
	return vals

def select_model_vals(model_to_vals, keys, stack_dim=0):

	select_d = {k: [] for k in keys}
	for model, vals in model_to_vals.items():
		for k in keys:
			select_d[k].append(vals[k])
	select_d = {k: th.stack(v, dim=stack_dim) for k,v in select_d.items()}
	return select_d

def log_mean(log_p,dim=0):
	return th.logsumexp(log_p,dim=dim) - np.log(log_p.shape[dim])

def log_std(log_p,dim=0,correction=1):
	support_size = log_p.shape[dim]
	log_mean_p = log_mean(log_p,dim=dim)
	log_mean_p_sq = log_mean(2*log_p,dim=dim)
	log_var = th.log(th.exp(log_mean_p_sq) - th.exp(2*log_mean_p))
	return 0.5*(log_var + np.log(support_size) - np.log(support_size - correction))

# define sparse kl divergence
def sparse_kl(log_p,log_q,b_idx,dim_size):
	p = th.exp(log_p)
	kl = scatter_reduce(
		p * (log_p - th.clamp(log_q,min=LOG_ZERO(log_p.dtype))),
		b_idx,
		reduce="sum",
		dim_size=dim_size
	)
	kl = th.clamp(kl,min=0.0)
	return kl

def pearson_r(x,y,dim=0):
	x_mean = th.mean(x,dim=dim)
	y_mean = th.mean(y,dim=dim)
	x_std = th.std(x,dim=dim)
	y_std = th.std(y,dim=dim)
	r = th.mean((x-x_mean)*(y-y_mean),dim=dim) / (x_std * y_std)
	return r

def load_model_and_init_config(
		ckpt_fp:str, 
		device:str|None = None, 
		batch_size:int = 32, 
		auxiliary_scores:list|None = None, 
		eval_mz_bin_res:list[float]|None = None, 
		custom_fp:str|None = None,
		template_fp:str|None = None,
		num_workers:int|None = None):
	"""load model and reset config for inference

	Args:
		ckpt_fp (str): check point path
		device (str, optional): device eg cpu or cuda:0. Defaults to None.
		batch_size (int, optional): batch_size. Defaults to 32.
		auxiliary_scores (str, optional): list of auxiliary_scores. Defaults to None, only used on split evalution ['cos_sim']
		eval_mz_bin_res (str, optional): bin size of auxiliary scores. Defaults to None, only used on split evalution, eg [03.01]

	Returns:
		_type_: _description_
	"""
	
	# load ckpt
	ckpt = th.load(ckpt_fp, map_location=device, weights_only=False)
	
	print(">> Loading configs")
	config_d = get_pl_hparams(ckpt)
	if custom_fp is not None:
		assert template_fp is not None
		config_d_2 = load_config(template_fp, custom_fp)
		for k, v in config_d_2.items():
			if k in config_d and v != config_d[k]:
				print(f"> overwrite config: {k} -- {config_d[k]} vs {v}")
			elif k not in config_d:
				print(f"> add config: {k} -- {v}")
			config_d[k] = v

	# overwrite config values
	# set eval batch size
	config_d['eval_batch_size'] = batch_size
	# disable metrics
	config_d['track_datapoint_metrics'] = False 
	config_d['auxiliary_scores'] = auxiliary_scores if auxiliary_scores is not None else []
	config_d['eval_mz_bin_res'] = eval_mz_bin_res if eval_mz_bin_res is not None else []
	# let us see if ram will be an issue
	config_d['frag_params']['preload'] = False
	config_d['frag_params']['preprocess'] = False
	config_d['magma_params']['preprocess'] = False
	# disable sampler
	config_d["dynamic_batch_sampler"] = False
	config_d["group_sampler"] = False 
	config_d["simple_group_sampler"] = False
	# additional spec_params
	config_d['spec_params']['unique_id'] = True
	config_d['spec_params']["counts"] = True
	# set num works and cpu stuff
	if num_workers is not None:
		config_d['num_workers'] = num_workers

    # config cuda and cpus
	if device is None:
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
	elif not th.cuda.is_available() and device != "cpu":
		device = "cpu"
	print(f"> using {device}")
	# cpu specifc setup
	if device == "cpu":
		th.set_num_threads(multiprocessing.cpu_count())
		th.multiprocessing.set_sharing_strategy('file_system')
		th.use_deterministic_algorithms(True)
  
	# check class
	print(f">> model type: {config_d['model_type']}")
	model_type = config_d['model_type']
	model_cls = get_model_cls(model_type)

 	# load models
	print(">> Loading models")
	print(f"> ckpt was compiled: {config_d['compile']}")
	if config_d['compile']:
		ckpt = decompile_jit_ckpt(ckpt)
		config_d['compile'] = False

	model = model_cls(**config_d)
	model.load_state_dict(ckpt["state_dict"])
	model.to(device)
	return model, config_d, device

def run_spectra_prediction(model,
				 config_d:dict,
				 mol_data_ptr:str|pd.DataFrame,
				 spec_data_ptr:str|pd.DataFrame,
				 device:str,
				 frag_dp:str|None= None,
				 magma_dp:str|None = None,
				 validate:bool = False,
				 ):
	"""_summary_

	Args:
		model (_type_): _description_
		config_d (dict): _description_
		mol_data_ptr (str | pd.DataFrame): _description_
		spec_data_ptr (str | pd.DataFrame): _description_
		frag_dp (str): _description_
		device (str): _description_

	Returns:
		_type_: _description_
	"""
	local_config_d = copy.deepcopy(config_d)
	# config proc and check if frags are avaibale
	if isinstance(mol_data_ptr, str):
		print(f"> mol_fp {mol_data_ptr}")
		mol_df = pd.read_pickle(mol_data_ptr)
	elif isinstance(mol_data_ptr, pd.DataFrame):
		mol_df = mol_data_ptr
		print(f"> mol_df {mol_df.shape}")
	else:
		raise ValueError("mol_data_ptr should be str or pd.DataFrame")

	if isinstance(spec_data_ptr, str):
		print(f"> spec_fp {spec_data_ptr}")
		spec_df = pd.read_pickle(spec_data_ptr)
	elif isinstance(spec_data_ptr, pd.DataFrame):
		spec_df = spec_data_ptr
		print(f"> spec_df {spec_df.shape}")
	else:
		raise ValueError("mol_data_ptr should be str or pd.DataFrame")


	# check model class and ds class
	model_ds_cls = None
 
	# only check for FragGNNet
	if isinstance(model,FragGNNPL):
		assert os.path.isdir(frag_dp)
		print(f">> frag_dp {frag_dp}")
		print(">> filtering input without frags")
		all_frags = set([f.split('.')[0] for f in os.listdir(frag_dp)])
		print(">> num of frags: ", len(all_frags))
		mol_df = mol_df[mol_df['mol_id'].astype(str).isin(all_frags)]
		local_config_d['frag_dp'] = frag_dp
		model_ds_cls = SpecMolFragDataset
		print("> num of spec before filtering by dag", len(spec_df))
		spec_df = spec_df[spec_df['mol_id'].isin(mol_df['mol_id'].to_list())]
		print("> num of spec after filtering by dag", len(spec_df))
	elif isinstance(model,NeimsPL) or isinstance(model,MassFormerPL) or isinstance(model, GNNPL):
		mz_max = local_config_d['mz_max']
		spec_df = spec_df[spec_df['prec_mz'] <= mz_max]
		#print("> [NeimsPL] num of spec after prec_mz filtering", len(spec_df))
		model_ds_cls = SpecMolDataset
	elif isinstance(model,IcebergIntenPL):
		assert os.path.isdir(magma_dp)
		model_ds_cls = SpecMolMagmaIntenDataset
		print(f">> filtering input without magma. magma_dp: {magma_dp}")
		magma_jsons = set([f.split('.')[0] for f in os.listdir(os.path.join(magma_dp,'magma_tree')) if f.endswith('.json')])
		print(">> num of magma jsons: ", len(magma_jsons))
		print("> num of spec before filtering by magma", len(spec_df))
		spec_df = spec_df[spec_df['group_id'].astype(str).isin(magma_jsons)]
		print("> num of spec after filtering by magma", len(spec_df))		
		local_config_d['magma_dp'] = magma_dp
	else:
		print(f"Not supported model: {model}")
  

	if len(spec_df) == 0:
		columns = ['spec_id','mol_id','pred_mzs','pred_ints','pred_oos']
		return pd.DataFrame([], columns=columns)
	
	# config mol df, spec df and frag_dp
	local_config_d['mol_fp'] = mol_df
	local_config_d['spec_fp'] = spec_df

	# load datasets
	print(">> Loading datasets")
	#if validate:
	#	ds = model_ds_cls(split='val', **config_d)
	#else:
	ds = model_ds_cls(split='predict_only', **local_config_d)
	print(">> init dataloader")
	dl = init_dataloader(ds, local_config_d)
	
	eval_score_types = []
	if validate:
		for score_type in local_config_d['auxiliary_scores']:
			if "_hun" in score_type:
				eval_score_types.append(score_type)
			else:
				for bin_res in local_config_d['eval_mz_bin_res']:
					eval_score_types.append(f"{score_type}_{bin_res}")
	preds_l = []
 	#print(">> set model to this to eval")
	model.eval()
	with th.inference_mode():
		for _, batch_input in tqdm(enumerate(dl),total=len(dl), desc=" Running inference"):
			spec_ids = batch_input['spec_unique_id']
			mol_ids = batch_input['mol_id']

			batch_input = to_device(batch_input, device)
			# use existing validate to compute cos sim and etc
			if validate:
				batch_result = model._common_step(batch_input, split="test", log=False)
			else:
				batch_result = model.predict_step(**batch_input)
    
			batch_size = batch_input['batch_size'].detach().cpu()

			pred_mzs = batch_result["pred_mzs"].detach().cpu()
			pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()
			
			has_oos_b = "pred_oos_logprobs" in batch_result
			if has_oos_b:
				pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()
	
			pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
			pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)

			eval_scores_d = {}
			for eval_score_type in eval_score_types:
				eval_scores_d[eval_score_type] = batch_result[eval_score_type].detach().cpu()
				#print(eval_score_type, eval_scores_d[eval_score_type])
			for b_idx in range(batch_size):
				b_peak_mask = pred_batch_idxs == b_idx
	
				b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy().item() if has_oos_b else np.nan
				b_spec_id = spec_ids[b_idx]
				b_mol_id = mol_ids[b_idx]

				b_spec_mzs = pred_mzs[b_peak_mask]
				b_spec_ints = pred_ints[b_peak_mask]

				data_row = [b_spec_id.item(), b_mol_id, b_spec_mzs.tolist(),b_spec_ints.tolist(), b_pred_oos_logprob]
				for eval_score_type in eval_score_types:
					data_row.append(eval_scores_d[eval_score_type][b_idx].item())
				preds_l.append(data_row)

	columns = ['spec_id','mol_id','pred_mzs','pred_ints','pred_oos'] + eval_score_types
	df = pd.DataFrame(preds_l, columns=columns)
	return df