import torch as th
from torch.utils.data import DataLoader
import torch_geometric as pyg

try:
	import lightning.pytorch as pl	
	from lightning.pytorch.callbacks import DeviceStatsMonitor
	from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
	from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
except ModuleNotFoundError:
	import pytorch_lightning as pl
	from pytorch_lightning.callbacks import DeviceStatsMonitor
	from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
	from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler

import logging
import argparse
import yaml
import os
import numpy as np
import sys
from pprint import pprint
import wandb
import glob
import shutil


from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.iceberg.pl_model import IcebergGenPL, IcebergIntenPL
from frag_gnn.utils.nn_utils import nan_forward_hook, nan_backward_hook
from frag_gnn.utils.frag_utils import NUM_ELEMENTS
from frag_gnn.utils.pl_utils import ConsoleLogger, PrintGradCallback
from frag_gnn.dataset import SpecMolDataset, SpecMolFragDataset, SpecMolFragDynamicBatchSampler
from frag_gnn.iceberg.dataset import SpecMolMagmaGenDataset, SpecMolMagmaIntenDataset
from frag_gnn.utils.misc_utils import deep_update


def load_config(template_fp, custom_fp):

	assert os.path.isfile(template_fp), template_fp
	if custom_fp:
		assert os.path.isfile(custom_fp), custom_fp
	with open(template_fp, "r") as template_file:
		config_d = yaml.load(template_file, Loader=yaml.FullLoader)
	# overwrite parts of the config
	if custom_fp:
		with open(custom_fp, "r") as custom_file:
			custom_d = yaml.load(custom_file, Loader=yaml.FullLoader)
		assert all([k in config_d for k in custom_d]), set(custom_d.keys()) - set(config_d.keys())
		config_d = deep_update(config_d, custom_d)
	return config_d


def load_wandb_config(wandb_config_dp) -> dict:
	"""Load wandb config file if exists, else return empty dict

	Args:
		wandb_config_dp (_type_): _description_

	Returns:
		dict: wandb config
	"""
	wandb_config_dp = os.path.join(wandb_config_dp,"config.yaml")

	if os.path.isfile(wandb_config_dp):
		with open(wandb_config_dp,"r") as wandb_config_file:	
			_config_d = yaml.load(
				wandb_config_file,
				Loader=yaml.FullLoader
			)
		del _config_d["wandb_version"]
		del _config_d["_wandb"]
		config_d = {}
		for k in _config_d.keys():
			config_d[k] = _config_d[k]["value"]
	else:
		config_d = {}
	return config_d


def init_run(template_fp, custom_fp, wandb_mode, job_id):

	# setup logger
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[logging.StreamHandler()]
	)

	# load config
	config_d = load_config(template_fp, custom_fp)

	# set random seeds
	th.manual_seed(config_d["seed"])
	np.random.seed(config_d["seed"])

	# set torch multiprocessing strategy
	th.multiprocessing.set_sharing_strategy(config_d["mp_sharing_strategy"])

	# setup logging and wandb
	logging.info("setup loggers")
	loggers = []
	console_logger = ConsoleLogger()
	loggers.append(console_logger)

	if wandb_mode != "disabled":
		wandb_d = dict(
			project=config_d["wandb_project"],
			name=config_d["wandb_name"],
			mode=wandb_mode,
			entity=config_d["wandb_entity"],
			tags=config_d["wandb_tags"],
			resume="allow",
		)

		if job_id:
			job_id_fp = os.path.join("job_id",f"{job_id}.id")
			if os.path.isfile(job_id_fp):
				is_resume = True
				with open(job_id_fp,"r") as job_id_file:
					text = job_id_file.read().strip()
				run_id, old_wandb_dp = text.split(";")
				wandb_config = load_wandb_config(old_wandb_dp)
				
				# for offline mode, if there is no configure file, just skip it
				# wandb config only exists after wandb sync
				if len(wandb_config) == 0:
					wandb_config = config_d
			else:
				is_resume = False
				run_id = old_wandb_dp = None
				wandb_config = config_d
			wandb_d["id"] = run_id
			wandb_d["config"] = wandb_config
		else:
			is_resume = False
			job_id_fp = None
			wandb_d["config"] = config_d
		wandb.init(**wandb_d)
		assert wandb.run is not None, wandb.run
		wandb_d["offline"] = wandb_mode == "offline"
		wandb_logger = pl.loggers.WandbLogger(**wandb_d)
		if job_id_fp:
			os.makedirs("job_id", exist_ok=True)
			with open(job_id_fp,"w+") as job_id_file:
				text = f"{wandb.run.id};{os.path.abspath(wandb.run.dir)}"
				job_id_file.write(text)
		# update config results (primarily for wandb)
		for k in list(config_d.keys()):
			if k in wandb.config:
				if config_d[k] != wandb.config[k]:
					print(f"> config diff -- {k}: {config_d[k]} vs {wandb.config[k]}")
				config_d[k] = wandb.config[k]
		loggers.append(wandb_logger)
	else:
		is_resume = False

	# CUDA and TF32 setup

	if config_d['use_tensor_float32'] == True and config_d['accelerator'] == 'gpu':
		# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
		th.backends.cuda.matmul.allow_tf32 = True
		# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
		th.backends.cudnn.allow_tf32 = True

	# setup model
	logging.info("setup model")
	if config_d["model_type"] == "frag_gnn":
		model_cls = FragGNNPL
	elif config_d["model_type"] == "neims":
		model_cls = NeimsPL
	elif config_d["model_type"] == "iceberg_gen":
		model_cls = IcebergGenPL
	elif config_d["model_type"] == "iceberg_inten":
		model_cls = IcebergIntenPL
	else:
		raise ValueError(config_d["model_type"])
	model = model_cls(**config_d)
	model.train()

	# setup callbacks
	callbacks = []
	if wandb_mode != "disabled":
		ckpt_dp = os.path.join(wandb.run.dir,"ckpt")
	else:
		ckpt_dp = "tmp_ckpt"
	os.makedirs(ckpt_dp, exist_ok=True)
	if is_resume:
		# copy the checkpoint files
		old_ckpt_fps = glob.glob(os.path.join(old_wandb_dp,"ckpt","*.ckpt"))
		for old_ckpt_fp in old_ckpt_fps:
			new_ckpt_fp = os.path.join(ckpt_dp,os.path.basename(old_ckpt_fp))
			# modify checkpoint metadata (hacky)
			new_ckpt_data = th.load(old_ckpt_fp)
			ckpt_callback_data = None
			for k,v in new_ckpt_data["callbacks"].items():
				if k.startswith("ModelCheckpoint"):
					ckpt_callback_data = v
					break
			assert ckpt_callback_data is not None
			ckpt_keys = ["best_model_path","last_model_path","dirpath","kth_best_model_path"]
			for k in ckpt_keys:
				if k in ckpt_callback_data:
					ckpt_callback_data[k] = ckpt_callback_data[k].replace(
						os.path.abspath(old_wandb_dp),
						os.path.abspath(wandb.run.dir)
					)
			if "best_k_models" in ckpt_callback_data:
				for k in list(ckpt_callback_data["best_k_models"].keys()):
					new_k = k.replace(
						os.path.abspath(old_wandb_dp),
						os.path.abspath(wandb.run.dir)
					)
					ckpt_callback_data["best_k_models"][new_k] = ckpt_callback_data["best_k_models"].pop(k)
			# save modified checkpoint in new wandb dir
			th.save(new_ckpt_data,new_ckpt_fp)
	checkpoint_callback = ModelCheckpoint(
		dirpath=ckpt_dp,
		filename="model-{epoch:03d}",
		monitor=config_d["checkpoint_metric"],
		mode=config_d["checkpoint_metric_mode"],
		save_last=True,
	)
	callbacks.append(checkpoint_callback)

	# setup profiler
	logging.info("setup profiler")
	if wandb_mode != "disabled":
		profile_dp = os.path.join(wandb.run.dir,"profile")
	else:
		profile_dp = "tmp_profile"
	if config_d["profiler"] == "simple":
		profiler = SimpleProfiler(
			dirpath=profile_dp,
			filename="profile"
		)
	elif config_d["profiler"] == "advanced":
		profiler = AdvancedProfiler(
			dirpath=profile_dp,
			filename="profile"
		)
	elif config_d["profiler"] == "pytorch":
		profiler = PyTorchProfiler(
			dirpath=profile_dp,
			filename="profile",
			activities=[
				th.profiler.ProfilerActivity.CPU,
				th.profiler.ProfilerActivity.CUDA
			],
			profile_memory=True,
			record_shapes=True,
			with_flops=True,
			with_stack=True
		)
	else:
		assert config_d["profiler"] == "none", config_d["profiler"]
		# no profiler
		profiler = None

	# setup datasets
	logging.info("setup dataset")
	if config_d["model_type"] == "frag_gnn":
		dataset_cls = SpecMolFragDataset
	elif config_d["model_type"] == "iceberg_gen":
		dataset_cls = SpecMolMagmaGenDataset
	elif config_d["model_type"] == "iceberg_inten":
		dataset_cls = SpecMolMagmaIntenDataset
	else:
		assert config_d["model_type"] == "neims", config_d["model_type"]
		dataset_cls = SpecMolDataset
	train_ds = dataset_cls(split="train",**config_d)
	val_ds = dataset_cls(split="val",**config_d)
	test_ds = dataset_cls(split="test",**config_d)

	# check config for sampler
	if config_d['custom_sampler'] == 'SpecMolFragDynamicBatchSampler':
		assert config_d["model_type"] == "frag_gnn",\
			f"custom sampler {config_d['custom_sampler']} can only be use with frag_gnn model"
		assert config_d['automatic_optimization'] == False,\
			f"custom sampler {config_d['custom_sampler']} can not use in junction of automatic optimization {config_d['automatic_optimization']}"
		assert config_d['specmolfrag_dynamic_batch_sampler_mode'] in ['frag_node','frag_edge'],\
			f"SpecMolFragDynamicBatchSampler only support frag_node or frag_edge mode (given {config_d['specmolfrag_dynamic_batch_sampler_mode']}) "
		assert config_d['specmolfrag_dynamic_batch_sampler_max'] is not None,\
			"SpecMolFragDynamicBatchSampler need a max num"
	
	# setup dataloaders
	logging.info("setup dataloader")
	# is_train_sampler = True
	#train_sampler = SpecMolBatchSampler(train_ds, batch_size=config_d["train_batch_size"], drop_last=config_d["drop_last"]) if is_train_sampler else None
	train_dl_param_d = {
		'dataset':train_ds,
		'num_workers':config_d["num_workers"],
		'collate_fn':train_ds.get_collate_fn(),
	}

	val_dl_param_d = {
		'dataset': val_ds,
		'num_workers':config_d["num_workers"],
		'collate_fn':train_ds.get_collate_fn(),
	}

	if config_d['custom_sampler'] == 'SpecMolFragDynamicBatchSampler':
		# taining 
		train_batchsampler = SpecMolFragDynamicBatchSampler(train_ds, 
								max_num = config_d['specmolfrag_dynamic_batch_sampler_max'], 
								limited_by = config_d['specmolfrag_dynamic_batch_sampler_mode'], 
								shuffle=(not config_d["debug_overfit"]), 
								skip_too_big = True,
								return_batch_at = config_d['train_batch_size'] * config_d['accumulate_grad_batches'])
		train_dl_param_d['batch_sampler'] = train_batchsampler

		# validation
		val_batchsampler = SpecMolFragDynamicBatchSampler(val_ds, 
						max_num = config_d['specmolfrag_dynamic_batch_sampler_max'], 
						limited_by = config_d['specmolfrag_dynamic_batch_sampler_mode'], 
						shuffle=False, 
						skip_too_big = True)
		val_dl_param_d['batch_sampler'] = val_batchsampler

	else:
		train_dl_param_d['batch_size'] = config_d["train_batch_size"]
		train_dl_param_d['shuffle'] = not config_d["debug_overfit"]
		train_dl_param_d['drop_last'] = config_d["drop_last"]
		val_dl_param_d['shuffle'] = False
	
	train_dl = DataLoader(
		**train_dl_param_d
	)

	val_dl = DataLoader(
		**val_dl_param_d
	)

	test_dl = DataLoader(
		test_ds,
		batch_size=config_d["eval_batch_size"],
		shuffle=False,
		num_workers=config_d["num_workers"],
		collate_fn=test_ds.get_collate_fn()
	)

	# setup trainer
	logging.info("setup trainer")
	if config_d["debug_overfit"]:
		overfit_batches = config_d["debug_overfit_batches"]
	else:
		overfit_batches = 0
	log_every_n_steps = min(len(train_dl),config_d["log_every_n_steps"])

	trainer_param_d  = {
		'logger': loggers,
		'callbacks': callbacks,
		'accelerator': config_d["accelerator"],
		'devices': config_d["devices"],
		'min_epochs': config_d["min_epochs"],
		'max_epochs': config_d["max_epochs"],
		'precision': config_d["precision"],
		'log_every_n_steps': config_d["log_every_n_steps"],
		'detect_anomaly': config_d["detect_anomaly"],
		'overfit_batches': overfit_batches,
		'profiler': profiler,
		'num_sanity_val_steps': config_d["num_sanity_val_steps"],
		'enable_progress_bar': config_d["pl_enable_progress_bar"]
	}

	# this things can only set if automatic_optimization 
	if config_d['automatic_optimization']:
		trainer_param_d['accumulate_grad_batches'] = config_d["accumulate_grad_batches"]
		trainer_param_d['gradient_clip_val'] = config_d["gradient_clip_val"]
		trainer_param_d['gradient_clip_algorithm'] = config_d["gradient_clip_algorithm"]
	elif config_d['custom_sampler'] == 'SpecMolFragDynamicBatchSampler':
		# with out this progress will not track dataset length change
		# this will call train_dataloader and val_dataloader before every epoch
		# not sure why this fixed our problem but it works 
		trainer_param_d['reload_dataloaders_every_n_epochs'] = 1

	trainer = pl.Trainer(**trainer_param_d)

	# set determinism
	th.use_deterministic_algorithms(config_d["deterministic"],warn_only=True)

	# register nan hook
	if config_d["nan_module_hook"]:
		th.nn.modules.module.register_module_forward_hook(nan_forward_hook)
		th.nn.modules.module.register_module_full_backward_hook(nan_backward_hook)
	
	# fit
	if config_d["debug_overfit"]:
		assert not is_resume
		logging.info("debug overfit model")
		trainer.fit(model, train_dl)
	else:
		logging.info("fit model")
		if is_resume:
			ckpt_fp = os.path.join(ckpt_dp,"last.ckpt")
			if os.path.isfile(ckpt_fp):
				logging.info(f"resuming from checkpoint: {ckpt_fp}")
			else:
				ckpt_fp = None
		else:
			ckpt_fp = None
		trainer.fit(
			model, 
			train_dl, 
			val_dl,
			ckpt_path=ckpt_fp
		)
	logging.info("callback metrics")

	if config_d["compile"]:
		print(model.dynamo_prof.report())

	if not trainer.interrupted:
		# find checkpoints
		ckpt_fps = glob.glob(os.path.join(ckpt_dp,"*.ckpt"))
		if wandb_mode != "disabled":
			# upload or remove checkpoints
			# if in offline mode, you shouldn't remove the checkpoint before syncing
			for ckpt_fp in ckpt_fps:
				if config_d["upload_checkpoints"]:
					wandb.save(ckpt_fp)
				else:
					os.remove(ckpt_fp)
			# cleanup
			if job_id_fp:
				os.remove(job_id_fp)
		else:
			# remove checkpoints
			for ckpt_fp in ckpt_fps:
				os.remove(ckpt_fp)

	return model
