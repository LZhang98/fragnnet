import torch as th
from torch.utils.data import DataLoader
import torch_geometric as pyg
import lightning.pytorch as pl
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
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

from frag_gnn.model import FragGNN, get_default_model_params
from frag_gnn.frag.frag_utils import NUM_ELEMENTS
from frag_gnn.utils.pl_utils import ConsoleLogger, PrintGradCallback
from frag_gnn.dataset import SpecMolFragDataset, find_largest_batch


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

		for k, v in custom_d.items():
			assert k in config_d, k
			config_d[k] = v

	return config_d


if __name__ == "__main__":

	# setup logger
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[logging.StreamHandler()]
	)

	# # get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-t",
		"--template_fp",
		type=str,
		default="config/template.yml",
		help="path to template config file"
	)
	parser.add_argument(
		"-c",
		"--custom_fp",
		type=str,
		required=False
	)
	parser.add_argument(
		"-w",
		"--wandb_mode",
		type=str, 
		default="online",
		choices=["online","offline","disabled"]
	)
	parser.add_argument(
		"-j",
		"--job_id",
		type=str,
		required=False
	)
	args = parser.parse_args()

	# load config
	config_d = load_config(args.template_fp, args.custom_fp)

	# set random seeds
	th.manual_seed(config_d["seed"])
	np.random.seed(config_d["seed"])

	# setup model
	logging.info("setup model")
	model = FragGNN(**config_d)
	model.train()

	# setup logging
	logging.info("setup loggers")
	loggers = []
	console_logger = ConsoleLogger()
	loggers.append(console_logger)
	if args.wandb_mode != "disabled":
		wandb_d = dict(
			project=config_d["wandb_project"],
			name=config_d["wandb_name"],
			offline=args.wandb_mode == "offline",
			entity=config_d["wandb_entity"],
			resume="allow",
		)
		if args.job_id:
			job_id_fp = os.path.join("job_id",f"{args.job_id}.id")
			if os.path.isfile(job_id_fp):
				is_resume = True
				with open(job_id_fp,"r") as job_id_file:
					text = job_id_file.read()
				run_id, old_wandb_dp = text.split(";")
			else:
				is_resume = False
				run_id = old_wandb_dp = None
			wandb_d["id"] = run_id
		else:
			is_resume = False
			job_id_fp = None
		wandb_logger = pl.loggers.WandbLogger(**wandb_d)
		assert wandb_logger.version == wandb.run.id, (wandb_logger.version, wandb.run.id)
		if job_id_fp:
			with open(job_id_fp,"w") as job_id_file:
				text = f"{wandb_logger.version};{os.path.abspath(wandb.run.dir)}"
				job_id_file.write(text)
		loggers.append(wandb_logger)
	else:
		is_resume = False

	# setup callbacks
	callbacks = []
	if args.wandb_mode != "disabled":
		ckpt_dp = os.path.join(wandb.run.dir,"ckpt")
	else:
		ckpt_dp = "tmp_ckpt"
	os.makedirs(ckpt_dp, exist_ok=True)
	if is_resume:
		# copy the checkpoint files
		old_ckpt_fps = glob.glob(os.path.join(old_wandb_dp,"ckpt","*.ckpt"))
		print(old_ckpt_fps)
		for old_ckpt_fp in old_ckpt_fps:
			new_ckpt_fp = os.path.join(ckpt_dp,os.path.basename(old_ckpt_fp))
			shutil.copyfile(old_ckpt_fp,new_ckpt_fp)
	checkpoint_callback = ModelCheckpoint(
		dirpath=ckpt_dp,
		filename="model-{epoch:03d}",
		monitor="epoch",
		mode="max",
		save_last=True,
	)
	callbacks.append(checkpoint_callback)

	# setup profiler
	logging.info("setup profiler")
	if args.wandb_mode != "disabled":
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
	train_ds = SpecMolFragDataset(split="train",**config_d)
	val_ds = SpecMolFragDataset(split="val",**config_d)
	test_ds = SpecMolFragDataset(split="test",**config_d)

	# setup dataloaders
	logging.info("setup dataloader")
	train_dl = DataLoader(
		train_ds, 
		batch_size=config_d["train_batch_size"], 
		shuffle=(not config_d["debug_overfit"]), 
		num_workers=config_d["num_workers"],
		collate_fn=train_ds.get_collate_fn(),
		drop_last=config_d["drop_last"]
	)
	val_dl = DataLoader(
		val_ds,
		batch_size=config_d["eval_batch_size"],
		shuffle=False,
		num_workers=config_d["num_workers"],
		collate_fn=val_ds.get_collate_fn()
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

	trainer = pl.Trainer(
		logger=loggers,
		callbacks=callbacks,
		accelerator=config_d["accelerator"],
		devices=config_d["devices"],
		min_epochs=config_d["min_epochs"],
		max_epochs=config_d["max_epochs"],
		gradient_clip_val=config_d["gradient_clip_val"],
		gradient_clip_algorithm=config_d["gradient_clip_algorithm"],
		accumulate_grad_batches=config_d["accumulate_grad_batches"],
		precision=config_d["precision"],
		log_every_n_steps=config_d["log_every_n_steps"],
		detect_anomaly=config_d["detect_anomaly"],
		overfit_batches=overfit_batches,
		profiler=profiler,
	)

	# set determinism
	th.use_deterministic_algorithms(config_d["deterministic"],warn_only=True)

	# fit
	if config_d["debug_overfit"]:
		assert not is_resume
		logging.info("debug overfit model")
		trainer.fit(model, train_dl)
	else:
		logging.info("fit model")
		if is_resume:
			ckpt_fp = os.path.join(ckpt_dp,"last.ckpt")
			print(ckpt_fp)
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

	if config_d["compile"]:
		print(model.dynamo_prof.report())

	if args.wandb_mode != "disabled" and not trainer.interrupted:
		# cleanup
		if job_id_fp:
			os.remove(job_id_fp)
		ckpt_fps = glob.glob(os.path.join(ckpt_dp,"*.ckpt"))
		for ckpt_fp in ckpt_fps:
			os.remove(ckpt_fp)