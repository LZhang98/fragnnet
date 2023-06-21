import torch as th
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import logging
import argparse
import yaml
import os
import numpy as np

from frag_gnn.model import FragGNN, get_default_model_params
from frag_gnn.frag.frag_utils import NUM_ELEMENTS
from frag_gnn.utils.pl_utils import ConsoleLogger, PrintGradCallback
from frag_gnn.dataset import get_default_ds_params, get_default_fps, SpecMolFragDataset


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
	args = parser.parse_args()

	# load config
	config_d = load_config(args.template_fp, args.custom_fp)

	# set random seeds
	th.manual_seed(config_d["seed"])
	np.random.seed(config_d["seed"])

	# setup model
	logging.info("setup model")
	# model_config = get_default_model_params()
	model = FragGNN(**config_d)
	model.train()

	# setup logging
	logging.info("setup loggers")
	loggers = []
	console_logger = ConsoleLogger()
	loggers.append(console_logger)
	if args.wandb_mode != "disabled":
		wandb_logger = pl.loggers.WandbLogger(
			project=config_d["wandb_project"],
			name=config_d["wandb_name"],
			offline=args.wandb_mode == "offline",
			entity=config_d["wandb_entity"],
		)
		loggers.append(wandb_logger)

	# setup callbacks
	if args.wandb_mode != "disabled":
		chkpt_dp = os.path.join(wandb_logger.save_dir,"chkpt")
	else:
		chkpt_dp = "tmp_chkpt"
	checkpoint_callback = ModelCheckpoint(
		dirpath=chkpt_dp,
		filename="model-{epoch:02d}",
		monitor="val_loss",
		mode="min",
	)
	callbacks = [checkpoint_callback]
	# print_grad_callback = PrintGradCallback()
	# callbacks = [checkpoint_callback, print_grad_callback]

	# setup datasets
	logging.info("setup dataset")
	train_ds = SpecMolFragDataset(**config_d)
	val_ds = SpecMolFragDataset(**config_d)
	test_ds = SpecMolFragDataset(**config_d)

	# setup dataloaders
	logging.info("setup dataloader")
	train_dl = DataLoader(
		train_ds, 
		batch_size=config_d["train_batch_size"], 
		shuffle=True, 
		num_workers=config_d["num_workers"],
		collate_fn=train_ds.get_collate_fn(),
		drop_last=True
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
		log_every_n_steps=log_every_n_steps,
		detect_anomaly=config_d["detect_anomaly"],
		overfit_batches=10 if config_d["debug_overfit"] else 0,
	)

	# fit
	if config_d["debug_overfit"]:
		logging.info("debug overfit model")
		trainer.fit(model, train_dl)
	else:
		logging.info("fit model")
		trainer.fit(model, train_dl, val_dl)

