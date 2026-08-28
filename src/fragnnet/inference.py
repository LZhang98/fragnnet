import copy
import os
import argparse

import pandas as pd
import torch as th
import yaml

from fragnnet.dataset import SpecMolFragDataset
from fragnnet.pl_model import FragGNNPL
from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.misc_utils import deep_update
from fragnnet.utils.nn_utils import decompile_jit_ckpt, get_pl_hparams, is_ckpt_compiled
from fragnnet.utils.script_utils import run_inference


def _select_device(config_d: dict, device: str | th.device | None) -> th.device:
	if device is None:
		if config_d.get("accelerator") == "gpu" and th.cuda.is_available():
			return th.device("cuda:0")
		return th.device("cpu")
	if isinstance(device, th.device):
		if device.type == "cuda" and not th.cuda.is_available():
			return th.device("cpu")
		return device
	if device.startswith("cuda") and not th.cuda.is_available():
		return th.device("cpu")
	return th.device(device)


def _decompile_ckpt_if_needed(ckpt: dict) -> dict:
	if "state_dict" not in ckpt:
		raise KeyError("Checkpoint is missing 'state_dict'.")
	has_hparams = "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict)
	if has_hparams and ckpt["hyper_parameters"].get("compile", False):
		return decompile_jit_ckpt(ckpt)
	if is_ckpt_compiled(ckpt):
		out_ckpt = copy.deepcopy(ckpt)
		out_state_dict = {}
		for src_key, value in out_ckpt["state_dict"].items():
			out_key = src_key.replace("._orig_mod", "")
			out_state_dict[out_key] = value
		out_ckpt["state_dict"] = out_state_dict
		if has_hparams:
			out_ckpt["hyper_parameters"]["compile"] = False
		return out_ckpt
	return ckpt


def _prepare_inference_config(
	config_d: dict,
	eval_batch_size: int | None,
	num_workers: int | None,
	disable_preproc: bool,
) -> dict:
	config_d = copy.deepcopy(config_d)
	config_d["compile"] = False
	config_d["track_datapoint_metrics"] = False
	config_d["dynamic_batch_sampler"] = False
	config_d["group_sampler"] = False
	config_d["simple_group_sampler"] = False
	if eval_batch_size is not None:
		config_d["eval_batch_size"] = eval_batch_size
	if num_workers is not None:
		config_d["num_workers"] = num_workers
	if disable_preproc:
		for key in ["spec_params", "mol_params", "frag_params", "magma_params", "ann_params"]:
			if key in config_d and isinstance(config_d[key], dict):
				config_d[key]["preprocess"] = False
		if "frag_params" in config_d and isinstance(config_d["frag_params"], dict):
			config_d["frag_params"]["preload"] = False
	return config_d


class FraGNNetInference:
	"""Reusable inference interface for FraGNNet checkpoints."""

	def __init__(self, model: FragGNNPL, config_d: dict, device: th.device):
		self.model = model
		self.config_d = config_d
		self.device = device

	@classmethod
	def from_checkpoint(
		cls,
		ckpt_fp: str,
		config_d: dict | None = None,
		template_fp: str | None = None,
		custom_fp: str | None = None,
		device: str | th.device | None = None,
		strict: bool = True,
		eval_batch_size: int | None = None,
		num_workers: int | None = None,
		disable_preproc: bool = True,
	):
		if not os.path.isfile(ckpt_fp):
			raise FileNotFoundError(ckpt_fp)

		ckpt = th.load(ckpt_fp, map_location="cpu")
		ckpt = _decompile_ckpt_if_needed(ckpt)

		if config_d is None:
			if template_fp is not None:
				config_d = load_config(template_fp, custom_fp)
			else:
				config_d = get_pl_hparams(ckpt)
		elif template_fp is not None:
			config_from_yaml = load_config(template_fp, custom_fp)
			config_d = deep_update(config_from_yaml, config_d)

		config_d = _prepare_inference_config(
			config_d=config_d,
			eval_batch_size=eval_batch_size,
			num_workers=num_workers,
			disable_preproc=disable_preproc,
		)

		model_type = config_d.get("model_type")
		if model_type != "frag_gnn":
			raise ValueError(f"FraGNNet inference currently supports only model_type='frag_gnn', got: {model_type}")
		model = FragGNNPL(**config_d)
		try:
			model.load_state_dict(ckpt["state_dict"], strict=strict)
		except RuntimeError as exc:
			if strict:
				raise exc
			model.load_state_dict(ckpt["state_dict"], strict=False)

		selected_device = _select_device(config_d, device)
		model.to(selected_device)
		model.eval()

		return cls(model=model, config_d=config_d, device=selected_device)

	def run(
		self,
		split: str = "predict_only",
		spec_df: pd.DataFrame | None = None,
		mol_df: pd.DataFrame | None = None,
		frag_dp: str | None = None,
		batch_cutoff: int = int(1e6),
		output_subset: set[str] | None = None,
		untransform_spec: bool = False,
	):
		"""Run model inference and return a merged output dictionary."""
		local_config = copy.deepcopy(self.config_d)

		if split == "predict_only":
			if spec_df is None or mol_df is None:
				raise ValueError("spec_df and mol_df are required for split='predict_only'.")
			local_config["spec_fp"] = spec_df
			local_config["mol_fp"] = mol_df
		elif spec_df is not None or mol_df is not None:
			raise ValueError("spec_df/mol_df can only be provided when split='predict_only'.")

		local_frag_dp = frag_dp if frag_dp is not None else local_config.get("frag_dp")
		if local_frag_dp is None:
			raise ValueError("frag_dp is required for FraGNNet inference.")
		local_config["frag_dp"] = local_frag_dp

		ds = SpecMolFragDataset(split=split, **local_config)
		dl = init_dataloader(ds, local_config)

		vals = run_inference(
			dl=dl,
			model=self.model,
			device=self.device,
			eval_split=split,
			batch_cutoff=batch_cutoff,
			nb_iso=local_config.get("nb_iso", False),
			output_subset=output_subset,
			untransform_spec=untransform_spec,
		)
		return vals

	@classmethod
	def from_config(cls, config_fp: str):
		"""Initialize FraGNNet inference from one config file."""
		with open(config_fp, "r") as config_file:
			config = yaml.load(config_file, Loader=yaml.FullLoader) or {}
		inference_config = config.pop("inference", {})
		template_fp = inference_config.pop("template_fp", "config/template.yml")
		ckpt_fp = inference_config.pop("ckpt_fp")
		config_d = load_config(template_fp, None)
		config_d = deep_update(config_d, config)
		return cls.from_checkpoint(
			ckpt_fp=ckpt_fp,
			config_d=config_d,
			device=inference_config.get("device"),
			eval_batch_size=inference_config.get("batch_size"),
			num_workers=inference_config.get("num_workers"),
		)


def main():
	parser = argparse.ArgumentParser(description="Run FraGNNet inference from one YAML config.")
	parser.add_argument("config_fp", type=str, help="Inference YAML configuration path.")
	args = parser.parse_args()

	with open(args.config_fp, "r") as config_file:
		config = yaml.load(config_file, Loader=yaml.FullLoader) or {}
	inference_config = config.get("inference", {})
	engine = FraGNNetInference.from_config(args.config_fp)
	split = inference_config.get("split", "predict_only")
	data_kwargs = {}
	if split == "predict_only":
		data_kwargs = {
			"spec_df": pd.read_pickle(config["spec_fp"]),
			"mol_df": pd.read_pickle(config["mol_fp"]),
		}
	vals = engine.run(
		split=split,
		**data_kwargs,
		frag_dp=config.get("frag_dp"),
		batch_cutoff=inference_config.get("batch_cutoff", int(1e6)),
		output_subset=set(inference_config["output_subset"]) if inference_config.get("output_subset") else None,
	)
	output_fp = inference_config.get("output_fp")
	if output_fp is None:
		raise ValueError("inference.output_fp is required when running the inference script.")
	os.makedirs(os.path.dirname(output_fp) or ".", exist_ok=True)
	th.save(vals, output_fp)


if __name__ == "__main__":
	main()

