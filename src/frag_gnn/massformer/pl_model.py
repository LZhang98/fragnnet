import torch as th
import torch._dynamo as th_dynamo

from frag_gnn.massformer.model import MassFormerModel
from frag_gnn.pl_model import BinnedPL
from frag_gnn.massformer.nn_utils import flag_bounded, PolynomialDecayLR


class MassFormerPL(BinnedPL):

	def _setup_model(self):

		self.model = MassFormerModel(
			mlp_hidden_size=self.hparams.mlp_hidden_size,
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			mz_max=self.hparams.mz_max,
			mz_bin_res=self.hparams.mz_bin_res,
			ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
			ff_bidirectional=self.hparams.ff_bidirectional,
			ce_insert_mlp=self.hparams.ce_insert_mlp,
			ce_insert_mlp_lin=self.hparams.ce_insert_mlp_lin,
			m_ce_insert_mlp_embed=self.hparams.m_ce_insert_mlp_embed,
			m_ce_insert_mlp_bin=self.hparams.m_ce_insert_mlp_bin,
			int_embedder=self.hparams.int_embedder,
		)

		assert self.hparams.automatic_optimization

		self._check_ce_params()

		# compile
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = th.compile(self.model,backend=self.dynamo_prof,dynamic=True)

		self.mf_flag = self.hparams.mf_flag
		if self.mf_flag:
			raise NotImplementedError
			assert self.hparams.precision == 32, self.hparams.precision
			assert not self.hparams.automatic_optimization
		
	def configure_optimizers(self):
		
		if self.hparams.optimizer == "adam":
			optimizer_cls = th.optim.Adam
		elif self.hparams.optimizer == "adamw":
			optimizer_cls = th.optim.AdamW
		elif self.hparams.optimizer == "sgd":
			optimizer_cls = th.optim.SGD
		else:
			raise ValueError(f"Unknown optimizer {self.optimizer}")
		
		nopt_params, pt_params = self.model.get_split_params()
		
		optimizer = optimizer_cls(
			[
				{"params": nopt_params, "weight_decay": self.hparams.weight_decay},
				{"params": pt_params, "weight_decay": self.hparams.pt_weight_decay},
			], 
			lr=self.hparams.lr
		)
		ret = {
			"optimizer": optimizer,
		}
		if self.hparams.lr_schedule:
			num_batches = 400 # approximation
			tot_updates = self.hparams.max_epochs * (num_batches // self.hparams.accumulate_grad_batches)
			warmup_updates = int(0.1 * tot_updates)
			scheduler = PolynomialDecayLR(
				optimizer,
				warmup_updates=warmup_updates,
				tot_updates=tot_updates,
				lr=0.0002,
				end_lr=0.000000001,
				power=1.0
			)
			ret["lr_scheduler"] = {
				"scheduler": scheduler,
				"frequency": 1,
				"interval": "step",
			}
		return ret

	# def _common_step(self, batch, split="train"):

	# 	# preprocess spec
	# 	batch_size = batch["batch_size"]
	# 	unique_id = batch["spec_unique_id"]
	# 	smiles = batch["mol_smiles"]
	# 	true_mzs = batch["spec_mzs"]
	# 	true_ints = batch["spec_ints"]
	# 	true_batch_idxs = batch["spec_batch_idxs"]
	# 	true_prec_mzs = batch["spec_prec_mz"]
	# 	spec_per_group = batch["spec_per_group"]
	# 	spec_per_mol = batch["spec_per_mol"]
	# 	group_per_mol = batch["group_per_mol"]
	# 	true_mzs, true_logprobs, true_batch_idxs = self._preproc_spec(
	# 		true_mzs,
	# 		true_ints,
	# 		true_batch_idxs
	# 	)
	# 	if self.mf_flag:
	# 		def flag_forward(perturb):
	# 			b_pred = self.model.forward(perturb=perturb,**batch)
	# 			return b_pred
	# 		def flag_backward(loss):
	# 			# this backward is only meant for generating perturbations
	# 			loss.backward()
	# 		n_graph, n_node = batch["mol_mf_x"].shape[:2]
	# 		embed_dim = self.model.gf.get_embed_dim()
	# 		b_perturb_shape = (n_graph, n_node, embed_dim)
	# 		targ_d = {
	# 			"true_mzs": true_mzs,
	# 			"true_logprobs": true_logprobs,
	# 			"true_batch_idxs": true_batch_idxs,
	# 			"true_prec_mzs": true_prec_mzs,
	# 		}
	# 		optimizer = self.optimizers()
	# 		dev = true_mzs.device
	# 		def flag_loss(pred_d, targ_d):
	# 			both_d = {**pred_d, **targ_d}
	# 			return self.loss_fn(**both_d)["loss"]
	# 		b_loss_agg, b_pred = flag_bounded(
	# 			(self.model, flag_forward, flag_backward),
	# 			b_perturb_shape,
	# 			targ_d,
	# 			optimizer,
	# 			dev,
	# 			flag_loss,
	# 			m=3,
	# 			step_size=0.001,
	# 			mag=0.001,
	# 			mask=None
	# 		)
	# 	else:
	# 		pred = self.forward(**batch)
	# 		both = {
	# 			"true_mzs": true_mzs,
	# 			"true_logprobs": true_logprobs,
	# 			"true_batch_idxs": true_batch_idxs,
	# 			"true_prec_mzs": true_prec_mzs,
	# 			**pred
	# 		}
	# 		loss_d = self.loss_fn(**both)
	# 	loss = loss_d["loss"]
	# 	mean_loss = self.batch_metric_reduce_fn(
	# 		b_metric=loss,
	# 		b_spec_per_group=spec_per_group,
	# 		b_spec_per_mol=spec_per_mol,
	# 		b_group_per_mol=group_per_mol,
	# 		reduce="w_mean"
	# 	)
	# 	total_loss, total_weight = self.batch_metric_reduce_fn(
	# 		b_metric=loss,
	# 		b_spec_per_group=spec_per_group,
	# 		b_spec_per_mol=spec_per_mol,
	# 		b_group_per_mol=group_per_mol,
	# 		reduce="w_sum",
	# 		return_weights=True
	# 	)
	# 	# just log loss
	# 	self.log(
	# 		f"{split}_batch_loss",
	# 		mean_loss,
	# 		batch_size=batch_size,
	# 		on_epoch=True
	# 	)
	# 	results = {
	# 		"unique_id": unique_id,
	# 		"smiles": smiles,
	# 		"mean_loss": mean_loss,
	# 		"total_loss": total_loss,
	# 		"total_weight": total_weight,
	# 		"spec_per_group": spec_per_group,
	# 		"spec_per_mol": spec_per_mol,
	# 		"group_per_mol": group_per_mol,
	# 		**both,
	# 		**loss_d
	# 	}
	# 	assert true_mzs.shape[0] > 0, true_mzs.shape[0]
	# 	with th.no_grad():
	# 		metric_input_d = {
	# 			"true_mzs": true_mzs,
	# 			"true_ints": th.exp(true_logprobs),
	# 			"true_batch_idxs": true_batch_idxs,
	# 			"pred_mzs": pred["pred_mzs"],
	# 			"pred_ints": th.exp(pred["pred_logprobs"]),
	# 			"pred_batch_idxs": pred["pred_batch_idxs"],
	# 			"true_prec_mzs": true_prec_mzs
	# 		}
	# 		metric_output_d = self.metric_fn(**metric_input_d)
	# 		for k,v in metric_output_d.items():
	# 			assert k not in results, k
	# 			results[k] = v
	# 	for metric_name in self.metric_names:
	# 		assert metric_name in results, metric_name
	# 	return results

	# def configue_optimizers(self):

	# 	opt_d = super().configure_optimizers()
	# 	if self.mf_flag:
	# 		mf_optimizer = th.optim.SGD(self.parameters(), lr=0.0)
	# 		mf_scheduler = None
	# 		optimizers = [opt_d["optimizer"], mf_optimizer]
	# 		schedulers = [opt_d["lr_scheduler"], None]
	# 		return optimizers, schedulers
	# 	else:
	# 		return opt_d

def test_massformer():

	from frag_gnn.runner import load_config
	from frag_gnn.dataset import SpecMolFragDataset
	from frag_gnn.utils.misc_utils import print_shapes
	from frag_gnn.utils.misc_utils import to_device

	template_fp = "config/template.yml"
	custom_fps = [
		"config/debug/debug_d3_massformer.yml",
		"config/debug/debug_d3_massformer_mnce.yml"
	]
	pl_model_clses = [
		MassFormerPL,
		MassFormerPL
	]

	for custom_fp, pl_model_cls in zip(custom_fps,pl_model_clses):

		print(custom_fp)

		config_d = load_config(template_fp, custom_fp)
		device = "cuda:0" if config_d["accelerator"] == "gpu" else "cpu"

		ds = SpecMolFragDataset(split="train", **config_d)
		dl = th.utils.data.DataLoader(
			ds,
			batch_size=8,
			shuffle=False,
			num_workers=0,
			collate_fn=ds.get_collate_fn(),
		)
		dl_iter = iter(dl)
		batch = next(dl_iter)
		print("> batch")
		print_shapes(batch)
		batch = to_device(batch,device)

		pl_model = pl_model_cls(**config_d)
		pl_model.train()
		pl_model.to(device)

		outputs = pl_model.forward(**batch)
		print("> train outputs")
		print_shapes(outputs)
		print(th.exp(outputs["pred_logprobs"]).sum())

		results = pl_model._common_step(batch, split="train")
		print_shapes(results)

		mean_loss = results["mean_loss"]
		print("> loss")
		print(mean_loss)
		mean_loss.backward()


if __name__ == "__main__":
	
	from frag_gnn.utils.misc_utils import th_temp_seed

	with th_temp_seed(420):
		test_massformer()
