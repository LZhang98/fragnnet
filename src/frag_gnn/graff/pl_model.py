import torch as th
import torch._dynamo as th_dynamo

from frag_gnn.graff.model import GrAFFModel
from frag_gnn.pl_model import SpectrumPL
from frag_gnn.graff.data_utils import x_map, e_map
from frag_gnn.utils.misc_utils import LOG_ZERO, MAX_CROSS_ENTROPY
from frag_gnn.loss import get_sparse_cross_entropy_fn


class GrAFFPL(SpectrumPL):

	def _setup_model(self):

		node_feats = {
			**x_map,
			**{
				'is_virtual_node': {False: 0, True: 1}
			}
		}
		edge_feats = {
			**e_map,
			**{
				'is_virtual_in_edge': {False: 0, True: 1},
				'is_virtual_out_edge': {False: 0, True: 1}
			}
		}

		self.model = GrAFFModel(
			vocab_size=self.hparams.ann_params["library_size"],
			num_isotope_types=self.hparams.ann_params["max_isotope"]+1,
			encoder_dim=self.hparams.graff_encoder_dim,
			decoder_dim=self.hparams.graff_decoder_dim,
			encoder_depth=self.hparams.graff_encoder_depth,
			decoder_depth=self.hparams.graff_decoder_depth,
			num_eigs=self.hparams.ann_params["num_eigs"],
			eig_dim=self.hparams.graff_eig_dim,
			eig_depth=self.hparams.graff_eig_depth,
			dropout=self.hparams.graff_dropout,
			min_probability=self.hparams.graff_min_probability,
			min_mz=self.hparams.graff_min_mz,
			node_feats=node_feats,
			edge_feats=edge_feats
		)

		# compile
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = self.model.get_compile(backend=self.dynamo_prof,dynamic=True)		

	def _setup_loss_names(self):

		# flag losses for tracking
		loss_names = [
			"loss",
			"primary_loss",
			# "spec_ce",
			# "oos_ce",
			# "ios_ce",
			# "null_formula_prob",
			# "oos_prob",
		]
		self.loss_names = loss_names
		self.metric_names.update(loss_names)

	def _setup_loss_fn(self):

		assert self.hparams.loss_type == "cross_entropy", self.hparams.loss_type
		# assert self.hparams.output_distribution == "peak_marginal", self.hparams.output_distribution

		# cross entropy
		sparse_ce_fn = get_sparse_cross_entropy_fn(
			dist=self.hparams.output_distribution,
			vectorized=self.hparams.loss_vectorized,
			tolerance=self.tolerance,
			relative=self.relative,
			tolerance_min_mz=self.tolerance_min_mz,
			oos_tolerance_multiple=self.hparams.oos_tolerance_multiple,
			gaussian_renormalize=self.hparams.gaussian_renormalize,
			pm_tolerance_multiple=self.hparams.pm_tolerance_multiple,
			loss_batch_size=self.hparams.loss_batch_size
		)

		self._setup_loss_names()

		def _loss_fn(true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			**kwargs):

			batch_size = th.max(true_batch_idxs)+1
			ios_ce, oos_ce, true_oos_logprob, true_oos_e = sparse_ce_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				th.full((batch_size,), LOG_ZERO(true_logprobs.dtype), device=pred_logprobs.device, dtype=pred_logprobs.dtype),
			)
			spec_ce = ios_ce + oos_ce
			if self.hparams.oos_loss:
				raise NotImplementedError
				primary_loss = spec_ce
			else:
				primary_loss = ios_ce
			loss = primary_loss
			loss_d = {
				"loss": loss,
				"primary_loss": primary_loss,
				# "spec_ce": spec_ce
			}
			return loss_d
		
		self.loss_fn = _loss_fn

def test_models():

	from frag_gnn.runner import load_config, init_dataset
	from frag_gnn.utils.misc_utils import print_shapes
	from frag_gnn.utils.misc_utils import to_device

	template_fp = "config/template.yml"
	custom_fp_to_model_cls = {
		# "config/debug_um/debug_graff_auto_um.yml": GrAFFPL,
		"config/debug_m/debug_graff_auto_m.yml": GrAFFPL
	}

	for custom_fp, pl_model_cls in custom_fp_to_model_cls.items():

		print(">>>", custom_fp)

		config_d = load_config(template_fp, custom_fp)
		device = "cuda:0" if config_d["accelerator"] == "gpu" else "cpu"

		ds = init_dataset(config_d)[0]
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

		results = pl_model._common_step(batch, split="train")
		print_shapes(results)

		mean_loss = results["mean_loss"]
		print("> loss")
		print(mean_loss)
		mean_loss.backward()

if __name__ == "__main__":

	from frag_gnn.utils.misc_utils import th_temp_seed

	with th_temp_seed(420):
		test_models()
