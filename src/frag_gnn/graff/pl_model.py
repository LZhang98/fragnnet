import torch as th
import torch._dynamo as th_dynamo

from frag_gnn.graff.model import GrAFFModel
from frag_gnn.pl_model import SpectrumPL
from frag_gnn.graff.data_utils import x_map, e_map
from frag_gnn.utils.misc_utils import LOG_ZERO
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
			num_isotope_types=self.hparams.ann_params["max_isotope"],
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
			isotopes=self.hparams.graff_isotopes,
			node_feats=node_feats,
			edge_feats=edge_feats
		)

		# compile
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model.compile_submodules(backend=self.dynamo_prof,dynamic=True)		

	def _setup_loss_fn(self):

		assert self.hparams.loss_type == "cross_entropy", self.hparams.loss_type
		assert self.hparams.output_distribution == "peak_marginal", self.hparams.output_distribution

		# cross entropy
		sparse_ce_fn = get_sparse_cross_entropy_fn(
			self.hparams.output_distribution,
			self.hparams.loss_vectorized,
			self.tolerance,
			self.relative,
			self.hparams.oos_tolerance_multiple,
			self.hparams.gaussian_renormalize,
			self.hparams.pm_tolerance_multiple
		)

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
				th.full((batch_size,), LOG_ZERO, device=pred_logprobs.device, dtype=pred_logprobs.dtype),
			)
			spec_ce = ios_ce + oos_ce
			primary_loss = ios_ce
			loss = primary_loss
			loss_d = {
				"loss": loss,
				"primary_loss": primary_loss,
				"spec_ce": spec_ce
			}
			return loss_d
		
		self.loss_fn = _loss_fn
		loss_names = [
			"loss",
			"primary_loss",
			"spec_ce"
		]
		self.metric_names.update(loss_names)

def test_models():

	from frag_gnn.runner import load_config
	from frag_gnn.utils.misc_utils import print_shapes
	from frag_gnn.utils.misc_utils import to_device
	from frag_gnn.graff.dataset import SpecMolAnnDataset

	template_fp = "config/template.yml"
	custom_fp_to_model_cls = {
		"config/debug/debug_graff.yml": GrAFFPL,
	}

	for custom_fp, pl_model_cls in custom_fp_to_model_cls.items():

		print(">>>", custom_fp)

		config_d = load_config(template_fp, custom_fp)
		device = "cuda:0" if config_d["accelerator"] == "gpu" else "cpu"

		ds = SpecMolAnnDataset(split="train", **config_d)
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
