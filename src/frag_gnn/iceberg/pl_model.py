import torch as th
import torch.nn as nn

from frag_gnn.pl_model import SpectrumPL
from frag_gnn.iceberg.model import IcebergGenModel


class IcebergGenPL(SpectrumPL):

	def _setup_model(self):

		self.model = IcebergGenModel(
			hidden_size=self.hparams.iceberg_hidden_size,
			layers=self.hparams.iceberg_num_layers,
			set_layers=self.hparams.iceberg_num_set_layers,
			dropout=self.hparams.iceberg_dropout,
			mpnn_type=self.hparams.iceberg_gnn_type,
			pool_op=self.hparams.iceberg_pool_type,
			inject_early=self.hparams.iceberg_inject_early,
			embed_adduct=self.hparams.iceberg_embed_adduct,
			encode_forms=self.hparams.iceberg_encode_forms,
			root_encode=self.hparams.magma_params["root_encode"],
			pe_embed_k=self.hparams.magma_params["pe_embed_k"],
			add_hs=self.hparams.magma_params["add_hs"],
		)

	def _setup_loss_fn(self):

		assert self.hparams.loss_type == "iceberg_cross_entropy", self.hparams.loss_type
		def _loss_fn(
			outputs,
			targets,
			natoms):
			bce_loss_fn = nn.BCELoss(reduction="none")
			loss = bce_loss_fn(outputs, targets.float())
			is_valid = (
				th.arange(loss.shape[1], device=loss.device)[None, :] < natoms[:, None]
			)
			mean_loss = th.sum(loss * is_valid) / th.sum(natoms)
			return loss, mean_loss
		self.loss_fn = _loss_fn

	def forward(self,**kwargs):

		outputs = self.model(
			graphs=kwargs["magma_frag_graphs"],
			root_repr=kwargs["magma_root_reprs"],
			ind_maps=kwargs["magma_inds"],
			broken=kwargs["magma_broken_bonds"],
			adducts=kwargs["magma_adducts"],
			root_forms=kwargs["magma_root_form_vecs"],
			frag_forms=kwargs["magma_frag_form_vecs"],
		)
		return outputs

	def _common_step(self, batch, name="train"):

		batch_size = batch["batch_size"]
		unique_id = batch["spec_unique_id"]
		targets = batch["magma_targ_atoms"]
		natoms = batch["magna_frag_atoms"]
		outputs = self.forward(**batch)
		both = {
			"targets": targets,
			"natoms": natoms,
			"outputs": outputs
		}
		loss, mean_loss = self.loss_fn(**both)
		# just log loss
		self.log(
			f"{split}_batch_loss",
			mean_loss,
			batch_size=batch_size,
			on_epoch=True
		)
		results = {
			"unique_id": unique_id,
			# "smiles": smiles,
			"loss": loss,
			"mean_loss": mean_loss,
			**both
		}
		return results

	def predict_mol(
		self,
		root_smi: str,
		adduct,
		threshold=0,
		device: str = "cpu",
		max_nodes: int = None,
	):

		pass

if __name__ == "__main__":

	from frag_gnn.runner import load_config
	from frag_gnn.iceberg.dataset import SpecMagmaGenDataset
	from frag_gnn.utils.misc_utils import print_shapes

	template_fp = "config/template.yml"
	custom_fp = "config/debug/debug_iceberg_gen.yml"

	config_d = load_config(template_fp, custom_fp)

	model = IcebergGenPL(**config_d)

	ds = SpecMagmaGenDataset(split="train", **config_d)
	dl = th.utils.data.DataLoader(
		ds,
		batch_size=2,
		shuffle=False,
		num_workers=0,
		collate_fn=ds.get_collate_fn(),
	)
	dl_iter = iter(dl)
	batch = next(dl_iter)
	print_shapes(batch)

	outputs = model.forward(**batch)
	print(outputs.shape)

	both = {
		"targets": batch["magma_targ_atoms"],
		"natoms": batch["magma_frag_atoms"],
		"outputs": outputs
	}
	loss, mean_loss = model.loss_fn(**both)
	print(mean_loss)
	mean_loss.backward()
