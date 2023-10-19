import torch as th
import torch.nn as nn
import torch._dynamo as th_dynamo

from frag_gnn.pl_model import SpectrumPL
from frag_gnn.utils.misc_utils import to_cpu, scatter_reduce
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

		# compile
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()

	def _setup_loss_fn(self):

		assert self.hparams.loss_type == "iceberg_gen_cross_entropy", self.hparams.loss_type
		def _loss_fn(
			outputs,
			targets,
			natoms,
			batch_idxs):
			bce_loss_fn = nn.BCELoss(reduction="none")
			loss = bce_loss_fn(outputs, targets.float())
			is_valid = (
				th.arange(loss.shape[1], device=loss.device)[None, :] < natoms[:, None]
			)
			assert (is_valid.sum(1) == natoms).all()
			loss = th.sum(loss * is_valid, dim=1) / natoms
			loss = scatter_reduce(
				loss, 
				batch_idxs,
				"sum",
				dim=0,
				dim_size=th.max(batch_idxs)+1
			)
			return loss
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

	def _common_step(self, batch, split="train"):

		batch_size = batch["batch_size"]
		unique_id = batch["spec_unique_id"]
		smiles = batch["mol_smiles"]
		targets = batch["magma_targ_atoms"]
		natoms = batch["magma_frag_atoms"]
		batch_idxs = batch["magma_inds"]
		outputs = self.forward(**batch)
		both = {
			"targets": targets,
			"natoms": natoms,
			"outputs": outputs,
			"batch_idxs": batch_idxs
		}
		loss = self.loss_fn(**both)
		mean_loss = th.mean(loss,dim=0)
		# just log loss
		self.log(
			f"{split}_batch_loss",
			mean_loss,
			batch_size=batch_size,
			on_epoch=True
		)
		results = {
			"unique_id": unique_id,
			"smiles": smiles,
			"loss": loss,
			"mean_loss": mean_loss,
			**both
		}
		return results

	def _update_results(self, batch_results, split):

		if split == "train":
			num_log_images = self.hparams.num_log_train_images
			results_attr = "train_results"
			counter_attr = "train_counter"
		elif split == "val":
			num_log_images = self.hparams.num_log_val_images
			results_attr = "val_results"
			counter_attr = "val_counter"
		else:
			raise NotImplementedError(split)
		# filter keys (filtering first to save time/memory)
		keys = [
			"unique_ids",
			"smiles",
		]
		keys.extend(list(self.metric_names))
		unique_ids = batch_results.pop("unique_id")
		batch_results = {k:v for k,v in batch_results.items() if k in keys}
		batch_results["unique_ids"] = unique_ids
		# check all metrics
		assert all([metric_name in batch_results for metric_name in self.metric_names])
		# transfer to cpu
		batch_results = to_cpu(batch_results,detach=True)
		# setup results dict
		if getattr(self,results_attr) is None:
			setattr(self,results_attr,{k:list() for k in keys})
		else:
			assert set(keys) == set(getattr(self,results_attr).keys())
		results_dict = getattr(self,results_attr)
		# add to results dict
		for k,v in batch_results.items():
			results_dict[k].append(v)
		# increment counter
		setattr(self,counter_attr,getattr(self,counter_attr)+unique_ids.shape[0])

	def _log_images(self,split):

		if split == "train":
			num_log_images = self.hparams.num_log_train_images
		elif split == "val":
			num_log_images = self.hparams.num_log_val_images
		if num_log_images > 0:
			raise NotImplementedError("Logging images is not supported for IcebergGenPL")

	def predict_mol(
		self,
		root_smi: str,
		adduct,
		threshold=0,
		device: str = "cpu",
		max_nodes: int = None):

		raise NotImplementedError("Inference is not yet implemented for IcebergGenPL")


if __name__ == "__main__":

	from frag_gnn.runner import load_config
	from frag_gnn.iceberg.dataset import SpecMolMagmaGenDataset
	from frag_gnn.utils.misc_utils import print_shapes

	template_fp = "config/template.yml"
	custom_fp = "config/debug/debug_iceberg_gen.yml"

	config_d = load_config(template_fp, custom_fp)

	model = IcebergGenPL(**config_d)

	ds = SpecMolMagmaGenDataset(split="train", **config_d)
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
