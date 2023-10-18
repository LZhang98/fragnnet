from frag_gnn.pl_model import SpectrumPL


class IcebergGenPL(SpectrumPL):

	def _setup_model(self):

		self.model = IcebergGenModel(
			hidden_size=self.hparams.iceberg_hidden_size,
			layers=self.hparams.iceberg_layers,
			set_layers=self.hparams.iceberg_set_layers,
			dropout=self.hparams.dropout,
			mpnn_type=self.hparams.mpnn_type,
			pool_op=self.hparams.pool_op,
			node_feats=self.hparams.node_feats,
			pe_embed_k=self.hparams.pe_embed_k,
			max_broken=self.hparams.max_broken,
			root_encode=self.hparams.root_encode,
			inject_early=self.hparams.inject_early,
			embed_adduct=self.hparams.embed_adduct,
			encode_forms=self.hparams.encode_forms,
			add_hs=self.hparams.add_hs,
		)

	def _setup_loss_fn(self):

		def _loss_fn(
			outputs,
			targets,
			natoms):
			bce_loss_fn = nn.BCELoss(reduction="none")
			loss = bce_loss_fn(outputs, targets.float())
			is_valid = (
				torch.arange(loss.shape[1], device=loss.device)[None, :] < natoms[:, None]
			)
			mean_loss = torch.sum(loss * is_valid) / torch.sum(natoms)
			return loss, mean_loss
		self.loss_fn = _loss_fn

	def forward(self,**kwargs):

		outputs = self.model(
			graphs=kwargs["iceberg_graphs"],
			root_repr=kwargs["iceberg_root_repr"],
			ind_maps=kwargs["iceberg_ind_maps"],
			broken=kwargs["iceberg_broken"],
			root_forms=kwargs["iceberg_root_forms"],
			frag_forms=kwargs["iceberg_frag_forms"],
		)
		return outputs

	def _common_step(self, batch, name="train"):

		batch_size = batch["batch_size"]
		unique_id = batch["spec_unique_id"]
		smiles = batch["mol_smiles"]
		targets = batch["iceberg_targets"]
		natoms = batch["iceberg_natoms"]

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
			"smiles": smiles,
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