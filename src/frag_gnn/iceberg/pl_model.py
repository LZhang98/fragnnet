import torch as th
import torch.nn as nn
import torch._dynamo as th_dynamo
import numpy as np
import dgl

from frag_gnn.pl_model import SpectrumPL
from frag_gnn.utils.misc_utils import to_cpu, scatter_reduce
import frag_gnn.iceberg.common as common
from frag_gnn.iceberg.model import IcebergGenModel, IcebergIntenModel
from frag_gnn.iceberg.dataset import TreeProcessor
from frag_gnn.iceberg import fragmentation
from frag_gnn.utils.spec_utils import calculate_cosine_similarity, binned_cosine_similarity


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
		assert self.hparams.spec_params["merge"]

		# compile
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()

	def _setup_loss_fn(self):

		assert self.hparams.loss_type == "iceberg_gen_cross_entropy", self.hparams.loss_type
		def _loss_fn(
			outputs,
			targets,
			natoms,
			batch_idxs,
			**kwargs):
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
			loss_d = {
				"loss": loss
			}
			return loss_d
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
		loss_d = self.loss_fn(**both)
		loss = loss_d["loss"]
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
			**both,
			**loss_d
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

	def _reduce_metrics(self, results):

		mean_metrics = {}
		for k, v in results.items():
			if k in self.metric_names:
				mean_metrics[k] = th.mean(v, dim=0)
		return mean_metrics

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
		max_nodes: int = None,
		tree_processor: TreeProcessor = None):
		"""predict_mol.

		Predict a new fragmentation tree from a starting root molecule
		autoregressively. First a new fragment is added to the
		frag_hash_to_entry dict and also put on the stack. Then it is
		fragmented and its "atoms_pulled" and "left_pred" are updated
		accordingly. The resulting new fragments are added to the hash.

		Args:
			root_smi (smi)
			threshold: Leaving probability
			device: Device
			max_nodes (int): Max number to include

		Return:
			Dictionary containing results
		"""

		# Step 1: Get a fragmentation engine for root mol
		engine = fragmentation.FragmentEngine(root_smi)
		max_depth = engine.max_tree_depth
		root_frag = engine.get_root_frag()
		root_form = common.form_from_smi(root_smi)
		root_form_vec = th.tensor(common.formula_to_dense(root_form), dtype=th.float, device=device)
		root_form_vec = root_form_vec.reshape(1, -1)
		adducts = th.tensor([common.ion2onehot_pos[adduct]], dtype=th.long, device=device)

		# Step 2: Featurize the root molecule
		root_graph_dict = tree_processor.featurize_frag(
			frag=root_frag,
			engine=engine,
			add_random_walk=True,
		)

		root_repr = None
		if self.model.root_encode == "gnn":
			root_repr = root_graph_dict["graph"].to(device)
		elif self.model.root_encode == "fp":
			root_fp = th.from_numpy(np.array(common.get_morgan_fp_smi(root_smi)))
			root_repr = root_fp.float().to(device)[None, :]

		form_to_min_score = {}
		frag_hash_to_entry = {}
		frag_to_hash = {}
		stack = [root_frag]
		depth = 0
		root_hash = engine.wl_hash(root_frag)
		frag_to_hash[root_frag] = root_hash
		root_score = engine.score_fragment(root_frag)[1]
		id_ = 0
		# TODO: Compute as in fragment engine
		root_entry = {
			"frag": int(root_frag),
			"frag_hash": root_hash,
			"parents": [],
			"atoms_pulled": [],
			"left_pred": [],
			"max_broken": 0,
			"tree_depth": 0,
			"id": 0,
			"prob_gen": 1,
			"score": root_score,
		}
		id_ += 1
		root_entry.update(engine.atom_pass_stats(root_frag, depth=0))
		form_to_min_score[root_entry["form"]] = root_entry["score"]
		frag_hash_to_entry[root_hash] = root_entry

		# Step 3: Run the autoregressive gen loop
		with th.no_grad():

			# Note: we don't fragment at the final depth
			while len(stack) > 0 and depth < max_depth:
				# Convert all new frags to graphs (stack is to run next)

				new_dgl_dicts = [
					tree_processor.featurize_frag(
						frag=i, engine=engine, add_random_walk=True
					)
					for i in stack
				]
				tuple_list = [
					(i, j)
					for i, j in zip(new_dgl_dicts, stack)
					if i["graph"].num_nodes() > 1
				]

				if len(tuple_list) == 0:
					break

				new_dgl_dicts, stack = zip(*tuple_list)
				mol_batch_graph = [i["graph"] for i in new_dgl_dicts]
				frag_forms = [i["form"] for i in new_dgl_dicts]
				frag_form_vecs = [common.formula_to_dense(i) for i in frag_forms]
				frag_form_vecs = th.tensor(np.array(frag_form_vecs), dtype=th.float, device=device)

				new_frag_hashes = [engine.wl_hash(i) for i in stack]

				frag_to_hash.update(dict(zip(stack, new_frag_hashes)))

				frag_batch = dgl.batch(mol_batch_graph).to(device)
				inds = th.zeros(frag_batch.batch_size).long().to(device)

				# Note: Can speed by reducing redundant root graph passes
				# TODO: Figure out how to include frag form vec and root form
				# vec

				# TODO: Compute broken for each of these...

				broken_nums_ar = np.array(
					[frag_hash_to_entry[i]["max_broken"] for i in new_frag_hashes]
				)
				broken_nums_tensor = th.tensor(broken_nums_ar, dtype=th.float, device=device)

				pred_leaving = self.model.forward(
					graphs=frag_batch,
					root_repr=root_repr,
					ind_maps=inds,
					broken=broken_nums_tensor,
					adducts=adducts,
					root_forms=root_form_vec,
					frag_forms=frag_form_vecs,
				)
				depth += 1

				# Rank order all the atom preds and predictions
				# Continuously add items to the stack as long as they maintain
				# the max node constraint ranked by prob

				# Get all frag probabilities and sort them
				cur_probs = sorted(
					[i["prob_gen"] for i in frag_hash_to_entry.values()]
				)[::-1]
				if max_nodes is None or len(cur_probs) < max_nodes:
					min_prob = threshold
				elif max_nodes is not None and len(cur_probs) >= max_nodes:
					min_prob = cur_probs[max_nodes - 1]
				else:
					raise NotImplementedError()

				new_items = list(
					zip(stack, new_frag_hashes, pred_leaving, new_dgl_dicts)
				)
				sorted_order = []
				for item_ind, item in enumerate(new_items):
					frag_hash = item[1]
					pred_vals_f = item[2]
					parent_prob = frag_hash_to_entry[frag_hash]["prob_gen"]
					for atom_ind, (atom_pred, prob_gen) in enumerate(
						zip(pred_vals_f, parent_prob * pred_vals_f)
					):

						sorted_order.append(
							dict(
								item_ind=item_ind,
								atom_ind=atom_ind,
								prob_gen=prob_gen.item(),
								atom_pred=atom_pred.item(),
							)
						)

				sorted_order = sorted(sorted_order, key=lambda x: -x["prob_gen"])
				new_stack = []

				# Process ordered list continuously
				for new_item in sorted_order:
					prob_gen = new_item["prob_gen"]
					atom_ind = new_item["atom_ind"]
					atom_pred = new_item["atom_pred"]
					item_ind = new_item["item_ind"]

					# Filter out on minimum prob
					if prob_gen <= min_prob:
						continue

					# Calc stack ind
					orig_entry = new_items[item_ind]
					frag_int = orig_entry[0]
					frag_hash = orig_entry[1]
					dgl_dict = orig_entry[3]

					# Get atom ind
					atom = dgl_dict["new_to_old"][atom_ind]

					# Calc remove dict
					out_dicts = engine.remove_atom(frag_int, int(atom))

					# Update atoms_pulled for parent
					frag_hash_to_entry[frag_hash]["atoms_pulled"].append(int(atom))
					frag_hash_to_entry[frag_hash]["left_pred"].append(float(atom_pred))
					parent_broken = frag_hash_to_entry[frag_hash]["max_broken"]

					for out_dict in out_dicts:
						out_hash = out_dict["new_hash"]
						out_frag = out_dict["new_frag"]
						rm_bond_t = out_dict["rm_bond_t"]
						frag_to_hash[out_frag] = out_hash
						current_entry = frag_hash_to_entry.get(out_hash)

						max_broken = parent_broken + rm_bond_t

						# Define probability of generating
						if current_entry is None:
							score = engine.score_fragment(int(out_frag))[1]

							new_stack.append(out_frag)
							new_entry = {
								"frag": int(out_frag),
								"frag_hash": out_hash,
								"score": score,
								"id": id_,
								"parents": [frag_hash],
								"atoms_pulled": [],
								"left_pred": [],
								"max_broken": max_broken,
								"tree_depth": depth,
								"prob_gen": prob_gen,
							}
							id_ += 1
							new_entry.update(
								engine.atom_pass_stats(out_frag, depth=max_broken)
							)

							# reset to best score
							temp_form = new_entry["form"]
							prev_best_score = form_to_min_score.get(
								temp_form, float("inf")
							)
							form_to_min_score[temp_form] = min(
								new_entry["score"], prev_best_score
							)
							frag_hash_to_entry[out_hash] = new_entry

						else:
							current_entry["parents"].append(frag_hash)
							current_entry["prob_gen"] = max(
								current_entry["prob_gen"], prob_gen
							)

						# Update cur probs
						# This is inefficeint and can be made smarter withotu
						# doing another minimum calculation
						cur_probs = sorted(
							[i["prob_gen"] for i in frag_hash_to_entry.values()]
						)[::-1]
						if max_nodes is None or len(cur_probs) < max_nodes:
							min_prob = threshold
						elif max_nodes is not None and len(cur_probs) >= max_nodes:
							min_prob = cur_probs[max_nodes - 1]
						else:
							raise NotImplementedError()

				# Truncate stack; this should be handled above by min prob
				# if max_nodes is not None:
				#    new_stack = sorted(new_stack,
				#                       key=lambda x:
				#                       -frag_hash_to_entry[frag_to_hash[x]]['prob_gen'])
				#    new_stack = new_stack[:max_nodes]
				stack = new_stack

		# Only get min score for ech formula
		frag_hash_to_entry = {
			k: v
			for k, v in frag_hash_to_entry.items()
			if form_to_min_score[v["form"]] == v["score"]
		}

		if max_nodes is not None:
			sorted_keys = sorted(
				list(frag_hash_to_entry.keys()),
				key=lambda x: -frag_hash_to_entry[x]["prob_gen"],
			)
			frag_hash_to_entry = {
				k: frag_hash_to_entry[k] for k in sorted_keys[:max_nodes]
			}
		return frag_hash_to_entry


class IcebergIntenPL(SpectrumPL):

	def _setup_model(self):

		self.model = IcebergIntenModel(
			hidden_size=self.hparams.iceberg_hidden_size,
			mlp_layers=self.hparams.iceberg_num_mlp_layers,
			gnn_layers=self.hparams.iceberg_num_gnn_layers,
			set_layers=self.hparams.iceberg_num_set_layers,
			frag_set_layers=self.hparams.iceberg_num_frag_set_layers,
			dropout=self.hparams.iceberg_dropout,
			mpnn_type=self.hparams.iceberg_gnn_type,
			pool_op=self.hparams.iceberg_pool_type,
			inject_early=self.hparams.iceberg_inject_early,
			embed_adduct=self.hparams.iceberg_embed_adduct,
			encode_forms=self.hparams.iceberg_encode_forms,
			root_encode=self.hparams.magma_params["root_encode"],
			pe_embed_k=self.hparams.magma_params["pe_embed_k"],
			add_hs=self.hparams.magma_params["add_hs"],
			binned_targs=self.hparams.magma_params["binned_targs"],
			mz_bin_res=self.hparams.magma_params["mz_bin_res"],
			mz_max=self.hparams.magma_params["mz_max"],
			sum_ints=self.hparams.magma_params["sum_ints"],
		)
		if self.hparams.magma_params["binned_targs"]:
			assert self.hparams.mz_bin_res == self.hparams.magma_params["mz_bin_res"]
			assert self.hparams.mz_max == self.hparams.magma_params["mz_max"]
			assert self.hparams.sum_ints == self.hparams.magma_params["sum_ints"]
		assert self.hparams.spec_params["merge"]

		# compile
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()

	def _setup_loss_fn(self):

		# cosine distance
		if self.hparams.loss_type == "cosine_distance_dense":
			def _loss_fn(true_spec, pred_spec, **kwargs):
				loss = 1. - binned_cosine_similarity(true_spec, pred_spec)
				loss_d = {
					"loss": loss,
					"spec_cd": loss
				}
				return loss_d
		elif self.hparams.loss_type == "cosine_distance":
			def _loss_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				**kwargs):
				loss = 1. - calculate_cosine_similarity(
					true_mzs,
					th.exp(true_logprobs),
					true_batch_idxs,
					pred_mzs,
					th.exp(pred_logprobs),
					pred_batch_idxs,
					mz_bin_res=self.hparams.mz_bin_res,
					mz_max=self.hparams.mz_max,
					sum_ints=self.hparams.sum_ints
				)
				loss_d = {
					"loss": loss,
					"spec_cd": loss
				}
				return loss_d
		else:
			raise ValueError(f"{self.hparams.loss_type} is not a valid loss type for IcebergIntenPL")
		self.loss_fn = _loss_fn
		loss_names = ["loss", "spec_cd"]
		self.metric_names.update(loss_names)

	def forward(self,**kwargs):

		outputs = self.model(
			graphs=kwargs["magma_frag_graphs"],
			root_repr=kwargs["magma_root_reprs"],
			ind_maps=kwargs["magma_inds"],
			num_frags=kwargs["magma_num_frags"],
			broken=kwargs["magma_broken_bonds"],
			adducts=kwargs["magma_adducts"],
			max_add_hs=kwargs["magma_max_add_hs"],
			max_remove_hs=kwargs["magma_max_remove_hs"],
			masses=kwargs["magma_masses"],
			root_forms=kwargs["magma_root_form_vecs"],
			frag_forms=kwargs["magma_frag_form_vecs"],
		)
		return outputs

	def _common_step(self, batch, split="train"):

		batch_size = batch["batch_size"]
		unique_id = batch["spec_unique_id"]
		smiles = batch["mol_smiles"]
		true_mzs = batch["spec_mzs"]
		true_ints = batch["spec_ints"]
		true_batch_idxs = batch["spec_batch_idxs"]
		true_prec_mzs = batch["spec_prec_mz"]
		spec_per_group = batch["spec_per_group"]
		spec_per_mol = batch["spec_per_mol"]
		group_per_mol = batch["group_per_mol"]
		true_mzs, true_logprobs, true_batch_idxs = self._preproc_spec(
			true_mzs,
			true_ints,
			true_batch_idxs
		)
		pred = self.forward(**batch)
		both = {
			"true_mzs": true_mzs,
			"true_logprobs": true_logprobs,
			"true_batch_idxs": true_batch_idxs,
			**pred
		}
		if "magma_inten_targs" in batch:
			assert self.hparams.magma_params["binned_targs"]
			both["true_spec"] = batch["magma_inten_targs"]
		loss_d = self.loss_fn(**both)
		loss = loss_d["loss"]
		mean_loss = self.batch_metric_reduce_fn(
			b_metric=loss,
			b_spec_per_group=spec_per_group,
			b_spec_per_mol=spec_per_mol,
			b_group_per_mol=group_per_mol,
			reduce="w_mean"
		)
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
			"mean_loss": mean_loss,
			"spec_per_group": spec_per_group,
			"spec_per_mol": spec_per_mol,
			"group_per_mol": group_per_mol,
			**both,
			**loss_d
		}
		assert true_mzs.shape[0] > 0, true_mzs.shape[0]
		with th.no_grad():
			metric_input_d = {
				"true_mzs": true_mzs,
				"true_ints": th.exp(true_logprobs),
				"true_batch_idxs": true_batch_idxs,
				"pred_mzs": pred["pred_mzs"],
				"pred_ints": th.exp(pred["pred_logprobs"]),
				"pred_batch_idxs": pred["pred_batch_idxs"],
				"true_prec_mzs": true_prec_mzs
			}
			metric_output_d = self.metric_fn(**metric_input_d)
			for k,v in metric_output_d.items():
				assert k not in results, k
				results[k] = v
		for metric_name in self.metric_names:
			assert metric_name in results, metric_name
		return results


def test_gen():

	from frag_gnn.runner import load_config
	from frag_gnn.iceberg.dataset import SpecMolMagmaGenDataset
	from frag_gnn.utils.misc_utils import print_shapes

	template_fp = "config/template.yml"
	custom_fp = "config/iceberg/d3_iceberg_gen.yml"

	config_d = load_config(template_fp, custom_fp)

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
	print("> batch")
	print_shapes(batch)

	pl_model = IcebergGenPL(**config_d)
	pl_model.train()

	outputs = pl_model.forward(**batch)
	print("> train outputs")
	print(outputs.shape)
	
	both = {
		"targets": batch["magma_targ_atoms"],
		"natoms": batch["magma_frag_atoms"],
		"outputs": outputs,
		"batch_idxs": batch["magma_inds"]
	}
	loss_d = pl_model.loss_fn(**both)
	mean_loss = th.mean(loss_d["loss"],dim=0)
	print("> loss")
	print(mean_loss)
	mean_loss.backward()

	# test inference
	pl_model.eval()
	# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
	smiles = "OCC(CO)(CO)NCCCNC(CO)(CO)CO"
	adduct = "[M+H]+"
	device = "cpu"
	tree_processor = SpecMolMagmaGenDataset.init_tree_processor(	
		**config_d["magma_params"]
	)
	outputs = pl_model.predict_mol(
		root_smi=smiles,
		adduct=adduct,
		device=device,
		tree_processor=tree_processor,
	)
	print("> inference outputs")
	print_shapes(next(iter(outputs.values())))

def test_inten():

	from frag_gnn.runner import load_config
	from frag_gnn.iceberg.dataset import SpecMolMagmaIntenDataset
	from frag_gnn.utils.misc_utils import print_shapes
	from frag_gnn.utils.spec_utils import bin_func, batched_bin_func, binned_cosine_similarity
	import torch.nn.functional as F
	from frag_gnn.utils.misc_utils import to_device

	template_fp = "config/template.yml"
	custom_fps = [
		"config/iceberg/d3_iceberg_inten_100.yml",
		"config/iceberg/d3_iceberg_inten_100_dense.yml",
	]
	
	for custom_fp in custom_fps:

		print(custom_fp)

		config_d = load_config(template_fp, custom_fp)
		device = "cuda:0" if config_d["accelerator"] == "gpu" else "cpu"

		ds = SpecMolMagmaIntenDataset(split="train", **config_d)
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

		pl_model = IcebergIntenPL(**config_d)
		pl_model.train()
		pl_model.to(device)

		outputs = pl_model.forward(**batch)
		print("> train outputs")
		print_shapes(outputs)
		print(th.exp(outputs["pred_logprobs"]).sum())
		if "pred_spec" in outputs:
			print(outputs["pred_spec"].sum())

		results = pl_model._common_step(batch, split="train")
		print_shapes(results)

		mean_loss = results["mean_loss"]
		print("> loss")
		print(mean_loss)
		mean_loss.backward()

	# # check that the spectra are consistent
	# spec_mzs = batch["spec_mzs"]
	# spec_ints = batch["spec_ints"]
	# spec_batch_idxs = batch["spec_batch_idxs"]
	# # print("pre_spec_max",batchwise_max(spec_ints,spec_batch_idxs))
	# spec_mzs, spec_ints, spec_batch_idxs = pl_model._preproc_spec(
	# 	spec_mzs,
	# 	spec_ints,
	# 	spec_batch_idxs
	# )
	# true_spec_2 = batched_bin_func(
	# 	spec_mzs,
	# 	th.exp(spec_ints),
	# 	spec_batch_idxs,
	# 	mz_bin_res=config_d["spec_params"]["mz_bin_res"],
	# 	mz_max=config_d["spec_params"]["mz_max"],
	# 	sum_ints=config_d["spec_params"]["sum_ints"],
	# )
	# print(binned_cosine_similarity(true_spec,true_spec_2))


if __name__ == "__main__":

	from frag_gnn.utils.misc_utils import th_temp_seed

	with th_temp_seed(420):
		test_gen()
	# with th_temp_seed(420):
	# 	test_inten()