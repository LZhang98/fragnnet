import torch as th
import numpy as np
from frag_gnn.dataset import SpecMolFragDataset, init_dataloader

import pytest

config_d = {'wandb_entity': 'frag-gnn', 'wandb_project': 'frag-gnn', 'wandb_name': 'debug_sampler_tiny',
			'spec_fp': 'data/proc/nist20mona23/spec_df.pkl',
			'mol_fp': 'data/proc/nist20mona23/mol_df.pkl',
			'split_dp': 'data/split/nist20_wr75_unittest',
						'frag_dp': 'data/frag/nist20_d3_h4_isoFalse/dags',
						'magma_dp': 'data/magma',
						'subsample_params': {'subsample_size': 64, 'subsample_seed': 420, 'train': True, 'val': True, 'test': True},
						'spec_params': {'merge': True, 'sparse': True, 'prec_type': True, 'prec_mz': True, 'prec_mass_diff': True,
										'nce': False, 'ace': False, 'preprocess': False, 'unique_id': True, 'counts': True},
						'mol_params': {'smiles': True, 'fingerprint': False, 'fingerprint_morgan': True,
									   'fingerprint_rdkit': True, 'fingerprint_maccs': True,
									   'pyg': True, 'pyg_node_feats': ['a_onehot', 'a_degree', 'a_hybrid', 'a_formal', 'a_radical', 'a_ring', 'a_mass', 'a_chiral'], 'pyg_edge_feats': ['b_degree'], 'pyg_pe_embed_k': 0, 'pyg_bigraph': True, 'preprocess': False}, 'frag_params': {'pyg': True, 'pyg_node_feats': ['cc', 'base_formula', 'depth', 'h_formulae_idx', 'h_counts'], 'pyg_edge_feats': ['cc', 'base_formula', 'h_range'], 'pyg_bigraph': True, 'num_isotopes': 1, 'formula_peak_mzs': True, 'formula_peak_probs': True, 'compressed': True, 'preload': False, 'preprocess': False}, 'magma_params': {'pe_embed_k': 0, 'root_encode': 'gnn', 'add_hs': False, 'binned_targs': False, 'mz_bin_res': 0.1, 'mz_max': 1500.0, 'sum_ints': True}, 'model_type': 'frag_gnn', 'num_depth': 4, 'num_hs': 4, 'num_elements': 12, 'int_embedder': 'abs-sines', 'int_embedder_tight': False, 'mol_hidden_size': 256, 'mol_num_layers': 4, 'mol_gnn_type': 'GINE', 'mol_dropout': 0.0, 'mol_normalization': 'batch', 'mol_pool_type': 'mean', 'cc_interstage_type': 'direct', 'frag_hidden_size': 256, 'frag_num_layers': 4, 'frag_gnn_type': 'GINE', 'frag_dropout': 0.0, 'frag_normalization': 'batch', 'frag_pool_type': 'mean', 'mlp_output_format': 'formula', 'mlp_hidden_size': 256, 'mlp_dropout': 0.0, 'mlp_normalization': 'none', 'mlp_num_layers': 2, 'mlp_use_residuals': True, 'ff_prec_mz_offset': 0, 'ff_bidirectional': True, 'iceberg_hidden_size': 256, 'iceberg_num_layers': 3, 'iceberg_num_set_layers': 1, 'iceberg_num_frag_set_layers': 0, 'iceberg_num_mlp_layers': 0, 'iceberg_num_gnn_layers': 0, 'iceberg_dropout': 0.0, 'iceberg_gnn_type': 'GGNN', 'iceberg_pool_type': 'avg', 'iceberg_inject_early': False, 'iceberg_embed_adduct': False, 'iceberg_encode_forms': False, 'lr': 0.0004, 'lr_schedule': False, 'weight_decay': 0.01, 'lr_decay_rate': 0.95, 'lr_warmup_steps': 1000, 'lr_decay_steps': 5000, 'optimizer': 'adamw', 'output_distribution': 'gaussian', 'loss_type': 'cross_entropy', 'loss_vectorized': False, 'loss_tolerance_rel': 1e-05, 'loss_tolerance_abs': None, 'auxiliary_scores': ['cos_sim', 'recall', 'wrecall', 'opt_cos_sim', 'true_spec_e', 'true_spec_ne', 'cos_hun', 'precision', 'dice', 'ndcg'], 'ints_thresh': 0.0, 'mz_max': 1500.0, 'mz_bin_res': 0.01, 'eval_mz_bin_res': [0.1, 0.01, 0.005], 'sum_ints': True, 'ints_transform': 'none', 'train_batch_size': 1, 'eval_batch_size': 1, 'accelerator': 'gpu', 'devices': 1, 'logging': ['console', 'wandb'], 'num_workers': 8, 'min_epochs': 100, 'max_epochs': 100, 'gradient_clip_val': 5.0, 'gradient_clip_algorithm': 'norm', 'accumulate_grad_batches': 5, 'precision': 32, 'debug_overfit': False, 'debug_overfit_batches': 10, 'log_every_n_steps': 0, 'primary_loss_weight': 1.0, 'formula_entropy_weight': 0.0, 'node_entropy_weight': 0.0, 'node_formula_entropy_weight': 0.0, 'formula_normalized_entropy_weight': 0.0, 'node_normalized_entropy_weight': 0.0, 'node_formula_normalized_entropy_weight': 0.0, 'skip_edge_loss': True, 'edge_h_range_loss_weight': 0.0, 'edge_h_transfer_loss_weight': 0.0, 'edge_h_range_loss_fn': 'linear', 'edge_h_transfer_loss_fn': 'linear', 'edge_entropy_weight': 0.0, 'edge_normalized_entropy_weight': 0.0, 'seed': 420, 'detect_anomaly': False, 'automatic_optimization': False, 'profiler': 'none', 'compile': True, 'num_log_train_images': 0, 'num_log_val_images': 0, 'deterministic': False, 'drop_last': False, 'num_sanity_val_steps': 2, 'mask_null_formula': False, 'predict_oos': True, 'oos_tolerance_multiple': 3, 'ce_insert_type': 'none', 'dynamic_batch_sampler': True, 'dynamic_batch_sampler_mode': 'frag_edge', 'dynamic_batch_sampler_max': 2000, 'dynamic_batch_sampler_chunk_size': 4, 'group_sampler': False, 'group_sampler_type': 'group', 'group_sampler_max': 10000, 'mp_sharing_strategy': 'file_descriptor', 'check_gradient_norm': False, 'use_tensor_float32': True, 'nan_module_hook': False, 'debug_zero_loss': False, 'checkpoint_metric': 'epoch', 'checkpoint_metric_mode': 'max', 'upload_checkpoints': True, 'pl_enable_progress_bar': True, 'sparse_cosine_similarity': True, 'node_formula_null_aggregate': False, 'debug_val_loop_trigger': False, 'debug_bypass_train_val': False, 'sample_weight': 'none', 'log_hist_metrics': True}


def test_train_split():

	# test sub sample
	train_ds = SpecMolFragDataset(split="train", **config_d)
	assert len(train_ds) == 64

	# setup dataloaders
	train_dl = init_dataloader(train_ds, config_d)
	all_batch_indices_strs = []
	for _ in range(10):
		batch_indices = [i for i in train_dl.batch_sampler]
		batch_indices_flatten =  [item for row in batch_indices for item in row]
		
		# we should have 64 unique indices
		assert len(batch_indices_flatten) == 64
		assert len(set(batch_indices_flatten)) == 64
		assert min(batch_indices_flatten) == 0
		assert max(batch_indices_flatten) == 63

		accumulated_batch_sizes = [0]
		for batch in batch_indices:
			accumulated_batch_sizes.append(len(batch) + accumulated_batch_sizes[-1])

		for i in range(64//5 + 1):
			assert i * 5 in accumulated_batch_sizes
		assert 64 in accumulated_batch_sizes

		all_batch_indices_strs.append(','.join([str(i) for i in batch_indices_flatten]))
		train_dl.batch_sampler._pre_compute_batches()
	
	# check if random suffule works
	assert len(set(all_batch_indices_strs)) == 10

	config_d["debug_overfit"] = True
	train_dl = init_dataloader(train_ds, config_d)
	all_batch_indices_strs = []
	for _ in range(10):
		batch_indices = [i for i in train_dl.batch_sampler]
		batch_indices_flatten =  [item for row in batch_indices for item in row]
		#print(batch_indices_flatten)
		# we should have 64 unique indices
		assert batch_indices_flatten == [i for i in range(64)]

		all_batch_indices_strs.append(','.join([str(i) for i in batch_indices_flatten]))
		train_dl.batch_sampler._pre_compute_batches()
	# without random suffuling, we should have the same order same time
	assert len(set(all_batch_indices_strs)) == 1

