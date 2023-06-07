import torch as th
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import logging

from frag_gnn.model import FragGNN
from frag_gnn.frag.frag_utils import NUM_ELEMENTS
from frag_gnn.utils.misc_utils import ConsoleLogger
from frag_gnn.dataset import get_default_ds_params, get_default_fps, SpecMolFragDataset
from frag_gnn.utils.spec_utils import MZ_MAX, MZ_BIN_RES, INTS_THRESH

logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)

# define configs
logger.debug("setup config")
default_config = {
    "num_depth": 5, # max_depth + 1
    "num_hs": 4,
    "num_elements": NUM_ELEMENTS,
    "int_embedder": "abs-sines",
    "mol_node_feats": [
		"a_onehot",
		"a_degree",
		"a_hybrid",
		"a_formal",
		"a_radical",
		"a_ring",
		"a_mass",
		"a_chiral",
	],
    "mol_edge_feats": ["b_degree"],
    "mol_pe_embed_k": 0,
    "mol_hidden_size": 256,
    "mol_num_layers": 4,
    "mol_gnn_type": "GINE",
    "mol_dropout": 0.0,
    "frag_node_feats": ["cc","base_formula","depth"],
    "frag_edge_feats": ["cc","base_formula"],
    "frag_hidden_size": 256,
    "frag_num_layers": 4,
    "frag_gnn_type": "GINE",
    "frag_dropout": 0.0,
    "mlp_hidden_size": 256,
    "mlp_dropout": 0.0,
    "mlp_num_layers": 2,
    "mlp_use_residuals": True,
    "lr": 1e-4,
    "lr_schedule": False,
    "weight_decay": 0.0,
    "lr_decay_rate": 0.95,
    "warmup": 1000,
    "optimizer": "adamw",
    "num_steps": 1000,
    "output_distribution": "gaussian",
    "loss_vectorized": False,
    "loss_tolerance_ppm": 10,
    "loss_tolerance_abs": None, #0.001,
    "auxiliary_scores": ["cos_sim"],
    "ints_thresh": INTS_THRESH,
    "mz_max": MZ_MAX,
    "mz_bin_res": MZ_BIN_RES,
    "bin_sum_ints": True,
    "ints_transform": "log10t3" # none
}

# setup model
logger.debug("setup model")
model = FragGNN(**default_config)

# setup logger
logger.debug("setup logger")
console_logger = ConsoleLogger()
# TODO log to wandb

# setup callbacks

# TODO checkpointing

# setup datasets
logger.debug("setup dataset")
spec_fp, mol_fp, frag_dp, train_fp, val_fp, test_fp = get_default_fps()
spec_params, mol_params, frag_params = get_default_ds_params()
train_ds = SpecMolFragDataset(
    spec_fp=spec_fp,
    mol_fp=mol_fp,
    frag_dp=frag_dp,
    split_fp=train_fp,
    spec_params=spec_params,
    mol_params=mol_params,
    frag_params=frag_params,
)
val_ds = SpecMolFragDataset(
    spec_fp=spec_fp,
    mol_fp=mol_fp,
    frag_dp=frag_dp,
    split_fp=val_fp,
    spec_params=spec_params,
    mol_params=mol_params,
    frag_params=frag_params,
)
test_ds = SpecMolFragDataset(
    spec_fp=spec_fp,
    mol_fp=mol_fp,
    frag_dp=frag_dp,
    split_fp=test_fp,
    spec_params=spec_params,
    mol_params=mol_params,
    frag_params=frag_params,
)

# setup dataloaders
logger.debug("setup dataloader")
train_dl = DataLoader(
    train_ds, 
    batch_size=2, 
    shuffle=True, 
    num_workers=0,
    collate_fn=train_ds.get_collate_fn()
)
val_dl = DataLoader(
    val_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=val_ds.get_collate_fn()
)
test_dl = DataLoader(
    test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=test_ds.get_collate_fn()
)

# setup trainer
logger.debug("setup trainer")
# TODO add gradient clipping
trainer = pl.Trainer(
    logger=[console_logger],
    accelerator="cpu",
    min_steps=2,
    max_steps=2,
)

# fit
logger.debug("fit model")
trainer.fit(model, train_dl)

