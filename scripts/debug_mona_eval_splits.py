import pandas as pd
import torch as th
from torch.utils.data import DataLoader

from frag_gnn.dataset import SpecMolFragDataset
from frag_gnn.pl_model import FragGNNPL
from frag_gnn.runner import load_config

# setup config

template_fp = "config/template.yml"
custom_fp = "config/nist20_d4_wr00_inchikey/fraggnn_d3/s0.yml"
config_d = load_config(template_fp, custom_fp)
config_d["split_dp"] = "data/split/debug_nist_mona"
config_d["accelerator"] = "cpu"

# setup datasets

nist_ds = SpecMolFragDataset(split="nist_noq", **config_d) # nist_nodups
mona_ds = SpecMolFragDataset(split="mona_noq", **config_d) # mona_nodups
# make sure nist and mona are in the same order
nist_ds.spec_df = nist_ds.spec_df.sort_values(by="mol_id").reset_index(drop=True)
mona_ds.spec_df = mona_ds.spec_df.sort_values(by="mol_id").reset_index(drop=True)
nist_dl = DataLoader(
	nist_ds,
	batch_size=config_d["eval_batch_size"],
	shuffle=False,
	num_workers=config_d["num_workers"],
	collate_fn=nist_ds.get_collate_fn(),
	drop_last=False
)
mona_dl = DataLoader(
	mona_ds,
	batch_size=config_d["eval_batch_size"],
	shuffle=False,
	num_workers=config_d["num_workers"],
	collate_fn=nist_ds.get_collate_fn(),
	drop_last=False
)

# setup pl model (just for transform)

pl_model = FragGNNPL(**config_d)

# iterate through data

nist_mzs, nist_logprobs, nist_batch_idxs, nist_mol_ids, nist_prec_mzs = [], [], [], [], []
nist_cur_batch_idx = 0
for b_idx, b in enumerate(nist_dl):
	mzs = b["spec_mzs"]
	ints = b["spec_ints"]
	batch_idxs = b["spec_batch_idxs"]
	mzs, logprobs, batch_idxs = pl_model._preproc_spec(
		mzs,
		ints,
		batch_idxs
	)
	mol_ids = b["mol_id"]
	prec_mzs = b["spec_prec_mz"]
	nist_mzs.append(mzs)
	nist_logprobs.append(logprobs)
	nist_batch_idxs.append(batch_idxs+nist_cur_batch_idx)
	nist_mol_ids.extend(mol_ids)
	nist_prec_mzs.append(prec_mzs)
	nist_cur_batch_idx += batch_idxs.max().item() + 1
nist_mzs = th.cat(nist_mzs, dim=0)
nist_logprobs = th.cat(nist_logprobs, dim=0)
nist_batch_idxs = th.cat(nist_batch_idxs, dim=0)
nist_mol_ids = th.tensor(nist_mol_ids, dtype=th.long)
nist_prec_mzs = th.cat(nist_prec_mzs, dim=0)
for t in [nist_mzs, nist_logprobs, nist_batch_idxs, nist_mol_ids, nist_prec_mzs]:
	print(t.shape)

mona_mzs, mona_logprobs, mona_batch_idxs, mona_mol_ids, mona_prec_mzs = [], [], [], [], []
mona_cur_batch_idx = 0
for b_idx, b in enumerate(mona_dl):
	mzs = b["spec_mzs"]
	ints = b["spec_ints"]
	batch_idxs = b["spec_batch_idxs"]
	mzs, logprobs, batch_idxs = pl_model._preproc_spec(
		mzs,
		ints,
		batch_idxs
	)
	mol_ids = b["mol_id"]
	prec_mzs = b["spec_prec_mz"]
	mona_mzs.append(mzs)
	mona_logprobs.append(logprobs)
	mona_batch_idxs.append(batch_idxs+mona_cur_batch_idx)
	mona_mol_ids.extend(mol_ids)
	mona_prec_mzs.append(prec_mzs)
	mona_cur_batch_idx += batch_idxs.max().item() + 1
mona_mzs = th.cat(mona_mzs, dim=0)
mona_logprobs = th.cat(mona_logprobs, dim=0)
mona_batch_idxs = th.cat(mona_batch_idxs, dim=0)
mona_mol_ids = th.tensor(mona_mol_ids, dtype=th.long)
mona_prec_mzs = th.cat(mona_prec_mzs, dim=0)
for t in [mona_mzs, mona_logprobs, mona_batch_idxs, mona_mol_ids, mona_prec_mzs]:
	print(t.shape)

# compare spectra

import pdb; pdb.set_trace()
assert nist_mol_ids.shape[0] == mona_mol_ids.shape[0]
assert th.all(nist_mol_ids == mona_mol_ids)

# nist_mol_ids_full = nist_mol_ids[nist_batch_idxs]
# mona_mol_ids_full = mona_mol_ids[mona_batch_idxs]
# nist_mol_ids_as = th.argsort(nist_mol_ids, dim=0)
# mona_mol_ids_as = th.argsort(mona_mol_ids, dim=0)
# nist_mol_ids_full_un, nist_mol_ids_full_as, nist_mol_ids_full_cs = th.unique(nist_mol_ids_full, return_inverse=True, return_counts=True, dim=0)
# mona_mol_ids_full_un, mona_mol_ids_full_as, mona_mol_ids_full_cs = th.unique(mona_mol_ids_full, return_inverse=True, return_counts=True, dim=0)
# nist_new_batch_idxs = th.repeat_interleave(th.arange(nist_mol_ids_full_un.shape[0]), nist_mol_ids_full_cs)
# mona_new_batch_idxs = th.repeat_interleave(th.arange(mona_mol_ids_full_un.shape[0]), mona_mol_ids_full_cs)

# metric_d = pl_model.metric_fn(
# 	nist_mzs[nist_mol_ids_full_as],
# 	th.exp(nist_logprobs[nist_mol_ids_full_as]),
# 	nist_new_batch_idxs,
# 	mona_mzs[mona_mol_ids_full_as],
# 	th.exp(mona_logprobs[mona_mol_ids_full_as]),
# 	mona_new_batch_idxs,
# 	nist_prec_mzs[nist_mol_ids_as],
# )

metric_d = pl_model.metric_fn(
	nist_mzs,
	th.exp(nist_logprobs),
	nist_batch_idxs,
	mona_mzs,
	th.exp(mona_logprobs),
	mona_batch_idxs,
	nist_prec_mzs,
)

import pdb; pdb.set_trace()