from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL
from frag_gnn.dataset import SpecMolFragDataset, init_dataloader
from frag_gnn.utils.misc_utils import to_device

import torch as th
from tqdm import tqdm

# load configs
print("Loading configs")
template_fp = "config/template.yml"
custom_fp = "config/nist20_d4_wr00_inchikey/fraggnn_d4.yml"
config_d = load_config(template_fp, custom_fp)
# config_d["custom_batch_sampler"] = False
# config_d["eval_batch_size"] = 2

# load datasets
print("Loading datasets")
ds = SpecMolFragDataset(split="val", **config_d)
dl = init_dataloader(ds, **config_d)
import pdb; pdb.set_trace()

# load models
print("Loading models")
ckpt_fp = "ckpt/fraggnn_d4.ckpt"
device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
model = FragGNNPL.load_from_checkpoint(ckpt_fp)
model.eval()

# make predictions on validation data
print("Iterating over data")
for b_idx, b in tqdm(enumerate(dl),total=len(dl)):
    b = to_device(b, device)
    model.validation_step(b, b_idx)

# save metrics
import pdb; pdb.set_trace()
model._consolidate_results("val")
results = model.val_results
print(results.keys())
th.save(results, "ckpt/fraggnn_d4_val_results_fixed.pt")
import pdb; pdb.set_trace()
