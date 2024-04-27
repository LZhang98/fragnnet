from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL

import wandb
import tempfile
import os

# load configs
print("Loading configs")
template_fp = "config/template.yml"
custom_fp = "config/nist20_d4_wr00_inchikey/fraggnn_d4_nodemlp.yml"
config_d = load_config(template_fp, custom_fp)

# load models
print("Loading models")
with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dp:
    api = wandb.Api()
    run = api.run("frag-gnn/frag-gnn/tpo4z1yh")
    ckpt_file, ckpt_epoch_num = None, None
    for file in run.files():
        if file.name.startswith("ckpt/model-epoch="):
            epoch_num = int(file.name.removeprefix("ckpt/model-epoch=").removesuffix(".ckpt"))
            if ckpt_file is None or epoch_num > ckpt_epoch_num:
                ckpt_epoch_num = epoch_num
                ckpt_file = file.name
    assert not ckpt_file is None
    assert not ckpt_epoch_num is None
    # ckpt_file = "last.ckpt"
    ckpt_file_local = run.file(ckpt_file).download(root=temp_dp,replace=False)
    ckpt_fp = ckpt_file_local.name
    model = FragGNNPL.load_from_checkpoint(ckpt_fp)

# get training datapoint metrics
train_datapoint_metrics = {}
for k, v in model.train_datapoint_metrics.items():
    train_datapoint_metrics[k.replace("-",".")] = v.data[:model.train_num_datapoints].detach().cpu()

# get validation datapoint metrics
val_datapoint_metrics = {}
for k, v in model.val_datapoint_metrics.items():
    val_datapoint_metrics[k.replace("-",".")] = v.data[:model.val_num_datapoints].detach().cpu()

# get training aggregated metrics (mean/std)
train_mean_metrics, train_std_metrics = {}, {}
for k, v in model.train_mean_metrics.items():
    train_mean_metrics[k.replace("-",".")] = v.data[ckpt_epoch_num].detach().cpu()
for k, v in model.train_std_metrics.items():
    train_std_metrics[k.replace("-",".")] = v.data[ckpt_epoch_num].detach().cpu()

# get validation aggregated metrics (mean/std)
val_mean_metrics, val_std_metrics = {}, {}
for k, v in model.val_mean_metrics.items():
    val_mean_metrics[k.replace("-",".")] = v.data[ckpt_epoch_num].detach().cpu()
for k, v in model.val_std_metrics.items():
    val_std_metrics[k.replace("-",".")] = v.data[ckpt_epoch_num].detach().cpu()

import pdb; pdb.set_trace()
