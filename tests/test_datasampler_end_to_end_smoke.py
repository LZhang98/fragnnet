import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

import torch as th
import torch.nn as nn
import pandas as pd
from types import SimpleNamespace
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl

from src.fragnnet.dataset.data_sampler import SpecMolFragDynamicBatchSampler


class FakeDataset:
    def __init__(self, sizes):
        self.sizes = sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        nodes, edges = self.sizes[idx]
        return {"frag_pyg": SimpleNamespace(num_nodes=nodes, num_edges=edges)}


class TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(th.tensor(0.5))

    def training_step(self, batch, batch_idx):
        # batch is list of dataset items due to collate_fn
        batch_size = len(batch)
        # simple differentiable loss so optimizer has gradients
        loss = (self.w - float(batch_size)) ** 2
        loss = loss.to(self.w.device)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return th.optim.SGD([self.w], lr=0.1)


def main():
    sizes = [(1, 2), (3, 4), (2, 1), (8, 2), (5, 6), (2, 2), (4, 1), (6, 6)]
    ds = FakeDataset(sizes)
    sampler = SpecMolFragDynamicBatchSampler(data_source=ds, max_num=10, limited_by='frag_node')
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda x: x)

    model = TinyModel()
    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        accelerator='cpu',
    )

    trainer.fit(model, train_dataloaders=dl)


if __name__ == '__main__':
    main()
