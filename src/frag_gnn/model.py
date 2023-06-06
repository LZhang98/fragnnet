import torch as th
import pytorch_lightning as pl

from frag_gnn.utils.nn_utils import build_lr_scheduler, JointPredictor
from frag_gnn.nn.loss import get_sparse_cross_entropy_fn


class FragGNN(pl.LightningModule):

    def __init__(
        self,
        num_depth: int,
		num_hs: int,
		num_elements: int,
		int_embedder: str,
		mol_node_feats: list[str],
		mol_edge_feats: list[str],
		mol_pe_embed_k: int,
		mol_hidden_size: int,
		mol_num_layers: int,
		mol_gnn_type: str,
		mol_dropout: float,
		frag_node_feats: list[str],
		frag_edge_feats: list[str],
		frag_hidden_size: int,
		frag_num_layers: int,
		frag_gnn_type: str,
		frag_dropout: float,
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
        lr: float,
        weight_decay: float,
        lr_decay_rate: float,
        warmup: int,
        optimizer: str,
        num_steps: int,
        loss_dist: str,
        loss_vectorized: bool,
        loss_tolerance: float,
        loss_relative: bool,
        auxiliary_scores: list[str],
    ):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = JointPredictor(
            num_depth=num_depth,
            num_hs=num_hs,
            num_elements=num_elements,
            int_embedder=int_embedder,
            mol_node_feats=mol_node_feats,
            mol_edge_feats=mol_edge_feats,
            mol_pe_embed_k=mol_pe_embed_k,
            mol_hidden_size=mol_hidden_size,
            mol_num_layers=mol_num_layers,
            mol_gnn_type=mol_gnn_type,
            mol_dropout=mol_dropout,
            frag_node_feats=frag_node_feats,
            frag_edge_feats=frag_edge_feats,
            frag_hidden_size=frag_hidden_size,
            frag_num_layers=frag_num_layers,
            frag_gnn_type=frag_gnn_type,
            frag_dropout=frag_dropout,
            mlp_hidden_size=mlp_hidden_size,
            mlp_dropout=mlp_dropout,
            mlp_num_layers=mlp_num_layers,
            mlp_use_residuals=mlp_use_residuals
        )
        self.lr = lr
        self.weight_decay = weight_decay
        if self.optimizer == "adam":
            self.optimizer_cls = th.optim.Adam
        elif self.optimizer == "adamw":
            self.optimizer_cls = th.optim.AdamW
        elif self.optimizer == "sgd":
            self.optimizer_cls = th.optim.SGD
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        self.loss_fn = get_sparse_cross_entropy_fn(
            loss_dist,
            loss_vectorized,
            loss_tolerance,
            loss_relative,
        )
        # TODO: bin before calculating score
        # TODO: support binning parameters
        self.auxiliary_score_fns = {}
        if "cos_sim" in auxiliary_scores:
            self.auxiliary_score_fns["cos_sim"] = calculate_cosine_similarity
        if "wrecall" in auxiliary_scores:
            self.auxiliary_score_fns["wrecall"] = \
                lambda *args,**kwargs: calculate_weighted_recalls(*args,**kwargs)[1]
        if "topk_wrecall" in auxiliary_scores:
            self.auxiliary_score_fns["topk_wrecall"] = \
                lambda *args,**kwargs: calculate_weighted_recalls(*args,**kwargs)[0]

    def predict(self):

        pass

    def forward(self,**batch_kwargs):

        pred_mzs, pred_logits, pred_batch_idxs = self.model.forward(**batch_kwargs)
        pred = {
            "pred_mzs": pred_mzs,
            "pred_logits": pred_logits,
            "pred_batch_idxs": pred_batch_idxs
        }
        return pred

    def _common_step(self, batch, name="train"):

        pred = self.forward(**batch)
        loss = self.loss_fn(
            batch["true_mzs"],
            batch["true_logits"],
            batch["true_batch_idxs"],
            pred["pred_mzs"],
            pred["pred_logits"],
            pred["pred_batch_idxs"]
        )
        self.log(
            f"{name}_loss",
            loss.item(),
            batch_size=batch["batch_size"],
            on_epoch=True
        )
        # auxiliary scores
        for fn_name,fn in self.auxiliary_score_fns.items():
            score = fn(
                batch["true_mzs"],
                batch["true_logits"],
                batch["true_batch_idxs"],
                pred["pred_mzs"],
                pred["pred_logits"],
                pred["pred_batch_idxs"]
            )
            self.log(
                f"{name}_{fn_name}",
                score,
                batch_size=batch["batch_size"]
            )

    def training_step(self, batch, batch_idx):

        return self._common_step(batch, name="train")

    def validation_step(self):

        return self._common_step(batch, name="val")

    def test_step(self):

        return self._common_step(batch, name="test")

    def configure_optimizers(self):

        optimizer = self.optimizer_cls(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = build_lr_scheduler(
            optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
        )
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }
        return ret