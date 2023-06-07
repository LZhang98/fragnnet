import torch as th
import lightning.pytorch as pl

from frag_gnn.nn.nn_utils import build_lr_scheduler, JointPredictor
from frag_gnn.nn.loss import get_sparse_cross_entropy_fn
from frag_gnn.utils.spec_utils import batched_filter_func, batched_bin_func, calculate_cosine_similarity, get_ints_transform_func, batched_normalize
from frag_gnn.utils.misc_utils import safelog


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
        lr_schedule: bool,
        weight_decay: float,
        lr_decay_rate: float,
        warmup: int,
        optimizer: str,
        num_steps: int,
        output_distribution: str,
        loss_vectorized: bool,
        loss_tolerance_ppm: float,
        loss_tolerance_abs: float,
        auxiliary_scores: list[str],
        ints_thresh: float,
        mz_max: float,
        mz_bin_res: float,
        bin_sum_ints: bool,
        ints_transform: str
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
        self.lr_schedule = lr_schedule
        self.lr = lr
        self.weight_decay = weight_decay
        if optimizer == "adam":
            self.optimizer_cls = th.optim.Adam
        elif optimizer == "adamw":
            self.optimizer_cls = th.optim.AdamW
        elif optimizer == "sgd":
            self.optimizer_cls = th.optim.SGD
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        if loss_tolerance_ppm is not None:
            loss_tolerance = loss_tolerance_ppm
            loss_relative = True
        else:
            assert loss_tolerance_abs is not None
            loss_tolerance = loss_tolerance_abs
            loss_relative = False
        self.loss_fn = get_sparse_cross_entropy_fn(
            output_distribution,
            loss_vectorized,
            loss_tolerance,
            loss_relative,
        )
        def _filter_func(mzs, ints, batch_idxs):
            return batched_filter_func(mzs,ints,batch_idxs,ints_thresh,mz_max)
        self.filter_func = _filter_func
        def _bin_func(mzs, ints, batch_idxs):
            return batched_bin_func(mzs,ints,batch_idxs,mz_bin_res,bin_sum_ints)
        self.bin_func = _bin_func
        self.ints_transform_func = get_ints_transform_func(ints_transform)
        self.normalize_func = batched_normalize
        self.auxiliary_score_fns = {}
        if "cos_sim" in auxiliary_scores:
            def _cos_sim(*args,**kwargs):
                return th.mean(calculate_cosine_similarity(*args,**kwargs),dim=0)
            self.auxiliary_score_fns["cos_sim"] = _cos_sim
        # if "wrecall" in auxiliary_scores:
        #     self.auxiliary_score_fns["wrecall"] = \
        #         lambda *args,**kwargs: calculate_recalls(*args,**kwargs)[1]
        # if "topk_wrecall" in auxiliary_scores:
        #     self.auxiliary_score_fns["topk_wrecall"] = \
        #         lambda *args,**kwargs: calculate_topk_weighted_recalls(*args,**kwargs)[0]

    def preproc_spec(self,spec_mzs,spec_ints,spec_batch_idxs):

        # filter
        spec_mzs, spec_ints, spec_batch_idxs = self.filter_func(
            spec_mzs,
            spec_ints,
            spec_batch_idxs
        )
        # transform
        spec_ints = self.ints_transform_func(spec_ints)
        # renormalize
        spec_ints = self.normalize_func(
            spec_ints,
            spec_batch_idxs
        )
        # log
        spec_ints = safelog(spec_ints)
        return spec_mzs, spec_ints, spec_batch_idxs

    def predict_step(self,**batch_kwargs):
        # TODO: might need to change this
        return self.forward(**batch_kwargs)

    def forward(self,**batch_kwargs):

        # get predictions
        pred_mzs, pred_logits, pred_batch_idxs = self.model.forward(**batch_kwargs)
        # collect in a dictionary
        pred = {
            "pred_mzs": pred_mzs,
            "pred_logits": pred_logits,
            "pred_batch_idxs": pred_batch_idxs
        }
        return pred

    def _common_step(self, batch, name="train"):

        print("_common_step")
        # preprocess spec
        spec_mzs = batch["spec_mzs"]
        spec_ints = batch["spec_ints"]
        spec_batch_idxs = batch["spec_batch_idxs"]
        batch_size = batch["batch_size"]
        spec_mzs, spec_ints, spec_batch_idxs = self.preproc_spec(
            spec_mzs,
            spec_ints,
            spec_batch_idxs
        )
        pred = self.forward(**batch)
        loss = self.loss_fn(
            spec_mzs,
            spec_ints,
            spec_batch_idxs,
            pred["pred_mzs"],
            pred["pred_logits"],
            pred["pred_batch_idxs"]
        )
        self.log(
            f"{name}_loss",
            loss.item(),
            batch_size=batch_size,
            on_epoch=True
        )
        if spec_mzs.shape[0] == 0:
            import pdb; pdb.set_trace()
        for fn_name,fn in self.auxiliary_score_fns.items():
            score = fn(
                spec_mzs,
                th.exp(spec_ints),
                spec_batch_idxs,
                pred["pred_mzs"],
                th.exp(pred["pred_logits"]),
                pred["pred_batch_idxs"]
            )
            self.log(
                f"{name}_{fn_name}",
                score,
                batch_size=batch_size
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
        ret = {
            "optimizer": optimizer,
        }
        if self.lr_schedule:
            scheduler = build_lr_scheduler(
                optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
            )
            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            }
        return ret