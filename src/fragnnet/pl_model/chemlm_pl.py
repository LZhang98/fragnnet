from typing import Any

import torch as th

from fragnnet.model.chemlm_model import ChemLMModel
from fragnnet.pl_model import BinnedPL


class ChemLMPL(BinnedPL):
    """Lightning wrapper for ``ChemLMModel``: a SMILES language model with the NEIMS head.

    The encoder trains end to end at its own learning rate (``chemlm_encoder_lr``); the head
    and metadata embeddings use ``lr``. Embeddings, normalization weights and biases take no
    weight decay. Plan: docs/plans/chemberta3_msgym_plan.md.
    """

    def _setup_model(self) -> None:
        """Build the model after rejecting settings it does not support.

        Raises:
            ValueError: If ``compile`` or ``activation_checkpointing`` is on, or
                ``weight_decay_exclude_norm_bias`` is off.
        """
        if self.hparams.compile:
            raise ValueError("chemlm does not support compile")
        if self.hparams.activation_checkpointing:
            # Whole-model recompute would redraw the random SMILES in backward.
            raise ValueError(
                "chemlm does not support activation_checkpointing; "
                "use chemlm_gradient_checkpointing"
            )
        if not self.hparams.weight_decay_exclude_norm_bias:
            raise ValueError("chemlm requires weight_decay_exclude_norm_bias: True")
        self.model = ChemLMModel(
            ckpt_dp=self.hparams.chemlm_ckpt_dp,
            num_layers=self.hparams.chemlm_num_layers,
            p_trav=self.hparams.chemlm_p_trav,
            gradient_checkpointing=self.hparams.chemlm_gradient_checkpointing,
            seed=self.hparams.seed,
            mlp_hidden_size=self.hparams.mlp_hidden_size,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_num_layers=self.hparams.mlp_num_layers,
            mlp_use_residuals=self.hparams.mlp_use_residuals,
            # The head is NEIMS's, so it reads NEIMS's bottleneck and E1/E4 stay matched.
            bottleneck_factor=self.hparams.neims_bottleneck_factor,
            mz_max=self.hparams.mz_max,
            mz_bin_res=self.hparams.mz_bin_res,
            ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
            ff_bidirectional=self.hparams.ff_bidirectional,
            ff_output_map_size=self.hparams.ff_output_map_size,
            ff_output_activation=self.hparams.ff_output_activation,
            int_embedder=self.hparams.int_embedder,
            ce_insert_type=self.hparams.ce_insert_type,
            ce_insert_location=self.hparams.ce_insert_location,
            ce_insert_merge=self.hparams.ce_insert_merge,
            ce_insert_size=self.hparams.ce_insert_size,
            use_nce=self.hparams.spec_params["nce"],
            nce_mean=self.hparams.nce_mean,
            nce_std=self.hparams.nce_std,
            nce_max=self.hparams.nce_max,
            use_ace=self.hparams.spec_params["ace"],
            ace_max=self.hparams.ace_max,
            ace_mean=self.hparams.ace_mean,
            ace_std=self.hparams.ace_std,
            prec_insert_location=self.hparams.prec_insert_location,
            prec_insert_size=self.hparams.prec_insert_size,
            prec_types=self.hparams.spec_params["prec_types"],
            inst_insert_location=self.hparams.inst_insert_location,
            inst_insert_size=self.hparams.inst_insert_size,
            inst_types=self.hparams.spec_params["inst_types"],
            log_min=self.hparams.log_min,
        )
        self._check_ce_params()

    def _weight_decay_parameter_groups(self) -> list[dict[str, Any]]:
        """Build four optimizer groups: encoder or head, crossed with decayed or exempt.

        Encoder groups carry ``chemlm_encoder_lr`` and head groups ``lr``; ``SpectrumPL``'s
        scheduler scales each group's own base LR. Exempt from decay: every normalization
        weight, bias and 1-D tensor (``_split_weight_decay_parameters``), plus the encoder's
        token, position and token-type embeddings, so rows of tokens absent from the data do not
        shrink while they get no gradient.

        Returns:
            Non-empty parameter groups covering every trainable parameter once.
        """
        decay, no_decay = self._split_weight_decay_parameters()
        encoder_ids = {id(p) for p in self.model.encoder.parameters()}
        embedding_ids = {id(p) for p in self.model.encoder.embeddings.parameters()}
        weight_decay = self.hparams.weight_decay
        groups = {
            ("encoder", True): [],
            ("encoder", False): [],
            ("head", True): [],
            ("head", False): [],
        }
        for parameter in decay:
            branch = "encoder" if id(parameter) in encoder_ids else "head"
            groups[(branch, id(parameter) not in embedding_ids)].append(parameter)
        for parameter in no_decay:
            branch = "encoder" if id(parameter) in encoder_ids else "head"
            groups[(branch, False)].append(parameter)
        lr_by_branch = {"encoder": self.hparams.chemlm_encoder_lr, "head": self.hparams.lr}
        return [
            {
                "params": parameters,
                "lr": lr_by_branch[branch],
                "weight_decay": weight_decay if decayed else 0.0,
            }
            for (branch, decayed), parameters in groups.items()
            if parameters
        ]

    def _grad_norms(self) -> dict[str, th.Tensor]:
        """Return the gradient L2 norm of the embeddings, each encoder layer and the head."""
        branches = {"encoder_embeddings": list(self.model.encoder.embeddings.parameters())}
        for layer_idx, layer in enumerate(self.model.encoder.encoder.layer):
            branches[f"encoder_layer_{layer_idx}"] = list(layer.parameters())
        encoder_ids = {id(p) for p in self.model.encoder.parameters()}
        branches["head"] = [p for p in self.model.parameters() if id(p) not in encoder_ids]
        norms = {}
        for name, parameters in branches.items():
            grads = [p.grad.detach().norm() for p in parameters if p.grad is not None]
            norms[name] = th.stack(grads).norm() if grads else th.zeros(())
        return norms

    def on_before_optimizer_step(self, optimizer: th.optim.Optimizer) -> None:
        """Log the per-branch gradient norms before clipping, which D8 monitors.

        Computed only on the trainer's logging steps; ``log_every_n_steps: 0`` turns it off.
        """
        super().on_before_optimizer_step(optimizer)
        every = self.trainer.log_every_n_steps
        if every <= 0 or self.trainer.global_step % every != 0:
            return
        for name, norm in self._grad_norms().items():
            self.log(f"train_grad_norm/{name}", norm, on_step=True, on_epoch=False)
