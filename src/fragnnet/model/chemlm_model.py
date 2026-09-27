"""
ChemLM: a pretrained SMILES language model with the NEIMS binned spectrum head.

The encoder is a Hugging Face ``RobertaModel`` (ChemBERTa-3) loaded from a local checkpoint
directory, optionally truncated to its first layers. It reads the batch's SMILES strings, is
mean-pooled over non-padding tokens, and feeds the same bidirectional ``SpecFFN`` head that
NEIMS uses, next to the CE, precursor-type and instrument embeddings. The whole forward pass is
differentiable: the spectrum loss trains every encoder parameter, token embeddings included.

Plan: docs/plans/chemberta3_msgym_plan.md.

``transformers`` is imported lazily, so environments without it can still import every other
model.
"""

import functools
import os
import re
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
from rdkit import Chem

from fragnnet.model.base_model import CEModel, InstModel, PrecModel
from fragnnet.model.nn_blocks import SpecFFN

# Unused checkpoint weights that truncating a ``RobertaForMaskedLM`` checkpoint may leave:
# the MLM head and the layers beyond the kept depth. Anything else unused is an error.
_LAYER_KEY_PATTERN = re.compile(r"^(?:roberta\.)?encoder\.layer\.(\d+)\.")


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as err:
        raise ImportError(
            "model_type chemlm needs the `transformers` package (pinned 4.46.3 in FRAGNNET-GPU)"
        ) from err
    return transformers


@functools.cache
def _mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES {smiles!r}")
    return mol


def random_smiles(smiles: list[str], p: float, rng: np.random.Generator) -> list[str]:
    """Replace each SMILES, with probability ``p``, by a random SMILES of the same molecule.

    A replacement is one unrestricted random traversal (RDKit ``doRandom``), drawn with
    ``Chem.MolToRandomSmilesVect(mol, 1, randomSeed=s)`` and a seed ``s`` taken from ``rng``, so
    the same generator state gives the same strings.

    Args:
        smiles: SMILES strings, one per spectrum.
        p: Replacement probability in [0, 1].
        rng: Generator supplying the replacement draws and the RDKit seeds.

    Returns:
        A list of the same length; kept entries are the input strings.

    Raises:
        ValueError: If a SMILES does not parse.
    """
    out = []
    for smi in smiles:
        if rng.random() < p:
            seed = int(rng.integers(1, 2**31 - 1))
            out.append(Chem.MolToRandomSmilesVect(_mol_from_smiles(smi), 1, randomSeed=seed)[0])
        else:
            out.append(smi)
    return out


def _check_loading_info(loading_info: dict[str, list], num_layers: int) -> None:
    """Raise unless every encoder weight came from the checkpoint.

    A missing key would be randomly initialized, and a mismatched one resized; both silently
    change the pretrained model. Unused keys may only be the MLM head and dropped layers.
    """
    if loading_info["missing_keys"] or loading_info["mismatched_keys"]:
        raise ValueError(
            "checkpoint does not cover the encoder: missing "
            f"{loading_info['missing_keys']}, mismatched {loading_info['mismatched_keys']}"
        )
    for key in loading_info["unexpected_keys"]:
        layer = _LAYER_KEY_PATTERN.match(key)
        if key.startswith("lm_head.") or (layer and int(layer.group(1)) >= num_layers):
            continue
        raise ValueError(
            f"unexpected checkpoint weight {key!r} is neither MLM head nor a cut layer"
        )


class ChemLMModel(nn.Module, CEModel, PrecModel, InstModel):
    """Pretrained SMILES encoder, mean pooling, metadata embeddings and the NEIMS ``SpecFFN``.

    Attributes:
        tokenizer: The checkpoint's fast tokenizer; pads to the batch's longest string.
        encoder: ``RobertaModel`` with ``num_layers`` layers and no pooler.
        ffn: The bidirectional binned spectrum head shared with NEIMS.
    """

    def __init__(
        self,
        ckpt_dp: str,
        num_layers: int,
        p_trav: float,
        gradient_checkpointing: bool,
        seed: int,
        mlp_hidden_size: int,
        mlp_dropout: float,
        mlp_num_layers: int,
        mlp_use_residuals: bool,
        bottleneck_factor: float,
        mz_max: float,
        mz_bin_res: float,
        ff_prec_mz_offset: int,
        ff_bidirectional: bool,
        ff_output_map_size: int,
        ff_output_activation: str,
        int_embedder: str,
        ce_insert_type: str,
        ce_insert_location: str,
        ce_insert_merge: bool,
        ce_insert_size: int,
        use_nce: bool,
        nce_mean: float,
        nce_std: float,
        nce_max: float,
        use_ace: bool,
        ace_max: float,
        ace_mean: float,
        ace_std: float,
        prec_insert_location: str,
        prec_insert_size: int,
        prec_types: list[str],
        inst_insert_location: str,
        inst_insert_size: int,
        inst_types: list[str],
        log_min: float,
    ) -> None:
        """Load the encoder and tokenizer from ``ckpt_dp`` and build the head.

        Args:
            ckpt_dp: Local Hugging Face checkpoint directory (weights and tokenizer files);
                never downloaded.
            num_layers: Encoder layers kept, counted from the input: layers 0 to num_layers-1.
            p_trav: Probability that a training SMILES is replaced by a random traversal of the
                same molecule; evaluation always reads the input string.
            gradient_checkpointing: Recompute each encoder layer in backward to save memory.
                Numerically the same gradients.
            seed: Seeds the random-SMILES generator.
            mlp_hidden_size .. log_min: As for ``NeimsModel``; the head is identical.

        Raises:
            FileNotFoundError: If ``ckpt_dp`` is not a directory.
            ValueError: If ``num_layers`` is outside the checkpoint's depth, ``p_trav`` is not a
                probability, or the checkpoint does not supply every encoder weight.
        """
        super().__init__()
        transformers = _import_transformers()
        if not os.path.isdir(ckpt_dp):
            raise FileNotFoundError(f"chemlm checkpoint directory not found: {ckpt_dp}")
        if not 0.0 <= p_trav <= 1.0:
            raise ValueError(f"chemlm_p_trav must be in [0, 1], got {p_trav}")
        ckpt_num_layers = transformers.AutoConfig.from_pretrained(
            ckpt_dp, local_files_only=True
        ).num_hidden_layers
        if not 1 <= num_layers <= ckpt_num_layers:
            raise ValueError(
                f"chemlm_num_layers must be in [1, {ckpt_num_layers}], got {num_layers}"
            )

        # 1. Tokenizer and encoder. The encoder is the checkpoint's first `num_layers` layers
        # plus its embeddings; the MLM head and pooler are never built.
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt_dp, local_files_only=True)
        self.encoder, loading_info = transformers.RobertaModel.from_pretrained(
            ckpt_dp,
            num_hidden_layers=num_layers,
            add_pooling_layer=False,
            attn_implementation="sdpa",
            local_files_only=True,
            output_loading_info=True,
        )
        _check_loading_info(loading_info, num_layers)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        encoder_config = self.encoder.config
        # RoBERTa numbers positions from pad_token_id + 1, so fewer tokens than positions fit.
        self.max_tokens = encoder_config.max_position_embeddings - encoder_config.pad_token_id - 1
        self.p_trav = p_trav
        self.rng = np.random.default_rng(seed)

        # 2. Metadata embeddings, appended to the pooled SMILES embedding as in NEIMS.
        self.mlp_input_dim = encoder_config.hidden_size
        self._ce_init(
            int_embedder=int_embedder,
            ce_insert_type=ce_insert_type,
            ce_insert_location=ce_insert_location,
            ce_insert_merge=ce_insert_merge,
            ce_insert_size=ce_insert_size,
            nce_max=nce_max,
            nce_mean=nce_mean,
            nce_std=nce_std,
            use_nce=use_nce,
            use_ace=use_ace,
            ace_max=ace_max,
            ace_mean=ace_mean,
            ace_std=ace_std,
        )
        self.mlp_input_dim += self.ce_mlp_input_dim
        self._prec_init(
            prec_insert_location=prec_insert_location,
            prec_insert_size=prec_insert_size,
            prec_num_types=len(prec_types),
        )
        self.mlp_input_dim += self.prec_mlp_input_dim
        self._inst_init(
            inst_insert_location=inst_insert_location,
            inst_insert_size=inst_insert_size,
            inst_num_types=len(inst_types),
        )
        self.mlp_input_dim += self.inst_mlp_input_dim

        # 3. The NEIMS head.
        self.ffn = SpecFFN(
            input_size=self.mlp_input_dim,
            hidden_size=mlp_hidden_size,
            mz_max=mz_max,
            mz_bin_res=mz_bin_res,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout,
            prec_mz_offset=ff_prec_mz_offset,
            bidirectional=ff_bidirectional,
            use_residuals=mlp_use_residuals,
            output_map_size=ff_output_map_size,
            output_activation=ff_output_activation,
            log_min=log_min,
            bottleneck_factor=bottleneck_factor,
        )

    def _ce_location_check(self) -> None:
        if self.ce_insert_location not in ["mlp", "none"]:
            raise ValueError(f"ce_insert_location={self.ce_insert_location} not supported")

    def _prec_location_check(self) -> None:
        if self.prec_insert_location not in ["mlp", "none"]:
            raise ValueError(f"prec_insert_location={self.prec_insert_location} not supported")

    def _inst_location_check(self) -> None:
        if self.inst_insert_location not in ["mlp", "none"]:
            raise ValueError(f"inst_insert_location={self.inst_insert_location} not supported")

    def encode(self, smiles: list[str]) -> th.Tensor:
        """Tokenize SMILES and mean-pool the encoder's last hidden states over real tokens.

        The pool includes the ``<s>`` and ``</s>`` tokens and excludes padding.

        Args:
            smiles: B SMILES strings.

        Returns:
            Pooled embeddings of shape (B, hidden_size).

        Raises:
            ValueError: If a string is longer than the encoder's positions allow.
        """
        tokens = self.tokenizer(smiles, padding="longest", return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.max_tokens:
            raise ValueError(
                f"a SMILES tokenizes to {tokens['input_ids'].shape[1]} tokens; "
                f"the encoder holds {self.max_tokens}"
            )
        device = self.encoder.embeddings.word_embeddings.weight.device
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

    def forward(
        self,
        mol_smiles: list[str],
        spec_prec_mz: th.Tensor,
        spec_ce: th.Tensor | None = None,
        spec_ce_batch_idxs: th.Tensor | None = None,
        spec_prec_type: th.Tensor | None = None,
        spec_inst_type: th.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, th.Tensor]:
        """Predict binned spectra from SMILES.

        On training steps with ``p_trav`` above 0, SMILES are first replaced by random
        traversals of the same molecules.

        Args:
            mol_smiles: B SMILES strings, one per spectrum.
            spec_prec_mz: Precursor m/z, shape (B,).
            spec_ce: Primary CE readings, flat over spectra.
            spec_ce_batch_idxs: Spectrum index of each ``spec_ce`` reading.
            spec_prec_type: Precursor-type indices, shape (B,).
            spec_inst_type: Instrument-type indices, shape (B,).
            **kwargs: The rest of the batch; ``spec_ace`` and ``spec_ace_batch_idxs`` are read
                when ACE is embedded.

        Returns:
            ``pred_mzs``, ``pred_logprobs`` and ``pred_batch_idxs`` (sparse, over non-zero bins)
            and ``pred_specs`` (dense, shape (B, num_bins)).
        """
        smiles = list(mol_smiles)
        if self.training and self.p_trav > 0.0:
            smiles = random_smiles(smiles, self.p_trav, self.rng)
        fh = self.encode(smiles)
        batch_size = fh.shape[0]

        ace = kwargs["spec_ace"] if self.use_ace else None
        ace_batch_idxs = kwargs["spec_ace_batch_idxs"] if self.use_ace else None
        ce_embed = self.embed_ce(
            spec_ce, spec_ce_batch_idxs, batch_size, ace=ace, ace_batch_idxs=ace_batch_idxs
        )
        prec_embed = self.embed_prec(spec_prec_type)
        inst_embed = self.embed_inst(spec_inst_type)
        if self.ce_insert_location == "mlp":
            fh = th.cat([fh, ce_embed], dim=1)
        if self.prec_insert_location == "mlp":
            fh = th.cat([fh, prec_embed], dim=1)
        if self.inst_insert_location == "mlp":
            fh = th.cat([fh, inst_embed], dim=1)

        pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(fh, spec_prec_mz)
        return {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
            "pred_specs": pred_specs,
        }
