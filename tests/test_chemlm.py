"""Tests for the ChemLM model: a pretrained SMILES encoder trained end to end with the NEIMS head.

Every test builds a tiny random RoBERTa checkpoint and a byte-level BPE tokenizer in a temp
directory, so nothing is downloaded. Plan: docs/plans/chemberta3_msgym_plan.md.
"""

import numpy as np
import pytest
import torch as th
import torch.nn.functional as F
from rdkit import Chem

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from fragnnet.model.chemlm_model import (  # noqa: E402
    ChemLMModel,
    _check_loading_info,
    random_smiles,
)
from fragnnet.pl_model import ChemLMPL  # noqa: E402
from fragnnet.runner import load_config  # noqa: E402

SMILES = ["CCO", "O=C(O)c1ccccc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"]
CKPT_LAYERS = 4
MAX_POSITIONS = 66  # 64 tokens once RoBERTa's pad offset is taken


@pytest.fixture(scope="module")
def ckpt_dp(tmp_path_factory) -> str:
    """A 4-layer random ``RobertaForMaskedLM`` and its tokenizer, saved like a HF checkpoint."""
    ckpt = tmp_path_factory.mktemp("tiny_chemlm")
    bpe = tokenizers.ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        SMILES * 20,
        vocab_size=300,
        min_frequency=1,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    vocab_fp, merges_fp = bpe.save_model(str(ckpt))
    tokenizer = transformers.RobertaTokenizerFast(vocab_file=vocab_fp, merges_file=merges_fp)
    tokenizer.save_pretrained(str(ckpt))
    config = transformers.RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=CKPT_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=MAX_POSITIONS,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        type_vocab_size=1,
    )
    th.manual_seed(0)
    transformers.RobertaForMaskedLM(config).save_pretrained(str(ckpt))
    return str(ckpt)


def _model(ckpt_dp: str, **overrides) -> ChemLMModel:
    kwargs = {
        "ckpt_dp": ckpt_dp,
        "num_layers": 2,
        "p_trav": 0.0,
        "gradient_checkpointing": False,
        "seed": 0,
        "mlp_hidden_size": 16,
        "mlp_dropout": 0.0,
        "mlp_num_layers": 1,
        "mlp_use_residuals": True,
        "bottleneck_factor": 0.5,
        "mz_max": 300.0,
        "mz_bin_res": 1.0,
        "ff_prec_mz_offset": 0,
        "ff_bidirectional": True,
        "ff_output_map_size": -1,
        "ff_output_activation": "relu",
        "int_embedder": "abs-sines",
        "ce_insert_type": "qet",
        "ce_insert_location": "mlp",
        "ce_insert_merge": True,
        "ce_insert_size": 8,
        "use_nce": True,
        "nce_mean": 68.3,
        "nce_std": 59.1,
        "nce_max": 300.0,
        "use_ace": True,
        "ace_max": 300.0,
        "ace_mean": 37.8,
        "ace_std": 27.8,
        "prec_insert_location": "none",
        "prec_insert_size": 8,
        "prec_types": ["[M+H]+"],
        "inst_insert_location": "mlp",
        "inst_insert_size": 8,
        "inst_types": ["FT", "QTOF"],
        "log_min": 1e-9,
    }
    kwargs.update(overrides)
    return ChemLMModel(**kwargs)


def _batch(smiles: list[str]) -> dict:
    n = len(smiles)
    return {
        "mol_smiles": list(smiles),
        "spec_prec_mz": th.linspace(100.0, 250.0, n),
        "spec_ce": th.linspace(20.0, 60.0, n),
        "spec_ce_batch_idxs": th.arange(n),
        "spec_ace": th.linspace(10.0, 40.0, n),
        "spec_ace_batch_idxs": th.arange(n),
        "spec_prec_type": th.zeros(n, dtype=th.long),
        "spec_inst_type": th.arange(n) % 2,
    }


def _spectrum_loss(out: dict, seed: int = 1) -> th.Tensor:
    generator = th.Generator().manual_seed(seed)
    target = th.rand(out["pred_specs"].shape, generator=generator)
    return (1.0 - F.cosine_similarity(out["pred_specs"], target, dim=1)).mean()


class TestLoading:
    def test_truncation_keeps_the_first_layers_bit_identical(self, ckpt_dp):
        model = _model(ckpt_dp, num_layers=2)
        full = transformers.RobertaForMaskedLM.from_pretrained(ckpt_dp).roberta.state_dict()
        kept = model.encoder.state_dict()
        assert len(model.encoder.encoder.layer) == 2
        assert model.encoder.pooler is None
        assert not any(key.startswith(("encoder.layer.2.", "encoder.layer.3.")) for key in kept)
        assert not any("lm_head" in name for name, _ in model.named_parameters())
        for key, value in kept.items():
            assert th.equal(value, full[key]), key

    @pytest.mark.parametrize("num_layers", [0, CKPT_LAYERS + 1])
    def test_depth_outside_the_checkpoint_raises(self, ckpt_dp, num_layers):
        with pytest.raises(ValueError, match="chemlm_num_layers"):
            _model(ckpt_dp, num_layers=num_layers)

    def test_invalid_p_trav_and_missing_directory_raise(self, ckpt_dp, tmp_path):
        with pytest.raises(ValueError, match="chemlm_p_trav"):
            _model(ckpt_dp, p_trav=1.5)
        with pytest.raises(FileNotFoundError):
            _model(str(tmp_path / "absent"))

    def test_loading_check_accepts_only_the_mlm_head_and_cut_layers(self):
        ok = {
            "missing_keys": [],
            "mismatched_keys": [],
            "unexpected_keys": ["lm_head.bias", "roberta.encoder.layer.6.output.dense.weight"],
        }
        _check_loading_info(ok, num_layers=6)
        with pytest.raises(ValueError, match="neither MLM head"):
            _check_loading_info(ok, num_layers=7)
        with pytest.raises(ValueError, match="does not cover"):
            _check_loading_info({**ok, "missing_keys": ["embeddings.word_embeddings.weight"]}, 6)


class TestForward:
    def test_output_is_one_normalized_spectrum_per_smiles(self, ckpt_dp):
        model = _model(ckpt_dp).eval()
        out = model(**_batch(SMILES))
        assert out["pred_specs"].shape == (3, 300)
        th.testing.assert_close(out["pred_specs"].sum(dim=1), th.ones(3))
        assert out["pred_batch_idxs"].max().item() == 2

    def test_padding_does_not_change_a_prediction(self, ckpt_dp):
        model = _model(ckpt_dp).eval()
        with th.no_grad():
            alone = model(**_batch(SMILES[:1]))["pred_specs"]
            padded = model(**_batch(SMILES))["pred_specs"][:1]
        th.testing.assert_close(alone, padded, atol=1e-6, rtol=1e-5)

    def test_a_string_longer_than_the_positions_raises(self, ckpt_dp):
        model = _model(ckpt_dp)
        with pytest.raises(ValueError, match="the encoder holds 64"):
            model.encode(["C(O)" * 40])


class TestEndToEndGradient:
    """D8: the spectrum loss alone trains the embeddings and every kept layer."""

    def test_gradient_reaches_the_batch_token_rows_positions_and_every_layer(self, ckpt_dp):
        model = _model(ckpt_dp).train()
        _spectrum_loss(model(**_batch(SMILES))).backward()
        tokens = model.tokenizer(SMILES, padding="longest", return_tensors="pt")
        pad = model.tokenizer.pad_token_id
        embeddings = model.encoder.embeddings

        word_rows = embeddings.word_embeddings.weight.grad.abs().sum(dim=1) > 0
        used = set(tokens["input_ids"].unique().tolist()) - {pad}
        assert set(th.nonzero(word_rows).flatten().tolist()) == used

        position_rows = embeddings.position_embeddings.weight.grad.abs().sum(dim=1) > 0
        longest = int(tokens["attention_mask"].sum(dim=1).max())
        assert set(th.nonzero(position_rows).flatten().tolist()) == set(
            range(pad + 1, pad + 1 + longest)
        )

        for layer_idx, layer in enumerate(model.encoder.encoder.layer):
            for name, parameter in layer.named_parameters():
                if parameter.ndim == 2:
                    assert parameter.grad.abs().sum() > 0, (layer_idx, name)
        assert all(p.requires_grad for p in model.encoder.parameters())

    def test_gradient_checkpointing_gives_the_same_gradients(self, ckpt_dp):
        grads = []
        for checkpointing in (False, True):
            # Same head initialization and the same dropout masks in both runs.
            th.manual_seed(2)
            model = _model(ckpt_dp, gradient_checkpointing=checkpointing).train()
            th.manual_seed(3)
            _spectrum_loss(model(**_batch(SMILES))).backward()
            # The precursor embedding is built but unused here, so it has no gradient.
            grads.append({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
        assert grads[0].keys() == grads[1].keys()
        assert any(name.startswith("encoder.") for name in grads[0])
        for name in grads[0]:
            th.testing.assert_close(grads[0][name], grads[1][name], atol=1e-6, rtol=1e-5)


class TestRandomSmiles:
    def test_same_seed_same_strings_and_each_parses_to_the_input(self):
        first = random_smiles(SMILES * 3, 1.0, np.random.default_rng(7))
        second = random_smiles(SMILES * 3, 1.0, np.random.default_rng(7))
        assert first == second
        for original, drawn in zip(SMILES * 3, first, strict=True):
            assert Chem.CanonSmiles(drawn) == Chem.CanonSmiles(original)

    def test_probability_zero_keeps_the_inputs_and_one_varies_them(self):
        assert random_smiles(SMILES, 0.0, np.random.default_rng(0)) == SMILES
        rng = np.random.default_rng(0)
        draws = {random_smiles([SMILES[2]], 1.0, rng)[0] for _ in range(10)}
        assert len(draws) > 1

    def test_training_draws_random_strings_and_evaluation_reads_the_input(
        self, ckpt_dp, monkeypatch
    ):
        model = _model(ckpt_dp, p_trav=1.0)
        seen = []
        encode = model.encode
        monkeypatch.setattr(model, "encode", lambda smiles: seen.append(smiles) or encode(smiles))
        model.train()(**_batch(SMILES))
        model.eval()(**_batch(SMILES))
        assert seen[0] != SMILES
        assert [Chem.CanonSmiles(s) for s in seen[0]] == [Chem.CanonSmiles(s) for s in SMILES]
        assert seen[1] == SMILES


@pytest.fixture
def pl_config(ckpt_dp) -> dict:
    config = load_config("config", None, None, model_type="chemlm")
    config.update(
        chemlm_ckpt_dp=ckpt_dp,
        chemlm_num_layers=2,
        chemlm_encoder_lr=3e-5,
        lr=4e-4,
        weight_decay=0.01,
        weight_decay_exclude_norm_bias=True,
        loss_type="cosine_distance",
        sparse_cosine_similarity=True,
        mz_max=300.0,
        mz_bin_res=1.0,
        mlp_hidden_size=16,
        mlp_num_layers=1,
        ce_insert_type="qet",
        ce_insert_location="mlp",
        inst_insert_location="mlp",
        lr_schedule=True,
        lr_schedule_type="cosine",
        lr_warmup_steps=2,
        lr_total_steps=10,
    )
    config["spec_params"].update(
        nce=True, ace=True, merge=False, merge_keep_ces=False, inst_types=["FT", "QTOF"]
    )
    return config


class TestLightningModule:
    def test_groups_cover_every_parameter_once_with_the_branch_lr(self, pl_config):
        pl = ChemLMPL(**pl_config)
        groups = pl._weight_decay_parameter_groups()
        grouped = [id(p) for group in groups for p in group["params"]]
        trainable = [id(p) for p in pl.parameters() if p.requires_grad]
        assert sorted(grouped) == sorted(trainable)
        assert len(groups) == 4

        def group_of(parameter):
            return next(g for g in groups if any(p is parameter for p in g["params"]))

        encoder = pl.model.encoder
        for parameter in (
            encoder.embeddings.word_embeddings.weight,
            encoder.embeddings.position_embeddings.weight,
            encoder.encoder.layer[0].attention.self.query.bias,
        ):
            assert group_of(parameter)["weight_decay"] == 0.0
            assert group_of(parameter)["lr"] == pytest.approx(3e-5)
        query = group_of(encoder.encoder.layer[0].attention.self.query.weight)
        assert query["weight_decay"] == pytest.approx(0.01)
        assert query["lr"] == pytest.approx(3e-5)
        head = group_of(pl.model.ffn.in_layer.weight)
        assert head["weight_decay"] == pytest.approx(0.01)
        assert head["lr"] == pytest.approx(4e-4)
        assert group_of(pl.model.ffn.in_layer.bias)["weight_decay"] == 0.0

    def test_cosine_schedule_scales_each_group_from_its_own_lr(self, pl_config):
        pl = ChemLMPL(**pl_config)
        optimizers = pl.configure_optimizers()
        optimizer = optimizers["optimizer"]
        scheduler = optimizers["lr_scheduler"]["scheduler"]
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        lrs = {group["lr"] for group in optimizer.param_groups}
        assert len(lrs) == 2
        assert max(lrs) / min(lrs) == pytest.approx(4e-4 / 3e-5)
        assert max(lrs) < 4e-4

    @pytest.mark.parametrize(
        "override, message",
        [
            ({"compile": True}, "compile"),
            ({"activation_checkpointing": True}, "chemlm_gradient_checkpointing"),
            ({"weight_decay_exclude_norm_bias": False}, "weight_decay_exclude_norm_bias"),
        ],
    )
    def test_unsupported_settings_raise(self, pl_config, override, message):
        with pytest.raises(ValueError, match=message):
            ChemLMPL(**{**pl_config, **override})
