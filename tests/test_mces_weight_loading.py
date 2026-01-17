"""
Test script to verify MCES pretrained weights are correctly loaded.

Run this after training starts to check:
1. Weights were actually loaded (not still random)
2. Weights can be frozen if specified
3. Gradients flow correctly
"""

import pytest
import torch as th


@pytest.mark.skip(reason="integration helper; requires real model")
def test_mces_weight_loading(model, mces_ckpt_path=None):
    """
    Test if MCES pretrained weights are loaded correctly.

    Args:
        model: FraGNNetPL model instance
        mces_ckpt_path: Path to MCES checkpoint (optional)
    """

    print("\n" + "=" * 70)
    print("Testing MCES Pretrained Weight Loading")
    print("=" * 70)

    # 1. Check if pretrained weights are enabled
    if not hasattr(model.hparams, "mces_pretrain"):
        print("❌ mces_pretrain not in hparams")
        return

    if not model.hparams.mces_pretrain:
        print("⚠️  MCES pretraining is disabled (mces_pretrain=False)")
        return

    print("✅ MCES pretraining enabled")
    print(f"   Checkpoint: {model.hparams.mces_pretrain_ckpt_fp}")

    # 2. Check shared components exist
    shared_components = ["mol_embedder", "mol_pool"]
    for comp in shared_components:
        if hasattr(model.model, comp):
            print(f"✅ {comp} exists in model")
        else:
            print(f"❌ {comp} NOT found in model")
            return

    # 3. Check if weights are frozen (if specified)
    if hasattr(model.hparams, "freeze_mces_weights"):
        if model.hparams.freeze_mces_weights:
            print("🔒 Freeze pretrained weights: ENABLED")

            frozen_count = 0
            trainable_count = 0

            for name, param in model.model.named_parameters():
                is_shared = any(name.startswith(comp) for comp in shared_components)
                if is_shared:
                    if param.requires_grad:
                        trainable_count += 1
                        print(f"   ⚠️  {name}: TRAINABLE (expected frozen)")
                    else:
                        frozen_count += 1

            print(f"   Frozen params: {frozen_count}")
            print(f"   Trainable params: {trainable_count}")

            if trainable_count > 0:
                print(f"   ⚠️  WARNING: {trainable_count} shared params are trainable!")
        else:
            print("🔓 Freeze pretrained weights: DISABLED (all trainable)")

    # 4. Check weight statistics (detect if still random)
    print("\n📊 Weight Statistics:")
    for comp in shared_components:
        component = getattr(model.model, comp, None)
        if component is None:
            continue

        params = list(component.parameters())
        if len(params) == 0:
            print(f"   {comp}: No parameters")
            continue

        first_param = params[0]
        param_mean = first_param.data.mean().item()
        param_std = first_param.data.std().item()
        param_abs_max = first_param.data.abs().max().item()

        print(f"   {comp}:")
        print(f"      First layer shape: {first_param.shape}")
        print(f"      Mean: {param_mean:.6f}")
        print(f"      Std:  {param_std:.6f}")
        print(f"      Max:  {param_abs_max:.6f}")

        # Heuristic check: pretrained weights usually have smaller variance
        if param_std < 0.01:
            print("      ⚠️  Very small std - might be zeros or need checking")
        elif param_std > 1.0:
            print("      ⚠️  Large std - might still be random initialized")

    # 5. Compare with MCES checkpoint (if provided)
    if mces_ckpt_path:
        print("\n🔍 Comparing with MCES checkpoint...")
        try:
            mces_checkpoint = th.load(mces_ckpt_path, map_location="cpu", weights_only=False)

            if "state_dict" in mces_checkpoint:
                mces_state_dict = mces_checkpoint["state_dict"]
            else:
                mces_state_dict = mces_checkpoint

            # Check if mol_embedder weights match
            test_key = None
            for key in mces_state_dict.keys():
                if "mol_embedder" in key:
                    clean_key = key.replace("model.", "")
                    if clean_key in dict(model.model.named_parameters()):
                        test_key = clean_key
                        break

            if test_key:
                mces_param = mces_state_dict[f"model.{test_key}"]
                fraggnn_param = dict(model.model.named_parameters())[test_key]

                if th.allclose(mces_param.cpu(), fraggnn_param.cpu(), atol=1e-6):
                    print(f"   ✅ Weights match! ({test_key})")
                else:
                    diff = (mces_param.cpu() - fraggnn_param.cpu()).abs().max()
                    print(f"   ❌ Weights DON'T match! ({test_key})")
                    print(f"      Max difference: {diff:.6f}")
            else:
                print("   ⚠️  No common keys found to compare")

        except Exception as e:
            print(f"   ❌ Failed to load MCES checkpoint: {e}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70 + "\n")


def verify_weight_loading_in_training_step(pl_module, batch_idx):
    """
    Quick check during training to verify pretrained weights.

    Call this in your training_step:

    ```python
    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and self.current_epoch == 0:
            verify_weight_loading_in_training_step(self, batch_idx)
        # ... rest of training step
    ```
    """

    if not hasattr(pl_module.hparams, "mces_pretrain"):
        return

    if not pl_module.hparams.mces_pretrain:
        return

    print(f"\n[Epoch {pl_module.current_epoch}, Batch {batch_idx}]")
    print("Checking pretrained weights...")

    # Check gradient flow
    mol_embedder = pl_module.model.mol_embedder
    first_param = next(mol_embedder.parameters())

    print("  mol_embedder.input_project.weight:")
    print(f"    requires_grad: {first_param.requires_grad}")
    print(f"    has grad: {first_param.grad is not None}")

    if pl_module.hparams.freeze_mces_weights and first_param.requires_grad:
        print("    ⚠️  WARNING: Weight should be frozen but has requires_grad=True!")


if __name__ == "__main__":
    print("To use this test:")
    print()
    print("1. After model creation:")
    print("   >>> test_mces_weight_loading(model, 'path/to/mces.ckpt')")
    print()
    print("2. In training_step (first batch):")
    print("   >>> verify_weight_loading_in_training_step(self, batch_idx)")


# ---------------------------------------------------------------------------
# Unit tests with lightweight dummy models
# ---------------------------------------------------------------------------


class _DummyParams:
    def __init__(self, mces_pretrain=True, freeze=False, ckpt_fp="ckpt.ckpt"):
        self.mces_pretrain = mces_pretrain
        self.freeze_mces_weights = freeze
        self.mces_pretrain_ckpt_fp = ckpt_fp


class _DummyInnerModel(th.nn.Module):
    def __init__(self, freeze: bool):
        super().__init__()
        self.mol_embedder = th.nn.Linear(4, 4, bias=False)
        self.mol_pool = th.nn.Linear(4, 2, bias=False)
        for p in self.parameters():
            p.requires_grad = not freeze


class _DummyPLModule:
    def __init__(self, hparams: _DummyParams):
        self.hparams = hparams
        self.model = _DummyInnerModel(freeze=hparams.freeze_mces_weights)
        self.current_epoch = 0


def _run_helper_and_capture(model, capsys):
    # Use the helper directly; it returns None but prints diagnostics
    test_mces_weight_loading(model)
    captured = capsys.readouterr()
    return captured.out


def test_helper_skips_when_disabled(capsys):
    hparams = _DummyParams(mces_pretrain=False)
    model = _DummyPLModule(hparams)
    out = _run_helper_and_capture(model, capsys)
    assert "pretraining is disabled" in out


def test_helper_reports_missing_component(capsys):
    hparams = _DummyParams(mces_pretrain=True)
    model = _DummyPLModule(hparams)
    # Remove mol_pool to trigger missing component branch
    delattr(model.model, "mol_pool")
    out = _run_helper_and_capture(model, capsys)
    assert "mol_pool" in out and "NOT found" in out


def test_helper_freeze_outputs_counts(capsys):
    hparams = _DummyParams(mces_pretrain=True, freeze=True)
    model = _DummyPLModule(hparams)
    out = _run_helper_and_capture(model, capsys)
    assert "Freeze pretrained weights: ENABLED" in out
    assert "Frozen params" in out


def test_verify_weight_loading_prints(capsys):
    hparams = _DummyParams(mces_pretrain=True, freeze=True)
    pl_module = _DummyPLModule(hparams)
    verify_weight_loading_in_training_step(pl_module, batch_idx=0)
    out = capsys.readouterr().out
    assert "Checking pretrained weights" in out
    assert "requires_grad: False" in out
