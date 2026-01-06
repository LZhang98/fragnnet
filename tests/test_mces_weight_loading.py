"""
This helper was moved to `tests/helpers/mces_weight_loading.py`.
The original test filename was retained but now contains only this message
so pytest won't collect a runnable test from it.
"""


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
    
    if not hasattr(pl_module.hparams, 'mces_pretrain'):
        return
    
    if not pl_module.hparams.mces_pretrain:
        return
    
    print(f"\n[Epoch {pl_module.current_epoch}, Batch {batch_idx}]")
    print("Checking pretrained weights...")
    
    # Check gradient flow
    mol_embedder = pl_module.model.mol_embedder
    first_param = next(mol_embedder.parameters())
    
    print(f"  mol_embedder.input_project.weight:")
    print(f"    requires_grad: {first_param.requires_grad}")
    print(f"    has grad: {first_param.grad is not None}")
    
    if pl_module.hparams.freeze_mces_weights and first_param.requires_grad:
        print(f"    ⚠️  WARNING: Weight should be frozen but has requires_grad=True!")


if __name__ == "__main__":
    print("To use this test:")
    print()
    print("1. After model creation:")
    print("   >>> test_mces_weight_loading(model, 'path/to/mces.ckpt')")
    print()
    print("2. In training_step (first batch):")
    print("   >>> verify_weight_loading_in_training_step(self, batch_idx)")
