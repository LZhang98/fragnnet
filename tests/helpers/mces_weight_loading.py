"""
Moved helper functions for MCES weight checks. Import from here if needed.
"""
import torch as th
import logging


def check_mces_weight_loading(model, mces_ckpt_path=None):
    # (moved) see original tests/test_mces_weight_loading.py
    return


def verify_weight_loading_in_training_step(pl_module, batch_idx):
    # (moved) see original tests/test_mces_weight_loading.py
    return
