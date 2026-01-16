import copy
import math
import torch as th
import torch.nn as nn

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_lr_scheduler(
    optimizer, decay_rate: float, decay_steps: int = 5000, warmup_steps: int = 1000
):
    """build_lr_scheduler.

    Args:
        optimizer:
        decay_rate (float): decay_rate
        decay_steps (int): decay_steps
        warmup_steps (int): warmup_steps
    """

    def lr_lambda(step):
        if step >= warmup_steps:
            # Adjust
            step = step - warmup_steps
            rate = decay_rate ** (step // decay_steps)
        else:
            rate = 1 - math.exp(-step / warmup_steps)
        return rate

    scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

def nan_forward_hook(self, input, output):
    if isinstance(output, tuple):
        outputs = list(output)
    elif isinstance(output, dict):
        outputs = list(output.values())
    else:
        outputs = [output]
    for i, val in enumerate(outputs):
        nan_mask = th.isnan(val)
        if nan_mask.any():
            print(">> In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", val[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def nan_backward_hook(self, grad_input, grad_output):
    for i, val in enumerate(grad_input):
        if val is None:
            continue
        nan_mask = th.isnan(val)
        if nan_mask.any():
            print(">> In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in grad_input {i} at indices: ", nan_mask.nonzero(), "where:", val[nan_mask.nonzero()[:, 0].unique(sorted=True)])
    for i, val in enumerate(grad_output):
        if val is None:
            continue
        nan_mask = th.isnan(val)
        if nan_mask.any():
            print(">> In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in grad_output {i} at indices: ", nan_mask.nonzero(), "where:", val[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def decompile_jit_ckpt(ckpt):
    """
    convert compiled ckpt to normal version and return a patched checkpoint
    Note this is a in place change
    """
 
    # inspired by
    # https://github.com/pytorch/pytorch/issues/101107#issuecomment-1801128683
    #print(f"> src: {path}")

    if ckpt['hyper_parameters']['compile'] == False:
        return ckpt

    in_state_dict = ckpt["state_dict"]
    pairings = [
        (src_key,  src_key.replace('._orig_mod',''))
        for src_key in in_state_dict.keys()
    ]
    #if all(src_key == dest_key for src_key, dest_key in pairings):
    #   return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        if src_key != dest_key:
            print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]

    ckpt["state_dict"] = out_state_dict
    ckpt['hyper_parameters']['compile'] = False
    return ckpt

def is_ckpt_compiled(ckpt) -> bool:
    state_dict = ckpt["state_dict"]
    for key in state_dict.keys():
        if '_orig_mod' in key:
            return True
    return False

def get_optimizer_class(optimizer_name):
    if optimizer_name == "adam":
        optimizer_cls = th.optim.Adam
    elif optimizer_name == "adamw":
        optimizer_cls = th.optim.AdamW
    elif optimizer_name == "sgd":
        optimizer_cls = th.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")
    return optimizer_cls