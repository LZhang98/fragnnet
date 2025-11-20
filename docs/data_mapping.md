N = total # nodes in batch
F = total # formulae in batch
B = total # spec in batch

all of these are aligned:

pred_formula_node_logprobs: P(n|f) of shape [N*F]
pred_joint_formula_idxs: formula idxs of shape [N*F]
pred_joint_batch_idxs: batch idxs of shape [N*F]s

all of these are aligned:

pred_formula_logprobs: P(f) of shape [F]
pred_formula_batch_idxs: batch idxs of shape [F]


step 1: select which spectrum you want (let's say batch_idx of k)

b_pred_formula_node_mask = (b_joint_batch_idxs == k)
b_pred_formula_node_logprobs = pred_formula_node_logprobs[b_pred_formula_node_mask]
b_pred_joint_formula_idxs = pred_joint_formula_idxs[b_pred_formula_node_mask]

b_pred_formula_mask = (pred_formula_batch_idxs == k)
b_pred_formula_logprobs = pred_formula_logprobs[b_pred_formula_mask]
b_pred_formula_formula_idxs = pred_formula_formula_idxs[b_pred_formula_mask]


step 2: figure out which formula you want (let's say formula_idx = j)
note: this assumes that formula_idx j is actually in the spec at batch_idx k

bf_pred_formula_node_mask = (b_pred_joint_formula_idxs == j)
...

bf_pred_formula_mask = (b_pred_formula_formula_idxs == j)
...


step 3: figure out which node you want (let's say i)
note: this assumes that node_idx i in in batch_idx k and has nonzero probability for formula_idx j

....


....