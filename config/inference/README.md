# Inference Output Columns

The `inference.output_subset` list controls which tensors are retained in the
prediction dictionary. Index columns are global within the complete inference
output after batches are concatenated.

| Column | Definition |
| --- | --- |
| `pred_mzs` | Predicted fragment mass-to-charge ratios. |
| `pred_logprobs` | Log probabilities associated with `pred_mzs`; exponentiating gives the predicted sparse intensities/probabilities. |
| `pred_batch_idxs` | Input-spectrum index for each predicted m/z and log-probability entry. |
| `pred_node_node_idxs` | Global fragment-DAG node index for each node-level prediction. |
| `pred_node_batch_idxs` | Input-spectrum index for each node-level prediction. |
| `pred_joint_node_idxs` | Global fragment-DAG node index for each joint node/formula prediction. |
| `pred_joint_batch_idxs` | Input-spectrum index for each joint node/formula prediction. |
| `pred_formula_logprobs` | Log probabilities for predicted fragment formulas. |
| `pred_formula_batch_idxs` | Input-spectrum index for each formula prediction. |
| `pred_formula_formula_idxs` | Global formula index for each formula-level prediction. |
| `pred_joint_formula_idxs` | Global formula index for each joint node/formula prediction. |
| `pred_nb_node_node_idxs` | Global neighbor/isotope fragment-node index for each neighbor node-level prediction. |
| `pred_nb_node_batch_idxs` | Input-spectrum index for each neighbor node-level prediction. |
| `pred_nb_node_node_batch_idxs` | Input-spectrum index for each neighbor node-to-node prediction. |
| `pred_nb_joint_node_idxs` | Global neighbor/isotope fragment-node index for each neighbor joint prediction. |
| `pred_nb_joint_batch_idxs` | Input-spectrum index for each neighbor joint prediction. |
| `pred_nb_node_node_node_idxs` | Global node index associated with each neighbor node-to-node prediction. |
| `pred_nb_joint_node_node_idxs` | Intended global node index for neighbor joint node-to-node predictions. This key is currently configured but is not emitted by the current `model.py` output dictionary. |
| `pred_nb_joint_formula_idxs` | Global formula index for each neighbor joint prediction. |

`batch_idxs` columns identify the input spectrum, while `node_idxs` and
`formula_idxs` identify entries in the concatenated fragment-node and formula
collections. The `nb_*` columns are only populated when neighbor/isotope
inference is enabled; otherwise they may be absent or contain `None` values.
