unique_ids = formula_vals["unique_id"]
true_batch_idxs = formula_vals["true_batch_idxs"]
pred_batch_idxs = formula_vals["pred_batch_idxs"]
true_mzs = formula_vals["true_mzs"]
true_logprobs = formula_vals["true_logprobs"]
pred_mzs = formula_vals["pred_mzs"]
pred_logprobs = formula_vals["pred_logprobs"]
spec_batch_idxs = th.arange(unique_ids.shape[0])
pred_formula_logprobs = formula_vals["pred_formula_logprobs"]
pred_formula_batch_idxs = formula_vals["pred_formula_batch_idxs"]
pred_formula_formula_idxs = formula_vals["pred_formula_formula_idxs"]
formula_idx_cumsum = th.cumsum(th.cat(
    [
        th.zeros([1],dtype=th.long),
        th.unique(pred_formula_batch_idxs, return_counts=True)[1]
    ], dim=0), dim=0
)

m_ann_df_cp = m_ann_df.copy().set_index("group_id")

thresholds = [0., 1e-6, 1e-5, 0.001, 0.01]

# ann_formulae, pred_formulae = [], []
jsds = []
coverages, peak_coverages, w_peak_coverages = [], [], []

# iterate over the group truth data
# for i in tqdm(range(len(group_ids))):
for i in tqdm(range(300)):
# for i in tqdm(range(1)):
    i_group_id = group_ids[i]
    i_mol_id = mol_ids[i]
    if i_group_id not in m_ann_df_cp.index:
        continue
    i_frag_d = frag_d_d[i_mol_id]
    i_formula_idx_to_str = i_frag_d["idx_to_formula"]
    # pprint(i_formula_idx_to_str)
    i_mask = (unique_ids == i_group_id).flatten()
    i_batch_idx = spec_batch_idxs[i_mask].item()
    i_pred_peak_mask = (pred_batch_idxs == i_batch_idx)
    i_pred_mzs = pred_mzs[i_pred_peak_mask]
    # i_pred_logprobs = pred_logprobs[i_pred_peak_mask]
    i_true_peak_mask = (true_batch_idxs == i_batch_idx)
    i_true_mzs = true_mzs[i_true_peak_mask]
    i_true_logprobs = true_logprobs[i_true_peak_mask]
    assert th.isclose(th.logsumexp(i_true_logprobs,dim=0),th.tensor(0.),rtol=0.,atol=1e-4), th.logsumexp(i_true_logprobs,dim=0)
    
    i_formula_mask = (pred_formula_batch_idxs == i_batch_idx).flatten()
    i_pred_formula_logprobs = pred_formula_logprobs[i_formula_mask][1:]
    assert th.isclose(th.logsumexp(i_pred_formula_logprobs,dim=0),th.tensor(0.),rtol=0.,atol=1e-4), th.logsumexp(i_pred_formula_logprobs,dim=0)
    i_pred_formula_idxs = pred_formula_formula_idxs[i_formula_mask] - formula_idx_cumsum[i_batch_idx]
    assert i_pred_formula_idxs[0] == 0, i_pred_formula_idxs
    i_pred_formula_idxs = i_pred_formula_idxs[1:]
    # TODO: modify to account for prec_type properly
    i_pred_formulae = [
        composition_to_string(Composition(formula=i_formula_idx_to_str[idx.item()]) + Composition(formula="H")) 
        for idx in i_pred_formula_idxs
    ]
    i_pred_formulae = np.array(i_pred_formulae)
    i_ann_formulae = m_ann_df_cp.loc[i_group_id,"formulae_dedup"]
    i_ann_formulae = [
        composition_to_string(Composition(formula=formula)) for formula in i_ann_formulae
    ]
    i_ann_formulae = np.array(i_ann_formulae)
    i_ann_mzs = th.tensor(m_ann_df_cp.loc[i_group_id,"mzs_dedup"])

    # get idxs
    i_both_formulae = np.union1d(i_pred_formulae, i_ann_formulae)
    i_both_formula_idxs = np.arange(len(i_both_formulae))
    i_pred_formula_idxs = th.as_tensor(i_both_formula_idxs[np.isin(i_both_formulae, i_pred_formulae)])
    i_ann_formula_idxs = th.as_tensor(i_both_formula_idxs[np.isin(i_both_formulae, i_ann_formulae)])
    assert i_pred_formula_idxs.shape[0] == len(i_pred_formulae), (i_pred_formula_idxs.shape[0], len(i_pred_formulae))
    assert i_ann_formula_idxs.shape[0] == len(i_ann_formulae), (i_ann_formula_idxs.shape[0], len(i_ann_formulae))
    assert i_pred_formula_idxs.shape[0] == i_pred_formula_logprobs.shape[0], (i_pred_formula_idxs.shape[0], i_pred_formula_logprobs.shape[0])

    # calculate i_true_formula_logprobs
    # match formulae within 10 ppm
    i_ann_mz_match_mask = calculate_match_mzs(
        i_ann_mzs, 
        i_true_mzs,
        tolerance=1e-5, # 10 ppm
        relative=True
    ) # A x T

    i_ann_formula_logprobs = (i_true_logprobs.unsqueeze(0) - th.log(th.sum(i_ann_mz_match_mask, dim=1, keepdim=True)))
    i_ann_formula_logprobs[~i_ann_mz_match_mask] = LOG_ZERO(i_ann_formula_logprobs.dtype)
    i_ann_formula_logprobs = th.logsumexp(i_ann_formula_logprobs, dim=1)
    assert i_ann_formula_idxs.shape[0] == i_ann_formula_logprobs.shape[0], (i_ann_formula_idxs.shape[0], i_ann_formula_logprobs.shape[0])
    
    i_pred_mz_match_mask = calculate_match_mzs(
        i_pred_mzs, 
        i_true_mzs,
        tolerance=1e-5, # 10 ppm
        relative=True
    ) # P x T

    # calculate JSD
    i_formula_jsd = js_div(
        i_pred_formula_idxs,
        i_pred_formula_logprobs,
        i_ann_formula_idxs,
        i_ann_formula_logprobs
    )
    jsds.append(i_formula_jsd.item())

    i_coverages, i_peak_coverages, i_w_peak_coverages = [], [], []
    for threshold in thresholds:
        # note: this is not the same shape as the i_formula_mask
        if threshold == 0.:
            i_pt_formula_mask = th.ones_like(i_pred_formula_logprobs,dtype=th.bool)
        else:
            i_pt_formula_mask = (i_pred_formula_logprobs >= np.log(threshold))
        i_coverage = len(np.intersect1d(i_pred_formulae[i_pt_formula_mask.numpy()],i_ann_formulae)) / len(np.unique(i_ann_formulae))
        i_both_formula_match_mask = th.as_tensor(
            i_ann_formulae.reshape(-1,1) == i_pred_formulae[i_pt_formula_mask.numpy()].reshape(1,-1)
        ) # A x P
        i_ann_mz_match_mask_2 = (i_both_formula_match_mask.long() @ i_pred_mz_match_mask[i_pt_formula_mask].long()) > 0 # A x T
        i_peak_match = th.any(i_ann_mz_match_mask & i_ann_mz_match_mask_2, dim=0)
        i_peak_coverage = i_peak_match.float().sum() / th.any(i_ann_mz_match_mask, dim=0).sum()
        i_w_peak_coverage = (i_peak_match.float() * i_true_logprobs.exp()).sum() / i_true_logprobs.exp().sum()
        i_coverages.append(i_coverage)
        i_peak_coverages.append(i_peak_coverage.item())
        i_w_peak_coverages.append(i_w_peak_coverage.item())

    # pred_formulae.append(i_pred_formulae)
    # ann_formulae.append(i_ann_formulae)
    coverages.append(i_coverages)
    peak_coverages.append(i_peak_coverages)
    w_peak_coverages.append(i_w_peak_coverages)

jsds = np.array(jsds)
coverages = np.array(coverages)
peak_coverages = np.array(peak_coverages)
w_peak_coverages = np.array(w_peak_coverages)

print(f">> JSD: {np.mean(jsds):.4f} +/- {np.std(jsds):.4f}")

for i, threshold in enumerate(thresholds):
    print(f">> Threshold = {threshold}")
    print(f"> Formula Coverage: {np.mean(coverages[:,i]):.4f} +/- {np.std(coverages[:,i]):.4f}") 
    print(f"> Peak Coverage: {np.mean(peak_coverages[:,i]):.4f} +/- {np.std(peak_coverages[:,i]):.4f}")
    print(f"> W Peak Coverage: {np.mean(w_peak_coverages[:,i]):.4f} +/- {np.std(w_peak_coverages[:,i]):.4f}")