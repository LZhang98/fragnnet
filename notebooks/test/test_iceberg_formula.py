unique_ids = formula_vals["unique_id"]
true_batch_idxs = formula_vals["true_batch_idxs"]
pred_batch_idxs = formula_vals["pred_batch_idxs"]
true_mzs = formula_vals["true_mzs"]
true_logprobs = formula_vals["true_logprobs"]
pred_mzs = formula_vals["pred_mzs"]
pred_logprobs = formula_vals["pred_logprobs"]
spec_batch_idxs = th.arange(unique_ids.shape[0])
# pred_formula_logprobs = formula_vals["pred_formula_logprobs"]
pred_formula_batch_idxs = formula_vals["pred_formula_batch_idxs"]
pred_formula_mzs = formula_vals["pred_formula_mzs"]
pred_formula_vecs = formula_vals["pred_formula_vecs"]

m_ann_df_cp = m_ann_df.copy().set_index("group_id")

thresholds = [0.]

# ann_formulae, pred_formulae = [], []
coverages, peak_coverages, w_peak_coverages = [], [], []

# iterate over the group truth data
for i in tqdm(range(len(group_ids))):
# for i in tqdm(range(1)):
    i_group_id = group_ids[i]
    i_mol_id = mol_ids[i]
    if i_group_id not in m_ann_df_cp.index:
        continue
    i_mask = (unique_ids == i_group_id).flatten()
    i_batch_idx = spec_batch_idxs[i_mask].item()
    i_pred_peak_mask = (pred_batch_idxs == i_batch_idx)
    # i_pred_mzs = pred_mzs[i_pred_peak_mask]
    # i_pred_logprobs = pred_logprobs[i_pred_peak_mask]
    i_true_peak_mask = (true_batch_idxs == i_batch_idx)
    i_true_mzs = true_mzs[i_true_peak_mask]
    i_true_logprobs = true_logprobs[i_true_peak_mask]
    assert th.isclose(th.logsumexp(i_true_logprobs,dim=0),th.tensor(0.),rtol=0.,atol=1e-4), th.logsumexp(i_true_logprobs,dim=0)
    i_formula_mask = (pred_formula_batch_idxs == i_batch_idx).flatten()
    # i_pred_formula_logprobs = pred_formula_logprobs[i_formula_mask]
    # assert th.isclose(th.logsumexp(i_pred_formula_logprobs,dim=0),th.tensor(0.),rtol=0.,atol=1e-4), th.logsumexp(i_pred_formula_logprobs,dim=0)
    i_pred_formula_vecs = pred_formula_vecs[i_formula_mask]
    i_pred_formula_mzs = pred_formula_mzs[i_formula_mask]
    # adjust for precursor
    i_pred_formula_vecs[:,element_to_ind["H"]] += 1
    # convert to str
    i_pred_formulae = [
        composition_to_string(Composition(formula=vec_to_formula(i_pred_formula_vec.squeeze(0)))) for i_pred_formula_vec \
            in i_pred_formula_vecs.split(1, dim=0)
    ]
    i_pred_formulae = np.array(i_pred_formulae)
    assert i_pred_formulae.shape[0] == i_pred_formula_mzs.shape[0]
    # print(i_pred_formula_mzs[-5:])
    # print(i_pred_formulae[-5:])
    i_ann_formulae = m_ann_df_cp.loc[i_group_id,"formulae_dedup"]
    i_ann_formulae = [
        composition_to_string(Composition(formula=formula)) for formula in i_ann_formulae
    ]
    i_ann_formulae = np.array(i_ann_formulae)
    i_ann_mzs = th.tensor(m_ann_df_cp.loc[i_group_id,"mzs_dedup"])
    # print(i_ann_mzs)
    # print(i_ann_formulae)
    # match formulae within 10 ppm
    i_ann_mz_match_mask = calculate_match_mzs(
        i_ann_mzs, 
        i_true_mzs,
        tolerance=1e-5, # 10 ppm
        relative=True
    ) # A x T
    i_pred_mz_match_mask = calculate_match_mzs(
        i_pred_formula_mzs, 
        i_true_mzs,
        tolerance=1e-5, # 10 ppm
        relative=True
    ) # P x T
    # # print(i_pred_mz_match_mask.any(dim=0).float().mean())
    # i_both_formulae = np.intersect1d(i_ann_formulae,i_pred_formulae)
    # # print(i_both_formulae)
    # for i_both_formula in i_both_formulae:
    #     print(">>>", i_both_formula)
    #     print(i_ann_mzs[i_ann_formulae==i_both_formula])
    #     print(i_pred_formula_mzs[i_pred_formulae==i_both_formula])

    i_coverages, i_peak_coverages, i_w_peak_coverages = [], [], []
    for threshold in thresholds:
        # note: this is not the same shape as the i_formula_mask
        assert threshold == 0., threshold
        i_pt_formula_mask = th.ones_like(i_pred_formula_mzs,dtype=th.bool)
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

coverages = np.array(coverages)
peak_coverages = np.array(peak_coverages)
w_peak_coverages = np.array(w_peak_coverages)

for i, threshold in enumerate(thresholds):
    print(f">> Threshold = {threshold}")
    print(f"> Formula Coverage: {np.mean(coverages[:,i]):.4f} +/- {np.std(coverages[:,i]):.4f}") 
    print(f"> Peak Coverage: {np.mean(peak_coverages[:,i]):.4f} +/- {np.std(peak_coverages[:,i]):.4f}")
    print(f"> W Peak Coverage: {np.mean(w_peak_coverages[:,i]):.4f} +/- {np.std(w_peak_coverages[:,i]):.4f}")