import pytest
from frag_gnn.utils.spec_utils import *
import torch as th
import numpy as np


def test_transform_nce_to_ace():
    # NCE of 40. a 1,000 m/z ion with a charge state of 1
    # ACE should be (40) x (1,000)/500 x 1 = 80 eV
    ace = transform_nce_to_ace(40, 1000, 1)
    assert ace == pytest.approx(80)


def test_transform_ace():
    # NCE of 40. a 1,000 m/z ion with a charge state of 1
    # ACE should be (40) x (1,000)/500 x 1 = 80 eV
    pass


def test_filter_func():

    mzs_np = np.array([0, 1, 2, 3])
    ints_np = np.array([0.1, 0.2, 0.3, 1.0])
    # nothing should happen
    f_mzs, f_ints = filter_func(mzs_np, ints_np, ints_thresh=-1, mz_max=4)
    assert np.array_equal(mzs_np, f_mzs)
    assert np.array_equal(ints_np, f_ints)

    # nothing tensor
    mzs_th = th.tensor(mzs_np)
    ints_th = th.tensor(ints_np)
    f_mzs, f_ints = filter_func(mzs_th, ints_th, ints_thresh=-1, mz_max=4)
    th.testing.assert_close(mzs_th, f_mzs)
    th.testing.assert_close(ints_th, f_ints)

    # this should remove 3 peaks
    f_mzs, f_ints = filter_func(mzs_np, ints_np, ints_thresh=0.3, mz_max=4)
    assert f_mzs[0] == 3
    assert f_ints[0] == 1.0
    assert len(f_mzs) == 1
    assert len(f_ints) == 1

    f_mzs, f_ints = filter_func(mzs_th, ints_th, ints_thresh=0.3, mz_max=4)
    assert f_mzs[0] == 3
    assert f_ints[0] == 1.0
    assert len(f_mzs) == 1
    assert len(f_ints) == 1


def test_bin_func():
    mzs_th = th.tensor([0.1, 0.201, 0.305, 0.306])
    ints_th = th.tensor([1, 1, 0.5, 1])
    mz_max = 5.0
    mz_bin_res = 0.01
    return_index = False
    sum_ints = True
    bin_spec = bin_func(mzs_th, ints_th, mz_max,
                        mz_bin_res, return_index, sum_ints)

    # we should have 500 bins
    assert len(bin_spec) == mz_max / mz_bin_res
    # 0.1 is on  the edge, thus it should in the 11th bin not 10th
    assert bin_spec[10] == ints_th[0]
    assert bin_spec[20] == ints_th[1]
    # check if sum works
    assert bin_spec[30] == ints_th[2] + ints_th[3]

    sum_ints = False
    bin_spec = bin_func(mzs_th, ints_th, mz_max,
                        mz_bin_res, return_index, sum_ints)
    # check if amax work
    assert bin_spec[30] == ints_th[3]

    # check if return_index works
    return_index = True
    bin_idx = bin_func(mzs_th, ints_th, mz_max,
                       mz_bin_res, return_index, sum_ints)
    expected_bin_idx = th.tensor([10, 20, 30, 30])
    th.testing.assert_close(bin_idx, expected_bin_idx)


def test_batch_func():
    pass


def test_calculate_match_mzs():
    predicted_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    ground_truth_mzs = th.tensor([0.1, 0.201, 0.305, 0.306, 0.307])
    macthes = calculate_match_mzs(predicted_mzs, ground_truth_mzs)

    expected_macthes = th.tensor([
        [True, False, False, False, False],
        [False, True, False, False, False],
        [False, False,  True,  True, False],
        [False, False,  True,  True,  True],
    ])
    # check if size is correct
    assert macthes.shape[0] == predicted_mzs.shape[0]
    assert macthes.shape[1] == ground_truth_mzs.shape[0]
    # check if matches are correct
    th.testing.assert_close(macthes, expected_macthes)

    # check if abs tol works
    macthes = calculate_match_mzs(
        predicted_mzs, ground_truth_mzs, relative=False)

    expected_macthes = th.tensor([
        [True, False, False, False, False],
        [False, True, False, False, False],
        [False, False,  True, False, False],
        [False, False,  False, True,  False],
    ])
    # check if size is correct
    assert macthes.shape[0] == predicted_mzs.shape[0]
    assert macthes.shape[1] == ground_truth_mzs.shape[0]
    # check if matches are correct
    th.testing.assert_close(macthes, expected_macthes)

def test_calculate_recall():
    p_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    p_batch_idxs = th.tensor([0, 0, 0, 0])
    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_batch_idxs = th.tensor([0, 0, 0, 0])
    recall = calculate_recall(t_mzs, t_batch_idxs, p_mzs, p_batch_idxs)

    # this should be 1.0
    th.testing.assert_close(recall, th.tensor([1.0]))

    # this has 4 matching, 10 items, dice = 0.8, precision = 4/6, recall = 4/4 
    p_mzs = th.tensor([0.01, 0.1, 0.201, 0.305, 0.306, 0.38])
    p_batch_idxs = th.tensor([0, 0, 0, 0, 0, 0])
    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_batch_idxs = th.tensor([0, 0, 0, 0])
    recall = calculate_recall(t_mzs, t_batch_idxs, p_mzs, p_batch_idxs)
    th.testing.assert_close(recall, th.tensor([1.0]))

def test_calculate_weighted_recall():
    p_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    p_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    p_batch_idxs = th.tensor([0, 0, 0, 0])
    
    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    t_batch_idxs = th.tensor([0, 0, 0, 0])

    w_recall = calculate_weighted_recall(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs)

    # this should be 1.0
    th.testing.assert_close(w_recall, th.tensor([1.0]))

    # this has 4 matching, 10 items, dice = 0.8, precision = 4/6, recall = 4/4 
    p_mzs = th.tensor([0.01, 0.1, 0.201, 0.305, 0.306, 0.38])
    p_ints = th.tensor([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])
    p_batch_idxs = th.tensor([0, 0, 0, 0, 0, 0])

    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    t_batch_idxs = th.tensor([0, 0, 0, 0])

    w_recall = calculate_weighted_recall(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs)
    th.testing.assert_close(w_recall, th.tensor([1.0]))

    # do it backward
    w_recall = calculate_weighted_recall(p_mzs, p_ints, p_batch_idxs, t_mzs, t_ints, t_batch_idxs)
    th.testing.assert_close(w_recall, th.tensor([0.7]))
    
def test_calculate_precision():
    p_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    p_batch_idxs = th.tensor([0, 0, 0, 0])
    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_batch_idxs = th.tensor([0, 0, 0, 0])
    precision = calculate_precision(t_mzs, t_batch_idxs, p_mzs, p_batch_idxs)

    # this should be 1.0
    th.testing.assert_close(precision, th.tensor([1.0]))

    # this has 4 matching, 10 items, dice = 0.8, precision = 4/6
    p_mzs = th.tensor([0.01, 0.1, 0.201, 0.305, 0.306, 0.38])
    p_batch_idxs = th.tensor([0, 0, 0, 0, 0, 0])
    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_batch_idxs = th.tensor([0, 0, 0, 0])
    precision = calculate_precision(t_mzs, t_batch_idxs, p_mzs, p_batch_idxs)
    th.testing.assert_close(precision, th.tensor([4/6]))

def test_calculate_weighted_precision():
    p_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    p_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    p_batch_idxs = th.tensor([0, 0, 0, 0])
    
    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    t_batch_idxs = th.tensor([0, 0, 0, 0])

    w_precision = calculate_weighted_precision(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs)

    # this should be 1.0
    th.testing.assert_close(w_precision, th.tensor([1.0]))

    # this has 4 matching, 10 items, dice = 0.8, precision = 4/6, recall = 4/4 
    p_mzs = th.tensor([0.01, 0.1, 0.201, 0.305, 0.306, 0.38])
    p_ints = th.tensor([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])
    p_batch_idxs = th.tensor([0, 0, 0, 0, 0, 0])

    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    t_batch_idxs = th.tensor([0, 0, 0, 0])

    w_precision = calculate_weighted_precision(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs)
    th.testing.assert_close(w_precision, th.tensor([0.7]))

    # do it backward
    w_precision = calculate_weighted_precision(p_mzs, p_ints, p_batch_idxs, t_mzs, t_ints, t_batch_idxs)
    th.testing.assert_close(w_precision, th.tensor([1.0]))

def test_calculate_dice():

    # this should be 1.0
    p_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    p_batch_idxs = th.tensor([0, 0, 0, 0])
    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_batch_idxs = th.tensor([0, 0, 0, 0])

    dice = calculate_dice(t_mzs, t_batch_idxs, p_mzs, p_batch_idxs)
    th.testing.assert_close(dice, th.tensor([1.0]))

    # this has 4 matching, 10 items, dice = 0.8
    p_mzs = th.tensor([0.01, 0.1, 0.201, 0.305, 0.306, 0.38])
    p_batch_idxs = th.tensor([0, 0, 0, 0, 0, 0])
    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_batch_idxs = th.tensor([0, 0, 0, 0])
    dice = calculate_dice(t_mzs, t_batch_idxs, p_mzs, p_batch_idxs)

    th.testing.assert_close(dice, th.tensor([0.8]))


def test_calculate_ndcg_hungarian():

    p_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    p_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    p_batch_idxs = th.tensor([0, 0, 0, 0])

    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    t_batch_idxs = th.tensor([0, 0, 0, 0])
    calculate_ndcg_hungarian(t_mzs, t_ints, t_batch_idxs,
                             p_mzs, p_ints, p_batch_idxs)

def test_calculate_cosine():

    p_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    p_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    p_batch_idxs = th.tensor([0, 0, 0, 0])

    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    t_batch_idxs = th.tensor([0, 0, 0, 0])

    # these are the same
    cos = calculate_sparse_cosine_hungarian(
        t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs)
    th.testing.assert_close(cos, th.tensor([1.0]))

    # eqauls to do bin size  0.0001
    cos = calculate_sparse_cosine_similarity(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs, mz_bin_res= 0.0001)
    th.testing.assert_close(cos, th.tensor([1.0]))

    cos = calculate_cosine_similarity(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs, mz_bin_res= 0.0001)
    th.testing.assert_close(cos, th.tensor([1.0]))

    cos = calculate_optimal_cosine_similarity(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs, mz_bin_res= 0.0001)
    th.testing.assert_close(cos, th.tensor([1.0]))

    # this has 4 matching, 10 items, dice = 0.8, precision = 4/6, recall = 4/4 
    p_mzs = th.tensor([0.01, 0.1, 0.201, 0.305, 0.306, 0.38])
    p_ints = th.tensor([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])
    p_batch_idxs = th.tensor([0, 0, 0, 0, 0, 0])

    t_mzs = th.tensor([0.1, 0.201, 0.305, 0.306])
    t_ints = th.tensor([0.1, 0.2, 0.3, 0.4])
    t_batch_idxs = th.tensor([0, 0, 0, 0])

    cos = calculate_sparse_cosine_hungarian(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs)
    th.testing.assert_close(cos, th.tensor([0.61237]))

    cos = calculate_sparse_cosine_similarity(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs, mz_bin_res= 0.0001)
    th.testing.assert_close(cos, th.tensor([0.61237]))

    cos = calculate_cosine_similarity(t_mzs, t_ints, t_batch_idxs, p_mzs, p_ints, p_batch_idxs, mz_bin_res= 0.0001)
    th.testing.assert_close(cos, th.tensor([0.61237]))
    
def round_aggregate_peaks_test():
    p_mzs = th.tensor([0.1, 0.201, 0.306, 0.307])
    p_ints = th.tensor([1, 1, 0.5, 1])
    p_batch_idxs = th.tensor([0, 0, 0, 0])

    # this should do nothing
    decimals = 3
    sum_ints = True
    r_mzs, r_ints, r_batch_idxs = round_aggregate_peaks(
        p_mzs, p_ints, p_batch_idxs, decimals, sum_ints)
    th.testing.assert_close(th.round(p_mzs, decimals=decimals), r_mzs)
    th.testing.assert_close(p_ints, r_ints)
    th.testing.assert_close(p_batch_idxs, r_batch_idxs)

    # this should do nothing
    except_r_mzs = th.tensor([0.1, 0.2, 0.31])
    except_r_ints = th.tensor([1, 1, 1.5])
    except_r_batch_idxs = th.tensor([0, 0, 0])

    decimals = 2
    sum_ints = True
    r_mzs, r_ints, r_batch_idxs = round_aggregate_peaks(
        p_mzs, p_ints, p_batch_idxs, decimals, sum_ints)
    th.testing.assert_close(except_r_mzs, r_mzs)
    th.testing.assert_close(except_r_ints, r_ints)
    th.testing.assert_close(except_r_batch_idxs, r_batch_idxs)

    decimals = 2
    sum_ints = False
    except_r_ints = th.tensor([1, 1, 1.0])
    r_mzs, r_ints, r_batch_idxs = round_aggregate_peaks(
        p_mzs, p_ints, p_batch_idxs, decimals, sum_ints)
    th.testing.assert_close(except_r_mzs, r_mzs)
    th.testing.assert_close(except_r_ints, r_ints)
    th.testing.assert_close(except_r_batch_idxs, r_batch_idxs)