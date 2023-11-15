import pytest
from frag_gnn.utils.spec_utils import *
import torch as th
import numpy as np

def test_transform_nce_to_ace():
    # NCE of 40. a 1,000 m/z ion with a charge state of 1
    # ACE should be (40) x (1,000)/500 x 1 = 80 eV
    ace = transform_nce_to_ace(40,1000,1)
    assert ace == pytest.approx(80)

def test_transform_ace():
    # NCE of 40. a 1,000 m/z ion with a charge state of 1
    # ACE should be (40) x (1,000)/500 x 1 = 80 eV
    pass

def test_filter_func():

    mzs_np = np.array([0,1,2,3])
    ints_np  =  np.array([0.1,0.2,0.3,1.0])
    # nothing should happen
    f_mzs, f_ints = filter_func(mzs_np,ints_np,ints_thresh=-1, mz_max=4)
    assert np.array_equal(mzs_np,f_mzs)
    assert np.array_equal(ints_np,f_ints)

    # nothing tensor
    mzs_th = th.tensor(mzs_np)
    ints_th = th.tensor(ints_np)
    f_mzs, f_ints = filter_func(mzs_th,ints_th,ints_thresh=-1, mz_max=4)
    assert th.equal(mzs_th,f_mzs)
    assert th.equal(ints_th,f_ints)

    # this should remove 3 peaks   
    f_mzs, f_ints = filter_func(mzs_np,ints_np,ints_thresh=0.3, mz_max=4)
    assert f_mzs[0] == 3
    assert f_ints[0] == 1.0
    assert len(f_mzs) == 1
    assert len(f_ints) == 1

    f_mzs, f_ints = filter_func(mzs_th,ints_th,ints_thresh=0.3, mz_max=4)
    assert f_mzs[0] == 3
    assert f_ints[0] == 1.0
    assert len(f_mzs) == 1
    assert len(f_ints) == 1

def bin_func_test():
    mzs_th = th.tensor([0.1,0.201,0.305,0.306])
    ints_th = th.tensor([1,1,0.5,1])
    mz_max = 5.0
    mz_bin_res = 0.01
    return_index = False
    sum_ints = True
    bin_spec = bin_func(mzs_th,ints_th, mz_max, mz_bin_res, return_index, sum_ints)

    # we should have 500 bins
    assert len(bin_spec) == mz_max/ mz_bin_res
    # 0.1 is on  the edge, thus it should in the 11th bin not 10th
    assert bin_spec[10] == ints_th[0]
    assert bin_spec[20] == ints_th[1]
    # check if sum works
    assert bin_spec[30] == ints_th[2] + ints_th[3]

    sum_ints = False
    bin_spec = bin_func(mzs_th,ints_th, mz_max, mz_bin_res, return_index, sum_ints)
    # check if amax work
    assert bin_spec[30] == ints_th[3]

    # check if return_index works
    return_index = True
    bin_idx = bin_func(mzs_th,ints_th, mz_max, mz_bin_res, return_index, sum_ints)
    expected_bin_idx = th.tensor([10,20,30,30])
    assert th.equal(bin_idx, expected_bin_idx)

def batch_func_test():
    pass
    
def calculate_match_mzs_test():
    predicted_mzs = th.tensor([0.1,0.201,0.305,0.306])
    ground_truth_mzs = th.tensor([0.1,0.201,0.305,0.306,0.307])
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
    assert th.equal(macthes, expected_macthes)

    # check if abs tol works
    macthes = calculate_match_mzs(predicted_mzs, ground_truth_mzs, relative = False)

    expected_macthes = th.tensor([
        [True, False, False, False, False],
        [False, True, False, False, False],
        [False, False,  True,  False, False],
        [False, False,  False,  True,  False],
    ])
    # check if size is correct
    assert macthes.shape[0] == predicted_mzs.shape[0]
    assert macthes.shape[1] == ground_truth_mzs.shape[0]
    # check if matches are correct
    assert th.equal(macthes, expected_macthes)