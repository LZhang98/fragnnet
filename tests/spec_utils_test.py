import pytest
from frag_gnn.utils.spec_utils import *
import torch as th

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
    pass
