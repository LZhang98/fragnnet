import pytest
from frag_gnn.formula_utils import *
import rdkit.Chem as Chem

def test_get_peaks_for_element():

    PERIODIC_TABLE = Chem.GetPeriodicTable()
    #for i in [1,2,10,100,1000]:
    element_counts = {"C":1}
    isotopes = False
    pff = get_peaks_for_element('C',1)
    #print(pff)
    #assert pff[0][0] == 12.00000000
    #assert pff[0][1] == 13.00335484
    #assert pff[0][2] == 14.0032420


def test_get_isotopes():
    #test H
    results = get_isotopes(1, 0.0)
    assert results[0] == pytest.approx((1, 0.999885, 1.007825032))
    assert results[1] == pytest.approx((2, 0.000115, 2.014101778))
    assert results[2] == pytest.approx((3, 0.0, 3.016049278))

    #test C
    results = get_isotopes(6, 0.0001)
    # C12
    assert results[0] == pytest.approx((12, 0.9893, 12.0))
    # C13
    assert results[1] == pytest.approx((13, 0.0107, 13.00335484))
    # C14
    # assert results[2] == (14, 0.0, 14.00324199)

    #test Br
    results = get_isotopes(35, 0.0001)
    # Br79
    assert results[0] == pytest.approx((79, 0.5069, 78.9183371))
    # Br81
    assert results[1] == pytest.approx((81, 0.4931, 80.9162906))

    #test Cl
    results = get_isotopes(17, 0.0001)
    # Cl 35
    assert results[0] == pytest.approx((35, 0.7578, 34.96885268))
    # Cl 37
    assert results[1] == pytest.approx((37, 0.2422, 36.96590259))


def test_get_peaks_for_element():

    #test H
    results = get_peaks_for_element('H', 1)
    assert results[0] == pytest.approx((0.999885, 1.007825032))
    
    results = get_peaks_for_element('H', 5)
    assert results[0] == pytest.approx((0.999885 ** 5, 5.039125159999999))

    #test C
    results = get_peaks_for_element('C', 1)
    # C12
    assert results[0] == pytest.approx((0.9893000000000001, 12.0))
    # C13
    assert results[1] == pytest.approx((0.0107, 13.00335484))

    # test 2 Carbons
    results = get_peaks_for_element('C', 2)
    # C12 + C12
    assert results[0] == pytest.approx((0.9787144900000002, 24.0))
    # C13 + C13
    assert results[1] == pytest.approx((0.021171019999999856, 25.00335484))

def test_get_peaks_for_formula():
    # Carbon
    pff = get_peaks_for_formula({"C":1})
    assert pff[0] == pytest.approx((12, 13.00335484)) 
    assert pff[1] == pytest.approx((0.9893, 0.0107))
    
    # 1C + 1H
    pff = get_peaks_for_formula({"C":1, "H":1})
    assert pff[0] == pytest.approx((12 + 1.007825032, 13.00335484 + 1.007825032)) 
    assert pff[1] == pytest.approx((0.98918623050, 0.010698769))
    