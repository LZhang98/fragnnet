import pytest
from frag_gnn.data_utils import *
import rdkit.Chem as Chem

def test_parse_formula():
    element_counts = parse_formula("NaHCO3")
    assert element_counts['Na'] == 1
    assert element_counts['H'] == 1
    assert element_counts['C'] == 1
    assert element_counts['O'] == 3
    assert len(element_counts) == 4

    element_counts = parse_formula("AgNO3")
    assert element_counts['Ag'] == 1
    assert element_counts['N'] == 1
    assert element_counts['O'] == 3
    assert len(element_counts) == 3

    element_counts = parse_formula("C2H6O+H")
    assert element_counts['C'] == 2
    assert element_counts['H'] == 6
    assert element_counts['O'] == 1
    assert len(element_counts) == 3

    element_counts = parse_formula("C2H6O-H")
    assert element_counts['C'] == 2
    assert element_counts['H'] == 6
    assert element_counts['O'] == 1
    assert len(element_counts) == 3

    
