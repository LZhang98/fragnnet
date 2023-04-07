import rdkit
import rdkit.Chem as Chem
import itertools
import numpy as np
import scipy
from collections import Counter

from frag_gnn.data_utils import parse_formula
from typing import List, Tuple, Dict

PERIODIC_TABLE = Chem.GetPeriodicTable()

ELEMENT_TO_MASS = {
    "H": PERIODIC_TABLE.GetMostCommonIsotopeMass("H"), #1.007825031898,
    "C": PERIODIC_TABLE.GetMostCommonIsotopeMass("C"), #12.0,
    "N": PERIODIC_TABLE.GetMostCommonIsotopeMass("N"), #14.003074004251,
    "O": PERIODIC_TABLE.GetMostCommonIsotopeMass("O"), #15.994914619257,
    "F": PERIODIC_TABLE.GetMostCommonIsotopeMass("F"), #18.998403162067,
    "Cl": PERIODIC_TABLE.GetMostCommonIsotopeMass("Cl"), #34.96885269,
    "P": PERIODIC_TABLE.GetMostCommonIsotopeMass("P"), #30.9737619986,
    "S": PERIODIC_TABLE.GetMostCommonIsotopeMass("S"), #31.9720711744 
}

ELECTRON_MASS = 0.00054858

PREC_TYPE_TO_MASS_DIFF = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
}


def enumerate_multinomial(N:int, p:List[float], cutoff_thold:float=0.0001):
    """
    Enumerate the state space of rolling a die with |p| faces
    N times
    
    returns: [(prob_1, (tuple_of_counts)), 
               (prob_2, (tuple_of_counts)), 
               ...]
    
    cutoff_thold: Do not return points in the state space 
    with prob less than cutoff_thold
    """
    
    K = len(p)
    counts = []
    for s in itertools.combinations_with_replacement(range(K), N):
        count = np.zeros(K, dtype=np.int32)
        for i in s:
            count[i] +=1
        counts.append(count)
    counts = np.array(counts)


    p = p / np.sum(p)
    rv = scipy.stats.multinomial(N, p)

    probs = rv.pmf(counts)
    sort_idx = np.argsort(probs)[::-1]

    probs_sorted = probs[sort_idx]
    thold_idx = np.argwhere(np.cumsum(probs_sorted) >= 1-cutoff_thold).flatten()[0] 
    thold_idx += 1
    
    
    return list(zip(probs_sorted[:thold_idx], 
                    [tuple(a) for a in counts[sort_idx][:thold_idx]]))


def get_isotopes(atomicno:int, pct_thold:float = 0.01) -> List[Tuple[int,float,float]]: 
    """  
    Get all isotopes and their masses for atomic number atomicno

    Args:
        atomicno (int): atomic number
        pct_thold (float, optional): probility theorshold. Defaults to 0.01.

    Returns:
        List[Tuple[int,float,float]]:  [(int_mass, prob, exact_mass),...] for each isotope up to 4
    """

    mass_probs = []
    pt = Chem.GetPeriodicTable()
    for m in range(atomicno, 4*atomicno):
        pct = pt.GetAbundanceForIsotope(atomicno, m)
        if pct >= pct_thold:
            exact_mass = pt.GetMassForIsotope(atomicno, m)
            mass_probs.append((m, pct/100, exact_mass))
    return sorted(mass_probs, key=lambda r: -r[1])


def get_peaks_for_element(element:str, num_atoms:int) -> List:
    """  
    Get the peaks for a formula with num_atoms of atomic no
    Args:
        element (str): element symbol
        num_atoms (int): num of atoms

    Returns:
        _type_: _description_
    """
    atomicno = Chem.GetPeriodicTable().GetAtomicNumber(element)

    isotopes = get_isotopes(atomicno)
    probs = [i[1] for i in isotopes]

    comb_probs = enumerate_multinomial(num_atoms, probs, 0.001)

    exact_masses = [np.sum([isotopes[a_i][2]*a for a_i, a in enumerate(m)]) for p, m in comb_probs]

    peaks_masses_probs = [(comb_probs[i][0], m) for i, m in enumerate(exact_masses)] 
    return peaks_masses_probs


def get_peaks_for_formula(element_counts:Dict[str,int],threshold:float=0.001) -> Tuple[List[float],List[float]]:
    """for a given element configs, return masses with all possible isotopic configus

    Args:
        element_counts (Dict): element configs eg:  {"C":6,"H":7,"O":6,"N":0}
        threshold (float, optional): _description_. Defaults to 0.001.

    Returns:
        Tuple[List[float],List[float]]: ([mass,mass,mass,mass],[prob,prob,prob,prb])
    """

    peaks_masses_probs = {}
    for k,v in element_counts.items():
        if v > 0:
            peaks_masses_probs[k] = get_peaks_for_element(k,v)
    assert len(peaks_masses_probs) > 0, peaks_masses_probs
    probs = [[item[0] for item in v] for v in peaks_masses_probs.values()]
    masses = [[item[1] for item in v] for v in peaks_masses_probs.values()]
    all_probs = [np.prod(tup) for tup in itertools.product(*probs)]
    all_masses = [np.sum(tup) for tup in itertools.product(*masses)]
    keep_masses, keep_probs = [], []
    for mass, prob in zip(all_masses, all_probs):
        if prob >= threshold:
            keep_masses.append(mass)
            keep_probs.append(prob)
    return keep_masses, keep_probs


def formula_to_peak_mzs(formula,prec_type,isotopes=True,return_map=True):
    """_summary_

    Args:
        formula (_type_): _description_
        prec_type (_type_): _description_
        isotopes (bool, optional): _description_. Defaults to True.
        return_map (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    element_counts = parse_formula(formula)
    mass_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    if not isotopes:
        peak_mz = 0.
        for k,v in element_counts.items():
            peak_mz += v*ELEMENT_TO_MASS.get(k,0.)
        peak_mz += mass_diff
        peak_mzs = [peak_mz]
    else:
        peak_mzs = get_peaks_for_formula(element_counts)[0]
        peak_mzs = [peak_mz+mass_diff for peak_mz in peak_mzs]
    if return_map:
        return {peak_mz:formula for peak_mz in peak_mzs}
    else:
        return peak_mzs


if __name__ == "__main__":

    # Test Carbon
    for i in [1,2,10,100,1000]:
        element_counts = {"C":i}
        print(element_counts, get_peaks_for_formula(element_counts))

    # Test Br
    for i in range(1,4):
        element_counts = {"Br":i}
        print(element_counts, get_peaks_for_formula(element_counts))
    
    # Test Clorine
    for i in range(1,4):
        element_counts = {"Cl":i}
        print(element_counts, get_peaks_for_formula(element_counts)) 
        
    element_counts = {"C":6,"H":7,"O":6,"N":0}
    print(get_peaks_for_formula(element_counts))
