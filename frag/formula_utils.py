import rdkit
import rdkit.Chem as Chem
import itertools
import numpy as np
import scipy


def enumerate_multinomial(N, p, cutoff_thold=0.0001):
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
    s = list()
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


def get_isotopes(atomicno, pct_thold = 0.01):
    """
    Get all isotopes and their masses for atomic
    number atomicno
    
    returns [(int_mass, prob, exact_mass),...]
    for each isotope
    
    """
    mass_probs = []
    pt = Chem.GetPeriodicTable()
    for m in range(atomicno, 4*atomicno):
        pct = pt.GetAbundanceForIsotope(atomicno, m)
        if pct >= pct_thold:
            exact_mass = pt.GetMassForIsotope(atomicno, m)
            mass_probs.append((m, pct/100, exact_mass))
    return sorted(mass_probs, key=lambda r: -r[1])


def get_peaks_for_element(element, num_atoms):
    """
    Get the peaks for a formula with num_atoms of atomicno
    
    returns list of [(prob, mass)]
    
    """
    atomicno = Chem.GetPeriodicTable().GetAtomicNumber(element)

    isotopes = get_isotopes(atomicno)
    probs = [i[1] for i in isotopes]

    comb_probs = enumerate_multinomial(num_atoms, probs, 0.001)

    exact_masses = [np.sum([isotopes[a_i][2]*a for a_i, a in enumerate(m)]) for p, m in comb_probs]

    peaks_masses_probs = [(comb_probs[i][0], m) for i, m in enumerate(exact_masses)] 
    return peaks_masses_probs


def get_peaks_for_formula(element_counts,threshold=0.001):

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

if __name__ == "__main__":

    element_counts = {"C":6,"H":7,"O":6,"N":0}
    print(get_peaks_for_formula(element_counts))