from typing import Dict
import numpy as np
from spectral_entropy import ms_distance, math_distance
from spectral_entropy.tools import match_peaks_in_spectra

def get_cfm_dice(ref_spectrum:np.array, predicted_spectrum:np.array, ppm_tol:float = 10, abs_tol:float = 0.01) -> float:
    """ Method compute cfm dice between two spectrum
    Args:
        ref_spectrum : reference spectrum
        predicted_spectrum : predicted spectrum
        verbose (bool, optional): flag for verbose. Defaults to False.
        abs_tol (_type_, optional): _description_. Defaults to None.
        ppm_tol (_type_, optional): _description_. Defaults to None.
    Returns:
        float: dice
    """
    matched_peaks_count = len(match_peaks_in_spectra(spec_a=ref_spectrum, spec_b=predicted_spectrum,
                                            ms2_ppm=ppm_tol, ms2_da=abs_tol))
    total_peak_count = len(ref_spectrum) + len(predicted_spectrum)

    if total_peak_count > 0:
        dice = 2.0 * matched_peaks_count / total_peak_count
    else:
        dice = 0.0
    return dice


def get_dot_product(ref_spectrum:np.array, predicted_spectrum:np.array, ppm_tol:float = 10, abs_tol:float = 0.01) -> float:
    """ Method compute dot product between two spectrum
        If both ppm_tol and abs_tol is defined, abs_tol will be used.
    Args:
        ref_spectrum: reference spectrum
        predicted_spectrum : predicted spectrum
        verbose (bool, optional): flag for verbose. Defaults to False.
        abs_tol (_type_, optional): _description_. Defaults to None.
        ppm_tol (_type_, optional): _description_. Defaults to None.
    Returns:
        float: dot product 
    """
    spec_matched = match_peaks_in_spectra(spec_a=ref_spectrum, spec_b=predicted_spectrum,
                                            ms2_ppm=ppm_tol, ms2_da=abs_tol)

    return math_distance.dot_product_distance(spec_matched[:, 1], spec_matched[:, 2])


def get_stein_dot_product(ref_spectrum:np.array, predicted_spectrum:np.array, ppm_tol:float = 10, abs_tol:float = 0.01) -> float:
    """ Method compute stein dot product between two spectrum
        If both ppm_tol and abs_tol is defined, abs_tol will be used.
    Args:
        ref_spectrum (_type_): _description_
        predicted_spectrum (_type_): _description_
        abs_tol (_type_, optional): _description_. Defaults to None.
        ppm_tol (_type_, optional): _description_. Defaults to None.

    Returns:
        float: _description_
    """
    return ms_distance.weighted_dot_product_distance(spec_a=ref_spectrum, spec_b=predicted_spectrum,
                                            ms2_ppm=ppm_tol, ms2_da=abs_tol)



def get_cfm_recall(ref_spectrum:np.array, predicted_spectrum:np.array, ppm_tol:float = 10, abs_tol:float = 0.01) -> float:
    """ Method compute recall between two spectrum
        If both ppm_tol and abs_tol is defined, abs_tol will be used.
    Args:
        ref_spectrum : reference spectrum
        predicted_spectrum : predicted spectrum
        abs_tol (_type_, optional): _description_. Defaults to None.
        ppm_tol (_type_, optional): _description_. Defaults to None.
    Returns:
        float: recall
    """
    
    matched_peaks_count = len(match_peaks_in_spectra(spec_a=ref_spectrum, spec_b=predicted_spectrum,
                                        ms2_ppm=ppm_tol, ms2_da=abs_tol))
    ref_peak_count = len(ref_spectrum)
    recall = matched_peaks_count / ref_peak_count if ref_peak_count != 0 else 0.0

    return recall


def get_cfm_precision(ref_spectrum:np.array, predicted_spectrum:np.array, ppm_tol:float = 10, abs_tol:float = 0.01) -> float:
    """ Method compute precision  between two spectrum
        If both ppm_tol and abs_tol is defined, abs_tol will be used.
    Args:
        ref_spectrum : reference spectrum
        predicted_spectrum : predicted spectrum
        abs_tol (_type_, optional): _description_. Defaults to None.
        ppm_tol (_type_, optional): _description_. Defaults to None.
    Returns:
        float: precision
    """
    matched_peaks_count = len(match_peaks_in_spectra(spec_a=ref_spectrum, spec_b=predicted_spectrum,
                                        ms2_ppm=ppm_tol, ms2_da=abs_tol))
    pred_peak_count =  len(predicted_spectrum)
    precision = matched_peaks_count / pred_peak_count if pred_peak_count != 0 else 0.0
    return precision

    
def get_metrics(ref_spectrum:np.array, predicted_spectrum:np.array, ppm_tol:float = 10, abs_tol:float = 0.01) -> Dict[str, float]:
    """ Method to compute 'Dice' 'Dot Product', 'Precision':, 'Recall' between two spectrum
        If both ppm_tol and abs_tol is defined, abs_tol will be used.

    Args:
        ref_spectrum (_type_): _description_
        predicted_spectrum (_type_): _description_
        ppm_tol (float, optional): _description_. Defaults to 10.
        abs_tol (float, optional): _description_. Defaults to 0.01.

    Returns:
        Dict[str, float]: return dict for 'Dice': 'Dot Product', 'Precision', 'Recall'
    """
 
    precision = get_cfm_precision(ref_spectrum, predicted_spectrum, ppm_tol , abs_tol)
    recall = get_cfm_recall(ref_spectrum, predicted_spectrum, ppm_tol , abs_tol)
    dice = get_cfm_dice(ref_spectrum, predicted_spectrum, ppm_tol , abs_tol)
    dot_product = get_dot_product(ref_spectrum, predicted_spectrum, ppm_tol, abs_tol)


    metrics = {'Dice': dice, 'Dot Product': dot_product, 'Precision': precision, 'Recall': recall}

    return metrics
