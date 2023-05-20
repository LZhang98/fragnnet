from typing import Dict, List, Tuple
from scipy.stats import entropy
import numpy as np

class Comparators:

    # DICE
    @staticmethod
    def _get_dice(ref_spectrum: Spectrum, predicted_spectrum: Spectrum, matched_peaks_count: int,
                  verbose: bool = False) -> float:
        dice = 2.0 * matched_peaks_count / \
               (ref_spectrum.peaks_count + predicted_spectrum.peaks_count) if ref_spectrum.peaks_count + \
                                                                              predicted_spectrum.peaks_count != 0 else 0.0
        if verbose:
            print("Dice: 2.0 * {} / ( {} + {} ) = {}".format(matched_peaks_count,
                                                             ref_spectrum.peaks_count, predicted_spectrum.peaks_count,
                                                             dice))
        return dice

    @staticmethod
    def get_dice(ref_spectrum: Spectrum, predicted_spectrum: Spectrum, verbose: bool = False) -> float:
        """ Method compute dice between two spectrum
        Args:
            ref_spectrum (Spectrum): reference spectrum
            predicted_spectrum (Spectrum): predicted spectrum
            verbose (bool, optional): flag for verbose. Defaults to False.

        Returns:
            float: dice
        """
        matched_peaks_count, _ = Spectrum.get_matched_peaks(
            ref_spectrum, predicted_spectrum)
        return Comparators._get_dice(ref_spectrum, predicted_spectrum, matched_peaks_count, verbose)

    
    # DOT PRODUCT
    @staticmethod
    def _get_adjusted_intensity(peak: Peak, intensity_weight, mz_weight, verbose: bool = False) -> float:
        result = pow(peak.intensity, intensity_weight) * \
                 pow(peak.mz, mz_weight)
        if verbose:
            print("Adjusted Intensity: {}^{} * {}^{} = {} ".format(peak.intensity,
                                                                   intensity_weight, peak.mz, mz_weight, result))
        return result

    @staticmethod
    def _get_total_peak_sum(spectrum: Spectrum, intensity_weight: float, mz_weight: float) -> float:
        result = 0.0
        for peak in spectrum.peaks:
            peak_value = Comparators._get_adjusted_intensity(
                peak, intensity_weight, mz_weight)
            result += peak_value * peak_value
        return result

    @staticmethod
    def _compute_dot_product(ref_spectrum: Spectrum, predicted_spectrum: Spectrum,
                             matched_peaks: List[Tuple[Peak, Peak]],
                             intensity_weight: float, mz_weight: float, verbose: bool = False) -> float:


        num = 0.0
        for matches in matched_peaks:
            p1, p2 = matches[0], matches[1]
            #print(p1.intensity)
            #print(p2.intensity)
            num += Comparators._get_adjusted_intensity(p1, intensity_weight, mz_weight, verbose) * \
                   Comparators._get_adjusted_intensity(
                       p2, intensity_weight, mz_weight, verbose)

        denomp = Comparators._get_total_peak_sum(
            ref_spectrum, intensity_weight, mz_weight)
        denomq = Comparators._get_total_peak_sum(
            predicted_spectrum, intensity_weight, mz_weight)

        score = num * num / (denomp * denomq) if denomp * denomq != 0 else 0.0

        if verbose:
            print("Dot Product: {} * {} / ( {} * {} ) = {}".format(num,
                                                                   num, denomp, denomq, score))

        return score

    @staticmethod
    def _get_dot_product(ref_spectrum: Spectrum, predicted_spectrum: Spectrum, matched_peaks: List[Tuple[Peak, Peak]],
                         verbose: bool = False) -> float:
        return Comparators._compute_dot_product(ref_spectrum, predicted_spectrum, matched_peaks, 0.5, 0.5, verbose)

    @staticmethod
    def get_dot_product(ref_spectrum: Spectrum, predicted_spectrum: Spectrum, verbose: bool = False) -> float:
        """ Method compute dot product between two spectrum
        Args:
            ref_spectrum (Spectrum): reference spectrum
            predicted_spectrum (Spectrum): predicted spectrum
            verbose (bool, optional): flag for verbose. Defaults to False.

        Returns:
            float: dot product 
        """
        _, matched_peaks = Spectrum.get_matched_peaks(
            ref_spectrum, predicted_spectrum)
        return Comparators._get_dot_product(ref_spectrum, predicted_spectrum, matched_peaks, verbose)

    @staticmethod
    def _get_stein_dot_product(ref_spectrum: Spectrum, predicted_spectrum: Spectrum,
                               matched_peaks: List[Tuple[Peak, Peak]],
                               verbose: bool = False) -> float:
        return Comparators._compute_dot_product(ref_spectrum, predicted_spectrum, matched_peaks, 0.6, 3.0, verbose)

    @staticmethod
    def get_stein_dot_product(ref_spectrum: Spectrum, predicted_spectrum: Spectrum, verbose: bool = False) -> float:
        _, matched_peaks = Spectrum.get_matched_peaks(
            ref_spectrum, predicted_spectrum)
        return Comparators._get_stein_dot_product(ref_spectrum, predicted_spectrum, matched_peaks, verbose)

    @staticmethod
    def _get_recall(ref_peak_count: int, matched_peaks_count: int, verbose: bool = False) -> float:
        # tp + fp : total number of predicted peak
        # tp : macthed_peaks
        recall = matched_peaks_count / ref_peak_count if ref_peak_count != 0 else 0.0

        if verbose:
            print("Recall : {} / {} = {}".format(ref_peak_count,
                                                 matched_peaks_count, recall))

        return recall

    @staticmethod
    def get_recall(ref_spectrum: Spectrum, predicted_spectrum: Spectrum, verbose: bool = False) -> float:
        """ Method compute recall between two spectrum
        Args:
            ref_spectrum (Spectrum): reference spectrum
            predicted_spectrum (Spectrum): predicted spectrum
            verbose (bool, optional): flag for verbose. Defaults to False.

        Returns:
            float: recall
        """
        matched_peaks_count, _ = Spectrum.get_matched_peaks(
            ref_spectrum, predicted_spectrum)
        ref_peak_count = ref_spectrum.peaks_count
        return Comparators._get_recall(ref_peak_count, matched_peaks_count, verbose)

    
    @staticmethod
    def _get_precision(pred_peak_count: int, matched_peaks_count: int, verbose: bool = False) -> float:
        # tp + fp : total number of predicted peak
        # tp : macthed_peaks
        precision = matched_peaks_count / pred_peak_count if pred_peak_count != 0 else 0.0

        if verbose:
            print("Precision: {} / {} = {}".format(pred_peak_count,
                                                   matched_peaks_count, precision))

        return precision

    @staticmethod
    def get_precision(ref_spectrum: Spectrum, predicted_spectrum: Spectrum, verbose: bool = False) -> float:
        """ Method compute precision  between two spectrum
        Args:
            ref_spectrum (Spectrum): reference spectrum
            predicted_spectrum (Spectrum): predicted spectrum
            verbose (bool, optional): flag for verbose. Defaults to False.

        Returns:
            float: precision
        """
        matched_peaks_count, _ = Spectrum.get_matched_peaks(
            ref_spectrum, predicted_spectrum)
        pred_peak_count = predicted_spectrum.peaks_count
        return Comparators._get_precision(pred_peak_count, matched_peaks_count, verbose)

    
    @staticmethod
    def get_metrics(ref_spectrum: Spectrum, predicted_spectrum: Spectrum, verbose: bool = False) -> Dict[str, float]:
        matched_peaks_count, matched_peaks = Spectrum.get_matched_peaks(
            ref_spectrum, predicted_spectrum)
        pred_peak_count = predicted_spectrum.peaks_count
        ref_peak_count = ref_spectrum.peaks_count

        precision = Comparators._get_precision(
            pred_peak_count, matched_peaks_count, verbose)
        recall = Comparators._get_recall(
            ref_peak_count, matched_peaks_count, verbose)
        dice = Comparators._get_dice(
            ref_spectrum, predicted_spectrum, matched_peaks_count, verbose)
        dot_product = Comparators._get_dot_product(
            ref_spectrum, predicted_spectrum, matched_peaks, verbose)

        metrics = {'Dice': dice, 'Dot Product': dot_product, 'Precision': precision, 'Recall': recall}

        return metrics

    @staticmethod
    def get_num_matched(ref_spectrum: Spectrum, predicted_spectrum: Spectrum) -> Tuple[int, int, int]:
        num_matched, _ = Spectrum.get_matched_peaks(
            ref_spectrum, predicted_spectrum)
        return num_matched, ref_spectrum.get_num_peaks() - num_matched, predicted_spectrum.get_num_peaks() - num_matched

    @staticmethod
    def get_num_matched_percent(ref_spectrum: Spectrum, predicted_spectrum: Spectrum) -> Tuple[float,float]:
        num_matched, _ = Spectrum.get_num_matched(ref_spectrum, predicted_spectrum)

        return num_matched/ref_spectrum.get_num_peaks(), num_matched/predicted_spectrum.get_num_peaks()

    @staticmethod
    def get_entropy_similarity(ref_spectrum: Spectrum, predicted_spectrum: Spectrum, abs_tol=0.01, ppm_tol=10):
    
        _, matched_peaks = Spectrum.get_matched_peaks(ref_spectrum, predicted_spectrum, abs_tol = abs_tol, ppm_tol = ppm_tol, include_miss_matches=True)
        ref_matched_peak = [p[0].intensity/100 for p in matched_peaks]
        predicted_matched_peak = [p[1].intensity/ 100 for p in matched_peaks]

        return Comparators._entropy_similarity(ref_matched_peak, predicted_matched_peak)

    @staticmethod
    def _entropy_similarity(a, b):
        entropy_a, a = Comparators._get_entropy_and_weighted_intensity(a)
        entropy_b, b = Comparators._get_entropy_and_weighted_intensity(b)

        entropy_merged = entropy(a + b)
        return 1 - (2 * entropy_merged - entropy_a - entropy_b) / np.log(4)

    @staticmethod
    def _get_entropy_and_weighted_intensity(intensity):
        spectral_entropy = entropy(intensity)
        if spectral_entropy < 3:
            weight = 0.25 + 0.25 * spectral_entropy
            weighted_intensity = np.power(intensity, weight)
            intensity_sum = np.sum(weighted_intensity)
            weighted_intensity /= intensity_sum
            spectral_entropy = entropy(weighted_intensity)
            return spectral_entropy, weighted_intensity
        else:
            return spectral_entropy, intensity