import copy
from typing import List, Tuple, Optional

import numpy as np

from frag_gnn.utils.standardization import Standardization
from frag_gnn.utils.misc_utils import get_mass_tol
import spectral_entropy

class Spectrum(object):
    """
    class for a mass spectrum
    """
    #_supported_intensity_aggregate_method = ['sum', 'min', 'max']
    #_supported_specrtrum_merge_method = ['union', 'intersect']

    def __init__(self):
        # default value to empty and values

        self.is_normalized: bool = False
        self.is_sorted: bool = False
        self.chromatography_type: str = Standardization.NA
        self.adduct_type: str = Standardization.NA
        self.analyzer_type: str = Standardization.NA
        self.ionization_type: str = Standardization.NA
        self.charge_type: str = Standardization.NA
        self.ms_level: str = Standardization.NA

        self.collision_energy_prefix: str = Standardization.NA
        self.collision_energy_unit: str = Standardization.NA
        self.collision_method: str = Standardization.NA
        self.collision_gas_type: str = Standardization.NA
        self.instrument_model: str = Standardization.NA

        self.inchikey: str = Standardization.NA
        self.formula: str = Standardization.NA
        self.compound_name: str = Standardization.NA
        self.smiles_or_inchi: str = Standardization.NA

        self.data_source = Standardization.NA
        self.data_source_id = Standardization.NA
        self.compound_data_source = Standardization.NA
        self.compound_data_source_id = Standardization.NA

        self.peaks_count: int = 0
        self.mz_decimal_place: int = 0
        self.precursor_mz: float = -1.0
        self.compound_exact_mass: float = -1.0  # Monoisotopic mass
        self.collision_energy: float = -1.0

        self.min_mz: float = -1.0
        self.max_mz: float = -1.0

        self.peaks = None
        self.annotations = None

    def add_peaks(self, peaks:List[List[float, float]], annotations:list[list[str]]):
        """
            method to add peak
        Args:
            new_peak (Peak): [description]
        """
        self.peaks = np.ndarray(peaks)
        self.annotations = annotations


    def get_max_intensity(self):
        return np.max(self.peaks[:,1])

    def get_max_mz(self):
        return np.max(self.peaks[:,0])
    
    def get_min_mz(self):
        return np.min(self.peaks[:,0])
    
    def normalize(self, scale=100.0):
        """    
        Method to normalize spectra
        normalized to 100 based intensity

        Args:
            scale (float, optional): [description]. Defaults to 100.0.
            cfm_style (bool, optional): [description]. cfm2 style normalize Defaults to False. 
        """
        if len(self.peaks) == 0:
            pass
        else:
            # make sure we have valied min and max
            # this will crash if peak array is empty
            divisor = self.get_max_intensity()
            self.peaks[:,1] =  self.peaks[:,1]/divisor * scale
            self.is_normalized = True

    def sort_by_mz(self):
        """Method to sort peaks by mass
        """
        self.peaks = self.peaks[self.peaks [:, 0].argsort()]

    def remove_noise(self, intensity_threshold=1.0,
                    centroid_mz_distance=0.1):
        """ 
            Method to remove noise
        Args:
            intensity_threshold (float, optional): [description]. Defaults to 1.0.
            centroid_mz_distance (float, optional): [description]. Defaults to 0.1.
        """
        if not self.is_sorted:
            self.sort_by_mz()

        if not self.is_normalized:
            self.normalize()

        max_allowed_mz: float = None,
        if self.precursor_mz is not None and self.precursor_mz != Standardization.NA:
            max_allowed_mz = self.precursor_mz + centroid_mz_distance
        self.peaks = spectral_entropy.clean_spectrum(self.peaks, max_mz= max_allowed_mz, noise_removal = intensity_threshold, \
                                                     ms2_da = centroid_mz_distance)

        self.peaks_count = len(self.peaks)


    def __str__(self):
        return self.to_string()

    def _get_meta_string(self) -> str:
        string = 'smiles_or_inchi: ' + str(self.smiles_or_inchi) + '\n' + \
                 'Inchikey: ' + str(self.inchikey) + '\n' + \
                 'precursor_mz: ' + str(self.precursor_mz) + '\n' + \
                 'compound_exact_mass: ' + str(self.compound_exact_mass) + '\n' + \
                 'is_normalized: ' + str(self.is_normalized) + '\n' + \
                 'is_sorted: ' + str(self.is_sorted) + '\n' + \
                 'chromatography_type: ' + str(self.chromatography_type) + '\n' + \
                 'adduct_type: ' + str(self.adduct_type) + '\n' + \
                 'analyzer_type: ' + str(self.analyzer_type) + '\n' + \
                 'ionization_type: ' + str(self.ionization_type) + '\n' + \
                 'charge_type: ' + str(self.charge_type) + '\n' + \
                 'ms_level: ' + str(self.ms_level) + '\n' + \
                 'collision_energy: ' + str(self.collision_energy) + '\n' + \
                 'collision_method: ' + str(self.collision_method) + '\n' + \
                 'collision_gas_type: ' + str(self.collision_gas_type) + '\n' + \
                 'instrument_model: ' + str(self.instrument_model) + '\n' + \
                 'mz_decimal_place: ' + str(self.mz_decimal_place) + '\n' + \
                 'data_source: ' + str(self.data_source) + '\n' + \
                 'data_source_id: ' + str(self.data_source_id) + '\n' + \
                 'compound_data_source: ' + str(self.compound_data_source) + '\n' + \
                 'compound_data_source_id: ' + str(self.compound_data_source_id) + '\n' + \
                 'peaks_count: ' + str(self.peaks_count) + '\n'
        return string

    def to_string(self, simplified=False) -> str:
        string = ''
        if simplified:
            string += "#" + str(self.ionization_type) + ' ' + str(self.adduct_type) + ' ' + str(self.ms_level) + '\n'
            if self.ms_level == Standardization.MS2:
                string += "#" + str(self.collision_method) + '@' + str(self.collision_energy) + str(
                    self.collision_energy_unit) + '\n'
            string += "#Inchikey: " + str(self.inchikey) + '\n'
            if self.smiles_or_inchi.startswith('InChI'):
                string += '#' + str(self.smiles_or_inchi) + '\n'
            else:
                string += '#Smiles: ' + str(self.smiles_or_inchi) + '\n'
            string += '#PrecursorMz: ' + str(self.precursor_mz) + '\n'
            string += '#DataSource: ' + str(self.data_source) + '\n'
            string += '#DataSourceID: ' + str(self.data_source_id) + '\n'
            string += '#CompoundDataSource: ' + str(self.compound_data_source) + '\n\n'

        else:
            string += self._get_meta_string() + 'peaks:\n'

        for peak in self.peaks:
            string += peak.to_string(simplified) + '\n'

        return string

    def get_summary(self, include_inchikey=True) -> str:
        """    Method Return spectrum summary

        Args:
            include_inchikey (bool, optional): [description]. Defaults to True.

        Returns:
            str: [description]
        """
        string = ""
        if include_inchikey:
            string = str(self.inchikey) + '\n'

        string += str(self.chromatography_type) + '-' + str(self.ionization_type) + '-' + str(
            self.analyzer_type) + '-' + str(self.adduct_type) + '@' + \
                  str(self.collision_energy) + str(self.collision_energy_unit) + '\n'

        # if include_equivalent_qtof_cid:
        #    string += 'Equivalent QTOF CID: {:.1f} eV \n'.format(
        #        self.equivalent_qtof_collision_energy)

        string += 'Source: {}-{}'.format(self.data_source, self.data_source_id)
        return string

    def get_spectrum_compound_summary(self) -> str:
        """
        Returns:
            str: Method Return spectrum compund summary
        """
        string = "Name: {} \nInchIKey: {} \n".format(
            self.compound_name, self.inchikey)
        return string

    def get_spectrum_type_summary(self) -> str:
        """    Method Return spectrum type summary
        Returns:
            str: [description]
        """
        ms_level = "MS/MS" if self.ms_level == Standardization.MS2 else "MS"

        #string = " {}-{}-{}-{}-{}@{}".format(self.data_source, self.ionization_type,
        #                                 self.analyzer_type, self.adduct_type,ms_level,
        #                                 str(self.collision_energy) + str(self.collision_energy_unit))
        
        string = "{} {}@{}".format(self.data_source, ms_level,
                                         str(self.collision_energy) + str(self.collision_energy_unit))

        return string

    def get_collision_energy_string(self) -> str:
        """ 
            Method Return collision energy string
        Returns:
            str: [description]
        """
        return str(self.collision_energy) + str(self.collision_energy_unit)

    def set_mz_decimal_place(self):
        """
        Method to set mz_decimal_place
        """
        for peak in self.peaks:
            mass_str = str(peak.mz)

            if mass_str.startswith('.'):
                curr_mz_decimal_place = len(mass_str) - 1
            else:
                data = mass_str.split('.')
                if len(data) == 1:
                    curr_mz_decimal_place = 0
                else:
                    curr_mz_decimal_place = len(data[1])
            if curr_mz_decimal_place > self.mz_decimal_place:
                self.mz_decimal_place = curr_mz_decimal_place

    def annotate_with_mz(self, intensity_threshold=0.0):
        """Method annotate peak with m/z
        """
        if self.annotations is None:
            annotations = [[] for _ in range(len(self.peaks))]
        for idx, values in enumerate(self.peaks):
            if values[1] > intensity_threshold:
                self.annotations[idx].append(str(values[1]))

    def post_process(self):
        """
            Run all post process
            NOTE when cfm_style normalization enabled, noise remove will also use the same style spectrum
            You may want to remove noise on a regular normalization and then transfer to cfm style spectrum

        Args:
            cfm_style (bool, optional): [description]. Defaults to False.
        """
        self.sort_by_mz()
        self.normalize()
        
        self.min_mz = self.get_min_mz()
        self.max_mz = self.get_max_mz()
        self.peaks_count = self.peaks.shape[0]
        #self.set_mz_decimal_place()
        self.peaks_count = len(self.peaks)
    

    def remove_all_peaks(self):
        """remove all peaks
        """
        self.peaks = None

    def get_peaks_as_np(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.peaks

    '''
    Method compute qtof ev from obitrap nce
    '''

    @staticmethod
    def nce_to_ev(isolation_center: float, nce: float, charge_state: float = 1.0) -> float:
        ev = nce * isolation_center / (500 * charge_state)
        return ev

    '''
    Method compute obitrap nce from qtof ev 
    '''
    @staticmethod
    def ev_to_nce(isolation_center: float, ev: float, charge_state: float = 1.0) -> float:
        """_summary_

        Args:
            isolation_center (float): _description_
            ev (float): _description_
            charge_state (float, optional): _description_. Defaults to 1.0.

        Returns:
            float: _description_
        """
        nce = ev * (500 * charge_state) / isolation_center
        return nce

    def convert_intensity_to_log_scale(self):
        """[Covert peak intensity to log scale, such that peak.intensity = np.log(peak.intensity+1)]
        """
        #for peak in self.peaks:
        #    peak.intensity = np.log(peak.intensity + 1)

        #self.normalize()
        raise NotImplementedError

    def convert_intensity_to_linear_scale(self):
        """[Covert peak intensity to linear scale, such that peak.intensity = np.log(peak.intensity+1)]
        """
        #ratio = 100 / np.log(101)
        #for peak in self.peaks:
        #    peak.intensity = np.exp(peak.intensity / ratio) - 1

        #self.normalize()
        raise NotImplementedError

    def get_num_peaks(self) -> int:
        """ Method to get number of peaks
        Returns: number of peaks
        """
        return len(self.peaks)