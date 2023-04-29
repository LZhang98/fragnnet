import copy
from typing import List, Tuple, Optional
from frag_gnn.utils.standardization import MsLevel

import numpy as np

class Spectrum(object):
    """
    class for a mass spectrum
    """
    _supported_intensity_aggregate_method = ['sum', 'min', 'max']
    _supported_specrtrum_merge_method = ['union', 'intersect']

    def __init__(self):
        # default value to None
        self.peaks: np.array = None
        self.is_normalized: bool = False
        self.is_sorted: bool = False
        self.chromatography_type: str = None
        self.adduct_type: str = None
        self.analyzer_type: str = None
        self.ionization_type: str = None
        self.charge_type: str = None
        self.ms_level: str = None

        self.collision_energy_prefix: str = None
        self.collision_energy_unit: str = None
        self.collision_method: str = None
        self.collision_gas_type: str = None
        self.instrument_model: str = None

        self.inchikey: str = None
        self.formula: str = None
        self.compound_name: str = None
        self.smiles_or_inchi: str = None

        self.data_source = None
        self.data_source_id = None
        self.compound_data_source = None
        self.compound_data_source_id = None

        self.peaks_count: int = 0
        self.mz_decimal_place: int = 0
        self.precursor_mz: float = -1.0
        self.compound_exact_mass: float = -1.0  # Monoisotopic mass
        self.collision_energy: float = -1.0

        self.min_mz: float = -1.0
        self.max_mz: float = -1.0
        self.min_intensity: float = -1.0
        self.max_intensity: float = -1.0

        self.compound_sample_frequency = 1


    def normalize(self, scale=100.0):
        """    
        Method to normalize spectra
        normalized to 100 based intensity

        Args:
            scale (float, optional): [description]. Defaults to 100.0.
            cfm_style (bool, optional): [description]. cfm2 style normalize Defaults to False. 
        """
        if len(self.peaks) > 0:
            # make sure we have valied min and max
            if self.max_intensity < 0:
                self.max_intensity = max(
                    self.peaks, key=lambda i: i.intensity).intensity
            if self.min_intensity < 0:
                self.min_intensity = min(
                    self.peaks, key=lambda i: i.intensity).intensity

            divisor = self.max_intensity 

            if divisor == 0:
                for peak in self.peaks:
                    print(peak.to_string())

            for peak in self.peaks:
                # print(peak)
                peak.intensity = peak.intensity / divisor * scale

            # re-assgin max and min
            self.max_intensity = max(
                self.peaks, key=lambda i: i.intensity).intensity
            self.min_intensity = min(
                self.peaks, key=lambda i: i.intensity).intensity
            self.is_normalized = True

    def sort_by_mz(self):
        """Method to sort peaks by mass
        """
        self.peaks.sort(key=lambda i: i.mz)
        self.is_sorted = True
        if len(self.peaks) > 0 :
            self.min_mz = self.peaks[0].mz
            self.max_mz = self.peaks[-1].mz

    def remove_noise(self, intensity_threshold=1.0, centroid=False, 
                    centroid_mz_distance=0.1, remove_noise_more_than_pmass = True):
        """ 
            Method to remove noise
        Args:
            intensity_threshold (float, optional): [description]. Defaults to 1.0.e.
            centroid (bool, optional): [description]. Defaults to False.
            centroid_mz_distance (float, optional): [description]. Defaults to 0.1.
        """
        if not self.is_sorted:
            self.sort_by_mz()

        if not self.is_normalized:
            self.normalize()



        if remove_noise_more_than_pmass and self.precursor_mz is not None and self.precursor_mz != None:
            self.peaks = [i for i in self.peaks if i.mz <= self.precursor_mz + 0.1]

        if centroid:
            self._centroid(mz_distance=centroid_mz_distance)
            self.sort_by_mz()

        self._remove_low_intensity(intensity_threshold=intensity_threshold)

        self.peaks_count = len(self.peaks)


    def _remove_low_intensity(self, intensity_threshold=1.0):
        """Method to remove low intensity peak, intensity_threshold is measued in percent
        Args:
            intensity_threshold (float, optional): [description]. Defaults to 1.0.
        """
        self.peaks = [i for i in self.peaks if i.intensity >=
                      intensity_threshold]

    def _centroid(self, mz_distance=0.1):
        """    Method that remove duplicated peaks within given mz_distance
        Args:
            mz_distance (float, optional): [description]. Defaults to 0.1.
        """
        self.peaks = self._get_centroid_spectrum(
            0, len(self.peaks) - 1, mz_distance)

    def _get_centroid_spectrum(self, lower_bond, upper_bond, mz_distance=0.1):
        if lower_bond > upper_bond:
            return []

        if lower_bond == upper_bond:
            # print(lower_bond, self.peaks[lower_bond].mz, self.peaks[lower_bond].intensity)
            return [self.peaks[lower_bond]]

        # get current highest peak idx
        index = self.peaks.index(
            max(self.peaks[lower_bond:upper_bond + 1], key=lambda i: i.intensity))
        current_mass = self.peaks[index].mz

        new_peaks = [self.peaks[index]]

        # left part upper bond
        ls_upper_bond = index - 1
        # if ls_upper_bond > 0:
        while ls_upper_bond > -1:
            if self.peaks[ls_upper_bond].mz > current_mass - mz_distance:
                ls_upper_bond -= 1
            else:
                break
            
        # print('ls_upper_bond',ls_upper_bond)
        if ls_upper_bond > -1:
            left = self._get_centroid_spectrum(lower_bond, ls_upper_bond, mz_distance)
            new_peaks = new_peaks + left

        # right part lower bond
        rs_lower_bound = index + 1
        while rs_lower_bound < len(self.peaks):
            if self.peaks[rs_lower_bound].mz < current_mass + mz_distance:
                rs_lower_bound += 1
            else:
                break
        if rs_lower_bound < len(self.peaks):
            right = self._get_centroid_spectrum(rs_lower_bound, upper_bond, mz_distance)
            new_peaks += right
        return new_peaks

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
                 'compound_sample_frequency: ' + str(self.compound_sample_frequency) + '\n' + \
                 'derivatization_type: ' + self.derivatization_type + '\n' + \
                 'derivatization_mass: ' + str(self.derivatization_mass) + '\n' + \
                 'peaks_count: ' + str(self.peaks_count) + '\n'
        return string

    def to_string(self, simplified=False) -> str:
        string = ''
        if simplified:
            string += "#" + str(self.ionization_type) + ' ' + str(self.adduct_type) + ' ' + str(self.ms_level) + '\n'
            if self.ms_level == MsLevel.MS2:
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


    def post_process(self):
        """
            Run all post process
            NOTE when cfm_style normalization enabled, noise remove will also use the same style spectrum
            You may want to remove noise on a regular normalization and then transfer to cfm style spectrum
        """
        self.sort_by_mz()
        self.normalize()
        #self.set_mz_decimal_place()
        self.peaks_count = len(self.peaks)

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
        nce = ev * (500 * charge_state) / isolation_center
        return nce

    @staticmethod
    def get_matched_peaks(ref_spectrum: 'Spectrum', predicted_spectrum: 'Spectrum',
                          annotate: bool = False, abs_tol: float = 0.01, ppm_tol: float = 10,
                          include_miss_matches: bool = False) -> Tuple[int, List[Tuple[Peak, Peak]]]:
        """
            Method to get matched peaks between two spectra, Return a list of matched peaks, empty match will be ignored
        Args:
            ref_spectrum (Spectrum): [description]
            predicted_spectrum (Spectrum): [description]
            annotate (bool, optional): [description]. Defaults to False.
            abs_tol (float, optional): [description]. Defaults to 0.01.
            ppm_tol (float, optional): [description]. Defaults to 10.
            include_miss_matches (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        if ref_spectrum.is_sorted is False:
            ref_spectrum.sort_by_mz()
        if predicted_spectrum.is_sorted is False:
            predicted_spectrum.sort_by_mz()

        matches = []
        num_matched = 0
        sp1_idx, sp2_idx = 0, 0

        while sp1_idx < len(ref_spectrum.peaks) and sp2_idx < len(predicted_spectrum.peaks):
            sp1_peak = ref_spectrum.peaks[sp1_idx]
            sp1_mz = sp1_peak.mz

            sp2_peak = predicted_spectrum.peaks[sp2_idx]
            sp2_mz = sp2_peak.mz

            # get tolarnc
            allowed_tol = Utilities.get_mass_tol(
                max(sp1_mz, sp2_mz), abs_tol, ppm_tol)

            diff = abs(sp1_mz - sp2_mz)
            # print(allowed_tol , diff, sp1_mz,  sp2_mz)
            # if found a pair
            if diff <= allowed_tol:
                sp1_idx += 1
                sp2_idx += 1
                matches.append((sp1_peak, sp2_peak))
                num_matched += 1
                if annotate:
                    sp1_peak.annotation = "#{}".format(num_matched)
                    sp2_peak.annotation = "#{}".format(num_matched)

            else:
                if sp1_mz < sp2_mz:
                    # skip sp1 peak
                    sp1_idx += 1
                    if include_miss_matches:
                        matches.append((sp1_peak, Peak(sp1_peak.mz, 0.0)))
                else:
                    # skip sp2 speak
                    sp2_idx += 1
                    if include_miss_matches:
                        matches.append((Peak(sp2_peak.mz, 0.0), sp2_peak))
            
        if include_miss_matches:
            while sp1_idx < len(ref_spectrum.peaks):
                sp1_peak = ref_spectrum.peaks[sp1_idx]
                matches.append((sp1_peak, Peak(sp1_peak.mz, 0.0)))
                sp1_idx += 1
            while sp2_idx < len(predicted_spectrum.peaks):
                sp2_peak = predicted_spectrum.peaks[sp2_idx]
                matches.append((Peak(sp2_peak.mz, 0.0), sp2_peak))
                sp2_idx+= 1

        return num_matched, matches

    def get_peak_by_mz(self, mz, abs_tol=0.001) -> Optional[Peak]:
        """    get peak by mz with abs_tol
        Args:
            mz ([type]): [description]
            abs_tol (float, optional): [description]. Defaults to 0.001.

        Returns:
            Peak: [description]
        """
        for peak in self.peaks:
            if abs(mz - peak.mz) < abs_tol:
                return peak
        return None

    def remove_all_peaks(self):
        self.peaks = []

    @staticmethod
    def _get_merged_peak(base_peak: Peak, additional_peak: Peak, intensity_aggregate_method: str) -> Peak:
        if intensity_aggregate_method not in Spectrum._supported_intensity_aggregate_method:
            raise Exception("intensity aggregate method [{}]".format(
                ',').join(Spectrum._supported_intensity_aggregate_method))

        first_intensity = base_peak.intensity
        second_intensity = 0.0
        if additional_peak is not None:
            second_intensity = additional_peak.intensity
        intensity = 0.0
        if intensity_aggregate_method == 'sum':
            intensity = (first_intensity + second_intensity)
        if intensity_aggregate_method == 'max':
            intensity = max(first_intensity, second_intensity)
        if intensity_aggregate_method == 'min':
            intensity = min(first_intensity, second_intensity)

        return Peak(base_peak.mz, intensity, base_peak.annotation)

    @staticmethod
    def merge_spectrum(base_spectrum: 'Spectrum', additional_spectrum: 'Spectrum', merge_method: str = 'union',
                         intensity_aggregate_method: str = 'sum', normalize = True) -> 'Spectrum':
        """    Method that create new spectra by intersect two spectra, mz is based on base_spectrum in case of matching peaks
                merge_method:str = 'union' or 'intersect'
                intensity_aggregate_method can be 'sum' , 'max' , or 'min'
        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """

        if merge_method not in Spectrum._supported_specrtrum_merge_method:
            raise Exception(
                "spectrum merge method [union,intersect], Input was " + intensity_aggregate_method)

        if additional_spectrum.get_num_peaks() == 0:
            return base_spectrum

        include_miss_matches = (merge_method == 'union')

        _, matched_peaks = Spectrum.get_matched_peaks(
            base_spectrum, additional_spectrum, include_miss_matches=include_miss_matches)

        merged_spectrum = copy.deepcopy(base_spectrum)
        merged_spectrum.remove_all_peaks()
        for peak_pairs in matched_peaks:
            merged_peak = None
            if peak_pairs[0] is not None:
                merged_peak = Spectrum._get_merged_peak(
                    peak_pairs[0], peak_pairs[1], intensity_aggregate_method)
            elif peak_pairs[1] is not None:
                merged_peak = Spectrum._get_merged_peak(
                    peak_pairs[1], peak_pairs[0], intensity_aggregate_method)
            if merged_peak:
                merged_spectrum.add_peak(merged_peak)

        merged_spectrum.sort_by_mz()
        
     
        merged_spectrum.max_intensity = max(merged_spectrum.peaks, key=lambda i: i.intensity).intensity
        merged_spectrum.min_intensity = min(merged_spectrum.peaks, key=lambda i: i.intensity).intensity

        if normalize:
            merged_spectrum.normalize()

        return merged_spectrum


    @staticmethod
    def merge_multi_spectrum(spectrum_list:List['Spectrum'], merge_method: str = 'union',
                         intensity_aggregate_method: str = 'sum',) -> 'Spectrum':
        """    Method that create new spectra by intersect two spectra, mz is based on base_spectrum in case of matching peaks
                merge_method:str = 'union' or 'intersect'
                intensity_aggregate_method can be 'sum' , 'max' , or 'min'
        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """

        if merge_method not in Spectrum._supported_specrtrum_merge_method:
            raise Exception(
                "spectrum merge method [union,intersect], Input was " + intensity_aggregate_method)

        include_miss_matches = (merge_method == 'union')
        base_spectrum = spectrum_list[0]

        peak_dict = {}
        
        for additional_spectrum in spectrum_list[1:]:
            _, matched_peaks = Spectrum.get_matched_peaks(
                base_spectrum, additional_spectrum, include_miss_matches=include_miss_matches)

            tmp_merged_spectrum = Spectrum()
            for peak_pairs in matched_peaks:
                merged_peak = None
                current_mz = None
                current_mz = peak_pairs[0].mz if peak_pairs[0] is not None else peak_pairs[1].mz
                if current_mz not in peak_dict:
                    peak_dict[current_mz] = [peak_pairs[0]]

                peak_dict[current_mz].append(peak_pairs[1])

                # update base_spectrum
                merged_peak = None
                if peak_pairs[0] is not None:
                    merged_peak = Spectrum._get_merged_peak(
                        peak_pairs[0], peak_pairs[1], intensity_aggregate_method)
                elif peak_pairs[1] is not None:
                    merged_peak = Spectrum._get_merged_peak(
                        peak_pairs[1], peak_pairs[0], intensity_aggregate_method)
                if merged_peak:
                    tmp_merged_spectrum.add_peak(merged_peak)

            
            base_spectrum = tmp_merged_spectrum
        
        merged_spectrum = Spectrum()
        for mz in peak_dict:
            merged_spectrum.add_peak(Spectrum._get_merged_peak_multi(peak_dict[mz], intensity_aggregate_method))

        merged_spectrum.sort_by_mz()
        merged_spectrum.max_intensity = max(merged_spectrum.peaks, key=lambda i: i.intensity).intensity
        merged_spectrum.min_intensity = min(merged_spectrum.peaks, key=lambda i: i.intensity).intensity
        merged_spectrum.normalize()

        return merged_spectrum
        
    @staticmethod
    def _get_merged_peak_multi(peak_list: Peak, intensity_aggregate_method: str) -> Peak:
        if intensity_aggregate_method not in Spectrum._supported_intensity_aggregate_method:
            raise Exception("intensity aggregate method [{}]".format(
                ',').join(Spectrum._supported_intensity_aggregate_method))

        mz = 0
        count = 0
        intensity = 10000 if intensity_aggregate_method == 'min' else 0
        for peak in peak_list:
            if peak is not None:
                mz += peak.mz
                count += 1

            if intensity_aggregate_method == 'sum':
                intensity += peak.intensity
            if intensity_aggregate_method == 'max':
                intensity = max(intensity, peak.intensity)
            if intensity_aggregate_method == 'min':
                intensity = min(intensity, peak.intensity)

        mz = mz/count
        #intensity = intensity/count 
        return Peak(mz, intensity)
        
    def get_cfm_text_str(self, annotation_right: bool = False):
        string = ''
        for peak in self.peaks:
            if not annotation_right or peak.annotation == None:
                string += '{:.6f} {:.2f}\n'.format(peak.mz, peak.intensity)
            else:
                string += '{:.6f} {:.2f} {}\n'.format(peak.mz, peak.intensity, peak.annotation)
        return string

    def fill_formula(self):
        if self.smiles_or_inchi != None and self.formula == None:
            try:
                self.formula = Utilities.get_formula_by_chemid(self.smiles_or_inchi)
            except ValueError:
                print('Warning,can not compute formula from', self.smiles_or_inchi)

    def compute_compound_exact_mass(self):
        #print(self.smiles_or_inchi, self.compound_exact_mass)
        if self.smiles_or_inchi != None and (self.compound_exact_mass == None or self.compound_exact_mass == -1.0):
            try:
                self.compound_exact_mass = Utilities.get_mol_mw_by_chemid(self.smiles_or_inchi)
            except ValueError:
                print('Warning,can not compute compound_exact_mass from', self.smiles_or_inchi)

    def compute_precursor_mz(self):
        if self.compound_exact_mass == None and self.adduct_type == None:
            #print("Warning, compound_exact_mass is unknown")
            return
        self.precursor_mz = Utilities.neutral_mass_to_ion_mass(self.compound_exact_mass, self.adduct_type)
         
    def get_data_for_spec2vec(self):

        mz = np.array([peak.mz for peak in self.peaks], dtype="float")
        intensities = np.array(
            [peak.intensity / self.max_intensity for peak in self.peaks], dtype="float")
        metadata = dict(precursor_mz=self.precursor_mz)

        return mz, intensities, metadata

    def get_peaks(self) -> List[Peak]:
        return self.peaks

    def get_mzs(self) -> List[float]:
        return [p.mz for p in self.peaks]

    def add_peaks(self, mz_array: List[float], intensity_array: List[float], annotation_array:Optional[List[str]] = None,
        min_mz=None, max_mz=None):

        if annotation_array is None:
            annotation_array = [None for _ in mz_array]
        for mz, intensity,annotation in zip(mz_array, intensity_array, annotation_array):
            if min_mz is not None and mz < min_mz:
                continue
            if max_mz is not None and mz > max_mz:
                continue
            if intensity > 0:
                self.add_peak( Peak(mz, intensity,annotation))
        self.post_process()

    def convert_intensity_to_log_scale(self):
        """[Covert peak intensity to log scale, such that peak.intensity = np.log(peak.intensity+1)]
        """
        for peak in self.peaks:
            peak.intensity = np.log(peak.intensity + 1)

        self.max_intensity = max(self.peaks, key=lambda i: i.intensity).intensity
        self.min_intensity = min(self.peaks, key=lambda i: i.intensity).intensity

        self.normalize()

    def convert_intensity_to_linear_scale(self):
        """[Covert peak intensity to linear scale, such that peak.intensity = np.log(peak.intensity+1)]
        """
        ratio = 100 / np.log(101)
        for peak in self.peaks:
            peak.intensity = np.exp(peak.intensity / ratio) - 1

        self.max_intensity = max(self.peaks, key=lambda i: i.intensity).intensity
        self.min_intensity = min(self.peaks, key=lambda i: i.intensity).intensity

        self.normalize()

    def get_splash_hash_input(self) -> List[Tuple[float,float]]:
        """[summary]

        Returns:
            [type]: [description]
        """
        rev_list = []
        for peak in self.peaks:
            rev_list.append((peak.mz, peak.intensity))
        return rev_list

    def get_num_peaks(self) -> int:
        """ Method to get number of peaks
        Returns: number of peaks
        """
        return len(self.peaks)

    def rescale_peaks(self, scale_factor):
        """Method to rescale peaks by a given factor

        Args:
            factor (float): factor to rescale peaks by
        """
        for peak in self.peaks:
            peak.intensity *= scale_factor
    
    def get_dot_ms_str(self) -> str:
        dot_ms_str = ""
        dot_ms_str +=">parentmass {}\n".format(self.precursor_mz)
        dot_ms_str +=">ionization {}\n".format(self.adduct_type)
        dot_ms_str +="#instrument ESI-QTOF\n\n"
        dot_ms_str +=">collision {}{}\n".format(self.collision_energy,self.collision_energy_unit)
        for peak in self.peaks:
            dot_ms_str +="{} {}\n".format(peak.mz, peak.intensity)
        dot_ms_str += "\n"
        return dot_ms_str
    
    def get_mgf_str(self, title) -> str:
        mgf_str = "BEGIN IONS\n"
        mgf_str +="PEPMASS={}\n".format(self.precursor_mz)
        mgf_str +="CHARGE=1+\n" if self.charge_type == Standardization.Pos else "CHARGE=1-\n"
        mgf_str +="INSTRUMENT=ESI-QTOF\n"
        mgf_str +="TITLE={}@{}{}\n".format(title, self.collision_energy,self.collision_energy_unit)
        for peak in self.peaks:
            mgf_str +="{} {}\n".format(peak.mz, peak.intensity)
        mgf_str += "END IONS\n\n"
        return mgf_str