import re
from tqdm import tqdm
import pandas as pd

MGF_KEY_DICT = {
	'FORMULA': 'formula',
	'SMILES': 'smiles',
	'INCHI': 'inchi',
	'INCHIKEY': 'inchi_key',
	'IONMODE': 'ion_mode',
	'ADDUCT': 'prec_type',
	'COMPOUND_NAME': 'compound_name',
	'NUM_PEAKS': 'num_peaks',
	'PRECURSOR_MZ': 'prec_mz',
	#'PEPMASS': 'precursor_mz',
	'PARENT_MASS': 'exact_mass',
	'MS_LEVEL': 'spec_type',
	'CHARGE': 'charge',
	'INSTRUMENT_TYPE': 'inst_type',
	'COLLISION_ENERGY_1': 'col_energy_1',
	'COLLISION_ENERGY_2': 'col_energy_2',
	'COLLISION_ENERGY_3': 'col_energy_3',
	'NORMALIZED_COLLISION_ENERGY_1': 'normalized_col_energy_1',
	'NORMALIZED_COLLISION_ENERGY_2': 'normalized_col_energy_2',
	'NORMALIZED_COLLISION_ENERGY_3': 'normalized_col_energy_3',
	'FOLD': 'fold',
	'INDEX': 'spec_id',
	'SOURCE': 'source',
	'TITLE': 'title',
	'NUM_PEAKS': 'num_peaks',
}
 
def read_mgfs_fp(mgf_fp):
	with open(mgf_fp) as mgf_in:
		spectra_list = parse_mgf_text_list(mgf_in)
				
	return spectra_list

def parse_mgf_text_list(mgf_text_list):

	data_dict = None
	spectra_list = []
	for row in tqdm(mgf_text_list):
		row = row.strip()
		if len(row) == 0:
			continue
		elif row == "BEGIN IONS":
			data_dict = {}
		elif row == "END IONS":
			spectra_list.append(data_dict)
			data_dict = None
		elif re.match(r"^(\d+(?:\.\d+)?)\s(\d+(?:\.\d+)?)", row):
			mz, rel_int = [float(i) for i in row.split()]
			if "peaks" not in data_dict:
				data_dict["peaks"] = ""
			data_dict["peaks"] += f"{mz} {rel_int}\n"
		else:
			for key, col in MGF_KEY_DICT.items():
				if row.startswith(key):
					value = row.split('=',1)[1].strip()
					# in case of empty values
					if value in ['nan','N/A','None','null','']:
						value = None
					# convert to correct type
					if col in ["exact_mass",'prec_mz']:
						value = float(value)
					elif col  == "ion_mode" and value == "POSITIVE":
						value = "P"
					elif col  == "ion_mode" and value == "NEGATIVE":
						value = "N"
					elif col == "num_peaks":
						value = int(value)
					elif col == "smiles": #in case of chemaxon smiles eg.CC1C[C@H](C)C[C@H](OC) |t:46|
						value = value.split()[0]
					data_dict[col] = value
	return spectra_list

if __name__ == "__main__":
	mgf_fp = "data/raw/spectraverse/metadata-can-rem-lowres-prec-matchms-dotcheck-noiserem-fold-metadata.mgf"
	spectra_list = read_mgfs_fp(mgf_fp)
	spectra_df = pd.DataFrame(spectra_list)
 
	spectra_df[:,'ramped'] = spectra_df.apply(lambda x: True if not x['col_energy_1'].isna() and not x['col_energy_2'].isna() and x['col_energy_3'].isna() else False, axis=1)
	spectra_df[:,'stepped'] = spectra_df.apply(lambda x: True if not x['col_energy_1'].isna() and not x['col_energy_2'].isna() and not x['col_energy_3'].isna() else False, axis=1)
	spectra_df.to_pickle("data/raw/spectraverse//metadata-can-rem-lowres-prec-matchms-dotcheck-noiserem-fold-metadata.pkl")
