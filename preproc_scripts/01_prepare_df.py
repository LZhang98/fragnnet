import re
import zipfile
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import argparse
import glob
from fragnnet.utils.data_utils import rdkit_import, seq_apply, par_apply, parse_mass_gym_ce_str
from fragnnet.utils.misc_utils import flatten_lol

MSP_KEY_DICT = {
	"Precursor_type": "prec_type",
	"Spectrum_type": "spec_type",
	"PrecursorMZ": "prec_mz",
	"Instrument_type": "inst_type",
	"Collision_energy": "col_energy",
	"Ion_mode": "ion_mode",
	"Ionization": "ion_type",
	"ID": "spec_id",
	"Collision_gas": "col_gas",
	"Pressure": "pressure",
	"Num peaks": "num_peaks",
	"MW": "mw",
	"ExactMass": "exact_mass",
	"CASNO": "cas_num",
	# "NISTNO": "dset_spec_id",
	"Name": "name",
	"MS": "peaks",
	"SMILES": "smiles",
	"Rating": "rating",
	"Frag_mode": "frag_mode",
	"Instrument": "inst",
	"RI": "ri",
	# "DB#": "dset_spec_id_2",
	"Notes": "notes", # NIST only
	"Formula": "formula",
	"InChIKey": "inchikey",
	# these are place holders for compatibility
	'Ramped': 'ramped',
	'Stepped': 'stepped',
	'Collision_energy_2':'col_energy_extra_1', # for ramped
	'Collision_energy_3':'col_energy_extra_2', # for stepped
}

MS_KEY_DICT = {
	">compound": "name",
	# ">ionization": "prec_type",
	# ">formula": "formula",
	">parentmass": "prec_mz",
	# "#smiles": "smiles",
	"#instrumentation": "inst_type",
	# ">collision": "col_energy",
}
META_KEYS = list(MS_KEY_DICT.keys())
SPEC_KEYS = [">ms1peaks",">ms1merged",">ms2peaks",">ms2merged",">collision"]

def extract_info_from_comments(comments,key):

	start_idx = comments.find(key)
	if start_idx == -1:
		return None
	start_idx += len(key)+1 # +1 is for =
	end_idx = start_idx+1
	cur_char = comments[end_idx]
	while cur_char != "\"":
		end_idx += 1
		cur_char = comments[end_idx]
	value = comments[start_idx:end_idx]
	return value


"""
Convert data from MSMS database format to pandas dataframe with JSON
No type conversions or filtering: all of that is done downstream
"""

def preproc_msp(msp_fp,keys,num_entries):
	""" """

	with open(msp_fp) as f:
		raw_data_lines = f.readlines()
	# split CAS# and NIST# on different lines
	_raw_data_lines = []
	for line in raw_data_lines:
		if "CAS#" in line and "NIST#" in line:
			assert ";" in line, line
			split_lines = line.split(";")
			_raw_data_lines.append(split_lines[0].rstrip(";")+"\n")
			_raw_data_lines.append(split_lines[1].lstrip())
		else:
			_raw_data_lines.append(line)
	raw_data_lines = _raw_data_lines
	raw_data_list = []
	raw_data_item = {key: None for key in keys}
	read_ms = False
	for raw_l in tqdm(raw_data_lines, desc=f"> processing {msp_fp}",total=len(raw_data_lines)):
		if num_entries > -1 and len(raw_data_list) == num_entries:
			break
		raw_l = raw_l.replace('\n', '')
		if raw_l == '':
			# check if double line
			if all(v is None for v in raw_data_item.values()):
				assert not read_ms	
			else:
				raw_data_list.append(raw_data_item.copy())
				raw_data_item = {key: None for key in keys}
				read_ms = False
		elif read_ms:
			raw_data_item['MS'] = raw_data_item['MS'] + raw_l + '\n'
		else:
			if "RI:" in raw_l:
				raw_l_split = raw_l.split(':')
			else:
				raw_l_split = raw_l.split(': ')
			assert len(raw_l_split) >= 2, len(raw_l_split)
			key = raw_l_split[0]
			if key == "Num peaks" or key == "Num Peaks":
				assert len(raw_l_split) == 2, raw_l_split
				value = raw_l_split[1]
				raw_data_item['Num peaks'] = int(value)
				raw_data_item['MS'] = ''
				read_ms = True
			elif key == "Comments":
				comments = ": ".join(raw_l_split[1:])
				smiles = extract_info_from_comments(comments,"computed SMILES")
				rating = extract_info_from_comments(comments,"MoNA Rating")
				frag_mode = extract_info_from_comments(comments,"fragmentation mode")
				if not (smiles is None):
					raw_data_item["SMILES"] = smiles
				if not (rating is None):
					raw_data_item["Rating"] = rating
				if not (frag_mode is None):
					raw_data_item["Frag_mode"] = frag_mode
			elif key in keys:
				value = raw_l_split[1]
				raw_data_item[key] = value
	if num_entries > -1:
		import pdb; pdb.set_trace()
	msp_df = pd.DataFrame(raw_data_list)
	# drop all-NaN rows
	msp_df = msp_df.dropna(axis=0,how="all")
 
	return msp_df

def preproc_nist_mol(mol_dp):
	""" read in all .MOL files and return a df """

	mol_fp_list = glob.glob(os.path.join(mol_dp,"*.MOL"))
	def proc_mol_file(mol_fp):
		modules = rdkit_import("rdkit.Chem","rdkit.Chem.rdinchi","rdkit.Chem.AllChem")
		Chem = modules[0]
		mol_fn = os.path.basename(os.path.normpath(mol_fp))
		spec_id = mol_fn.lstrip("ID").rstrip(".MOL")
		mol = Chem.MolFromMolFile(mol_fp, sanitize=True)
		if not (mol is None):
			smiles = Chem.MolToSmiles(mol)
		else:
			smiles = None
		entry = dict(
			spec_id=spec_id,
			smiles=smiles
		)
		return entry
	mol_fp_iter = tqdm(mol_fp_list, desc="> proc_mol_files", total=len(mol_fp_list))
	mol_df_entries = par_apply(mol_fp_iter, proc_mol_file)
	mol_df = pd.DataFrame(mol_df_entries)
	return mol_df

def merge_and_check(msp_df,mol_df,rename_dict):

	# get rid of the columns that you don't care about
	msp_bad_cols = set(msp_df.columns)-set(rename_dict.keys())
	msp_df = msp_df.drop(columns=msp_bad_cols)
	# rename to be consistent
	msp_df = msp_df.rename(columns=rename_dict)
	if mol_df is None:
		assert not msp_df["smiles"].isna().all()
		msp_df.loc[:,"spec_id"] = np.arange(msp_df.shape[0])
		spec_df = msp_df
	else:
		assert msp_df["smiles"].isna().all()
		assert not msp_df["spec_id"].isna().all()
		# merge with mol on spec_id
		msp_df = msp_df.drop(columns=["smiles"])
		spec_df = pd.merge(msp_df,mol_df,how="inner",on="spec_id")
	if "dset_spec_id" not in spec_df.columns:
		spec_df.loc[:,"dset_spec_id"] = spec_df["spec_id"]
	print("> num spec: ", spec_df.isna().sum())
	spec_df = spec_df.reset_index(drop=True)
	return spec_df


def preproc_ms_files(ms_dp,ms_meta_fp,keys,num_entries):

	# read in meta file
	ms_meta_df = pd.read_csv(ms_meta_fp,sep="\t")
	ms_meta_df = ms_meta_df.rename(columns={"ionization":"prec_type","spec":"dset_spec_id_base"}).drop(columns=["name"])
	assert not ms_meta_df.isna().any().any(), ms_meta_df.isna().any()
	# get all .ms files
	ms_fp_list = glob.glob(os.path.join(ms_dp,"*.ms"))
	if num_entries > -1:
		ms_fp_list = ms_fp_list[:num_entries]
	# read in all .ms files
	def proc_ms_file(ms_fp):
		ms_fn = os.path.basename(os.path.normpath(ms_fp))
		dset_spec_id = ms_fn.removesuffix(".ms")
		with open(ms_fp,'r') as f:
			ms_lines = f.readlines()
		ms_meta_entry = {}
		ms_levels, ms_ces, ms_peakses = [], [], []
		cur_level, cur_ce, cur_peaks = None, None, []
		in_spec = False
		for ms_line in ms_lines:
			ms_line = ms_line.strip()
			if in_spec:
				if ms_line == "":
					in_spec = False
					ms_levels.append(cur_level)
					ms_ces.append(cur_ce)
					ms_peakses.append("\n".join(cur_peaks))
					cur_level, cur_ce, cur_peaks = None, None, []
				else:
					mz_ints = ms_line
					cur_peaks.append(mz_ints)
			else:
				for meta_key in META_KEYS:
					if ms_line.startswith(meta_key):
						value = ms_line.removeprefix(meta_key+" ")
						# if meta_key == ">ionization":
						# 	value = value.replace(" ","")
						ms_meta_entry[MS_KEY_DICT[meta_key]] = value
				for spec_key in SPEC_KEYS:
					if ms_line.startswith(spec_key):
						in_spec = True
						if ms_line.startswith(">ms1peaks") or ms_line.startswith(">ms1merged"):
							cur_level = 1
						else:
							cur_level = 2
						if ms_line.startswith(">collision"):
							cur_ce = ms_line.removeprefix(">collision ")
		# check if still in_spec
		if in_spec:
			in_spec = False
			ms_levels.append(cur_level)
			ms_ces.append(cur_ce)
			ms_peakses.append("\n".join(cur_peaks))
			cur_level, cur_ce, cur_peaks = None, None, []
		# flatten entries
		ms_entries = []
		for idx, (ms_level, ms_ce, ms_peaks) in enumerate(zip(ms_levels,ms_ces,ms_peakses)):
			if ms_level == 2:
				ms_entry = dict(
					dset_spec_id_base=dset_spec_id, # for debugging
					dset_spec_id=dset_spec_id+f"_{idx}",
					col_energy=ms_ce,
					peaks=ms_peaks,
					**ms_meta_entry
				)
				ms_entries.append(ms_entry)
		assert len(ms_entries) > 0, (ms_fp, ms_lines)
		return ms_entries
	ms_fp_iter = tqdm(ms_fp_list,desc="> proc_ms_files",total=len(ms_fp_list))
	spec_df_entries = seq_apply(ms_fp_iter,proc_ms_file)
	spec_df_entries = flatten_lol(spec_df_entries)
	spec_df = pd.DataFrame(spec_df_entries)
	# drop all-NaN rows
	spec_df = spec_df.dropna(axis=0,how="all")
	# add metadata
	spec_df = spec_df.merge(ms_meta_df[["dset_spec_id_base","prec_type","formula","smiles"]],on=["dset_spec_id_base"],how="inner")
	# add ion_mode
	spec_df.loc[:,"ion_mode"] = "P"
	# add spec_type
	spec_df.loc[:,"spec_type"] = "MS2"
	# add spec_id (this will be relabeled later)
	spec_df.loc[:,"spec_id"] = np.arange(len(spec_df))
	# add extra columns for compatibility
	for value in MSP_KEY_DICT.values():
		if value not in spec_df.columns:
			spec_df.loc[:,value] = np.nan
   
	# drop unnecessary columns
	spec_df = spec_df.drop(columns=["dset_spec_id_base"])
	spec_df = spec_df.reset_index(drop=True)
	return spec_df

PCDL_KEYS = {
	"ACCESSION:" : "dset_spec_id",
	"CH$SMILES:" : "smiles",
	"CH$IUPAC: InChI" : "inchi",
	"CH$NAME:" : "name",
	"CH$FORMULA:" : "formula",
	"CH$LINK: INCHIKEY" : "inchikey",
	"CH$EXACT_MASS:" : "exact_mass",
	"AC$MASS_SPECTROMETRY: ION_MODE" : "ion_mode",
	"AC$MASS_SPECTROMETRY: COLLISION_ENERGY" : "col_energy",
	"AC$INSTRUMENT_TYPE: " : "inst_type",
	"AC$MASS_SPECTROMETRY: MS_TYPE" : "spec_type",
	"MS$FOCUSED_ION: PRECURSOR_M/Z" : "prec_mz",
	"MS$FOCUSED_ION: PRECURSOR_TYPE" : "prec_type",
	"PK$NUM_PEAK:" : "num_peaks",
}

def read_pcdl(zipped_spectra_dp: str):
	
	def parse_pcdl(text_list: list[str], id_prefix = None):
		data_dict = {}
		for row in text_list:
			str_row = row.decode("utf-8")
			if str_row.startswith(" "):
				mz, rel_int, _ = [float(i) for i in str_row.split()]
				if "peaks" not in data_dict:
					data_dict["peaks"] = ""
				data_dict["peaks"] += f"{mz} {rel_int}\n"
			else:
				for key, col in PCDL_KEYS.items():
					if str_row.startswith(key):
						value = str_row.removeprefix(key).strip()
						if col in ["exact_mass",'prec_mz']:
							value = float(value)
						elif col == 'col_energy':
							value = f"{value} eV"
						elif col  == "ion_mode" and value == "POSITIVE":
							value = "P"
						elif col  == "ion_mode" and value == "NEGATIVE":
							value = "N"
						elif col == "num_peaks":
							value = int(value)
						elif col == "smiles": #in case of chemaxon smiles eg.CC1C[C@H](C)C[C@H](OC) |t:46|
							value = value.split()[0]
						data_dict[col] = value
		data_dict["dset_spec_id"] = f"{id_prefix}_{data_dict['dset_spec_id']}"
		return data_dict

	spectrum_list = []
	print(zipped_spectra_dp)
	zipped_spectra_filenames = [f for f in os.listdir(zipped_spectra_dp) if f.endswith(".zip")]
	for zipped_spectra_filename in zipped_spectra_filenames:
		zipped_spectra_fp = f"{zipped_spectra_dp}/{zipped_spectra_filename}"
		spectra_libname = zipped_spectra_filename.removesuffix(".zip")
		print(f"> parse zip file {zipped_spectra_fp}")
		archive = zipfile.ZipFile(zipped_spectra_fp, 'r')
		for file in archive.namelist():
			with archive.open(file) as f:
				text_list = f.readlines()
				spectrum_list.append(parse_pcdl(text_list, spectra_libname))
	
	print(f"> {len(spectrum_list)} spectra parsed")
	spec_df = pd.DataFrame(spectrum_list)
	# this is one off, we need fix this
	spec_df = spec_df.drop(spec_df[~spec_df["prec_type"].isin(["[M+H]+","[M-H]-"])].index)
	# fill some extrat meta data
	spec_df.loc[:,"frag_mode"] = "CID"
	spec_df.loc[:,"notes"] = ""
	selected_idx = spec_df.groupby(["inchikey","prec_type","col_energy"])['num_peaks'].idxmax()
	spec_df = spec_df.loc[selected_idx]
	spec_df.loc[:,"spec_id"] = np.arange(spec_df.shape[0])

	# add ion_type
	spec_df.loc[:,"ion_type"] = "ESI"
	# add extra columns for compatibility
	for value in MSP_KEY_DICT.values():
		if value not in spec_df.columns:
			spec_df.loc[:,value] = np.nan
   
	print(f"> {len(spec_df['inchikey'].unique())} unique inchikey")
	return spec_df

def process_ms_gym(ms_gym_tsv_fp, subset):
	spec_df = pd.read_csv(ms_gym_tsv_fp, sep="\t")
 
	# only use on in simulation_challenge
	spec_df = spec_df.sort_values(by='simulation_challenge', ascending=False)
	print(f"> processs ms gym file {ms_gym_tsv_fp}")
	print(f"> using ms gym subset {subset}")
	print("> if you are doing simulation challenge make sure use only simulation_challenge subset")
 
	if subset == 'simulation_challenge':
		spec_df = spec_df[spec_df['simulation_challenge']]
		print(f"> {len(spec_df)} rows selected for subset {subset}")
	elif subset == 'extra':
		spec_df = spec_df[~spec_df['simulation_challenge']]
		print(f"> {len(spec_df)} rows selected for subset {subset}")
	elif subset != 'all':
		print(f">warningm unknown subset [{subset}] to msygm. Every row will be included")
  
	spec_df["peaks"] = spec_df.apply(lambda x: "\n".join([f"{m} {i}" for m,i in zip(x["mzs"].split(","),x["intensities"].split(","))]),axis=1)
	spec_df = spec_df.drop(columns=["mzs","intensities"])
	spec_df.rename(columns={"precursor_mz":"prec_mz", 
						   "adduct":"prec_type",
						   "precursor_mz" :"prec_mz",
						   "parent_mass":"exact_mass",
						   "instrument_type":"inst_type",
						   "collision_energy":"col_energy",
						   "identifier":"dset_spec_id"},
				  	inplace=True)
 
	# NOTE: MSGYM discaled all ramped and stepped spectra
	# thus this does not matter
	spec_df["col_energy"], spec_df["normalized"], spec_df["ramped"] = zip(*spec_df["col_energy"].apply(parse_mass_gym_ce_str))

	#spec_df = spec_df.astype({"prec_mz":float,"exact_mass":float})

	# add ion_mode
	spec_df.loc[:,"ion_mode"] = "P"
	# add spec_type
	spec_df.loc[:,"spec_type"] = "MS2"
	# add ion_type
	spec_df.loc[:,"ion_type"] = "ESI"
	# add extra columns for compatibility
	for value in MSP_KEY_DICT.values():
		if value not in spec_df.columns:
			spec_df.loc[:,value] = np.nan
	split_df = spec_df[["dset_spec_id","fold"]]

	return spec_df, split_df

SPECTRAVERSE_MGF_KEY_DICT = {
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
	'COLLISION_ENERGY_1': 'col_energy',
	'COLLISION_ENERGY_2': 'col_energy_extra_1',
	'COLLISION_ENERGY_3': 'col_energy_extra_2',
	#'NORMALIZED_COLLISION_ENERGY_1': 'normalized_col_energy_1',
	#'NORMALIZED_COLLISION_ENERGY_2': 'normalized_col_energy_2',
	#'NORMALIZED_COLLISION_ENERGY_3': 'normalized_col_energy_3',
	'FOLD': 'fold',
	'INDEX': 'spec_id',
	'SOURCE': 'source',
	'TITLE': 'title',
	'NUM_PEAKS': 'num_peaks',
}

def process_specverse(mgf_fp):
	spectra_list = []
	with open(mgf_fp) as mgf_in:
		data_dict = None
		for row in tqdm(mgf_in, desc=f"> process {mgf_fp}"):
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
				for key, col in SPECTRAVERSE_MGF_KEY_DICT.items():
					if row.startswith(key):
						value = row.split('=',1)[1].strip()
						# in case of empty values
						if value in ['nan','N/A','None','null','']:
							value = np.nan
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
						elif col.startswith("col_energy"):
							if not value is np.nan:
								value = float(value.split()[0]) 
						data_dict[col] = value
      
	spec_df = pd.DataFrame(spectra_list)
	spec_df.loc[:,"fold"] = spec_df.loc[:,"fold"].astype(float).astype(int)
	spec_df.loc[:, 'ramped'] = (~spec_df['col_energy'].isna()) & (~spec_df['col_energy_extra_1'].isna()) & (spec_df['col_energy_extra_2'].isna())
	spec_df.loc[:, 'stepped'] = (~spec_df['col_energy'].isna()) & (~spec_df['col_energy_extra_1'].isna()) & (~spec_df['col_energy_extra_2'].isna())
	# setup dset
	spec_df.loc[:,"dset"] = "spectraverse"
	spec_df.loc[:,"dset_spec_id"] = spec_df["spec_id"]
	
	split_df = spec_df[["spec_id","fold"]]
 
 	# add extra columns for compatibility
	for value in MSP_KEY_DICT.values():
		if value not in spec_df.columns:
			spec_df.loc[:,value] = np.nan
   
	return spec_df, split_df

if __name__ == "__main__":

	## TODO: add support for NPLLIB
	parser = argparse.ArgumentParser()
	parser.add_argument('--msp_file', type=str, required=False)
	parser.add_argument('--mol_dp', type=str, required=False)
	parser.add_argument('--pkl_file', type=str, required=False)
	parser.add_argument('--ms_dp', type=str, required=False)
	parser.add_argument('--ms_meta_file', type=str, required=False)
	parser.add_argument('--mgf_fp', type=str, required=False)
	parser.add_argument('--input_format', type=str, choices=["msp","msp+mol","ms+meta", "msp+pkl", "ms_gym", "ms_gym_extra", "pcdl_zip", "spectraverse"], required=True)
	parser.add_argument('--output_name', type=str, required=True)
	parser.add_argument('--output_dp', type=str, default='data/df')
	parser.add_argument('--num_entries', type=int, default=-1)
	parser.add_argument('--output_format', type=str, default="json", choices=["json","csv"])
	parser.add_argument('--msp_dset_spec_id', type=str, default="NISTNO", choices=["NISTNO","DB#","NIST#","ID"])
	parser.add_argument('--ms_gym_tsv', type=str, required=False)
	parser.add_argument('--pcdl_dp', type=str, required=False)
	args = parser.parse_args()

	os.makedirs(args.output_dp,exist_ok=True)

	if args.input_format in ["msp","msp+mol","msp+pkl"]:
		# select dset_spec_id
		if args.msp_dset_spec_id not in MSP_KEY_DICT:
			MSP_KEY_DICT[args.msp_dset_spec_id] = "dset_spec_id"
		else:
			assert "dset_spec_id" not in MSP_KEY_DICT.values()
		# nist or mona
		msp_fp = args.msp_file
		assert os.path.isfile(msp_fp), msp_fp
		msp_df = preproc_msp(msp_fp,MSP_KEY_DICT.keys(),args.num_entries)
		if args.input_format == "msp+mol":
			mol_dp = args.mol_dp
			assert os.path.isdir(mol_dp), mol_dp
			mol_df = preproc_nist_mol(mol_dp)
		elif args.input_format == "msp+pkl":
			pkl_fp = args.pkl_file
			assert os.path.isfile(pkl_fp), pkl_fp
			nist_id_map = pd.read_pickle(pkl_fp)
			nist_id_map	= nist_id_map.rename(columns={"nist_id":args.msp_dset_spec_id,"smiles":"SMILES"})
			msp_df = msp_df.drop(columns=["SMILES"]).merge(nist_id_map,on=[args.msp_dset_spec_id],how="left")
			mol_df = None
		else:
			mol_df = None
		spec_df = merge_and_check(msp_df,mol_df,MSP_KEY_DICT)
	elif args.input_format == "ms+meta":
		# npllib
		# just use NISTNO as default
		MSP_KEY_DICT["NISTNO"] = "dset_spec_id"
		ms_dp = args.ms_dp
		assert os.path.isdir(ms_dp), ms_dp
		ms_meta_fp = args.ms_meta_file
		assert os.path.isfile(ms_meta_fp), ms_meta_fp
		spec_df = preproc_ms_files(ms_dp,ms_meta_fp,MSP_KEY_DICT.keys(),args.num_entries)
	elif args.input_format == "ms_gym":
		assert os.path.isfile(args.ms_gym_tsv), args.ms_gym_tsv
		spec_df, split_df = process_ms_gym(args.ms_gym_tsv, 'simulation_challenge')
		split_df_fp = os.path.join(args.output_dp,f"{args.output_name}_fold.csv")
		split_df.to_csv(split_df_fp,index=False)
	elif args.input_format == "ms_gym_extra":
		assert os.path.isfile(args.ms_gym_tsv), args.ms_gym_tsv
		spec_df, split_df = process_ms_gym(args.ms_gym_tsv, 'extra')
		split_df_fp = os.path.join(args.output_dp,f"{args.output_name}_fold.csv")
		split_df.to_csv(split_df_fp,index=False)
	elif args.input_format == "pcdl_zip":
		assert os.path.isdir(args.pcdl_dp), args.pcdl_dp
		spec_df = read_pcdl(args.pcdl_dp)
	elif args.input_format == "spectraverse":
		assert os.path.isfile(args.mgf_fp), args.mgf_fp
		spec_df, split_df  = process_specverse(args.mgf_fp)
		split_df_fp = os.path.join(args.output_dp,f"{args.output_name}_fold.csv")
		split_df.to_csv(split_df_fp,index=False)
	else:
		raise ValueError(f"Invalid input format: {args.input_format}")
	
	# save files
	spec_df_fp = os.path.join(args.output_dp,f"{args.output_name}_df.{args.output_format}")
	print(f"> saving spec df to {spec_df_fp}")
	if args.output_format == "json":
		spec_df.to_json(spec_df_fp)
	else:
		assert args.output_format == "csv"
		spec_df.to_csv(spec_df_fp,index=False)